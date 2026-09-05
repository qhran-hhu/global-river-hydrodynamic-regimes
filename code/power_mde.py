# -*- coding: utf-8 -*-
"""A3 功效分析（意见书问题 3）：MDE 最小可检效应

两个问题：
(1) 富集 MDE：regime 空间某区域大小 n、基线坝占比 p0 下，80% 功效、
    α=0.05（及 Bonferroni 0.01）可检出的最小坝富集倍数（精确二项检验）。
    → "坝不造新类"断言的功效边界。
(2) 位移 MDE：坝邻近 vs 对照两组样本量与 f2 观测方差下，80% 功效
    可检出的最小 f2 位移（原始与 log10 口径，对照中位数相对量）。
    → 检验观测到的 −52.7% 是否远超功效极限（是则"清晰可见"断言成立）。

输出：output/human_activity/power_analysis_MDE.csv + 控制台汇总
  python power_mde.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "human_activity"

MATCH_DEG = 0.2
CONTROL_DEG = 0.5


def load_sets():
    """复现 dam_contrast.py 的坝邻近/对照集合（同逻辑同种子）。"""
    import shapefile
    from scipy.spatial import cKDTree

    fm_full = pd.read_parquet(BASE / "output" / "feature_matrix_v1.parquet")
    fm = pd.read_parquet(BASE / "output" / "feature_matrix_v1_qc.parquet")
    sf = shapefile.Reader(str(OUT / "GDAT" / "GDAT_data_v1" / "data"
                              / "GDAT_v1_dams.shp"))
    fields = [f[0] for f in sf.fields[1:]]
    df = pd.DataFrame([dict(zip(fields, r)) for r in sf.iterRecords()])
    df["lon"] = [s.points[0][0] if len(s.points) else np.nan
                 for s in sf.shapes()]
    df["lat"] = [s.points[0][1] if len(s.points) else np.nan
                 for s in sf.shapes()]
    dams = df.dropna(subset=["lon", "lat"])

    qc_ids = set(fm.index)
    # 与 dam_contrast.py 一致：坝先匹配到全矩阵最近 reach，再按 QC 过滤
    pts = fm_full[["x", "y"]].values.copy()
    pts[:, 0] *= np.cos(np.deg2rad(pts[:, 1]))
    tree = cKDTree(pts)
    dpts = dams[["lon", "lat"]].values.copy()
    dpts[:, 0] *= np.cos(np.deg2rad(dpts[:, 1]))
    dist, idx = tree.query(dpts)
    dams["reach_id"] = fm_full.index[idx]
    dams["dist_deg"] = dist
    matched = dams[dams.dist_deg <= MATCH_DEG]
    dam_reaches = fm_full.loc[matched.reach_id.unique()]
    dam_reaches = dam_reaches[dam_reaches.index.isin(qc_ids)]

    qpts = fm_full[["x", "y"]].values.copy()
    qpts[:, 0] *= np.cos(np.deg2rad(qpts[:, 1]))
    dd = cKDTree(dpts).query(qpts)[0]
    control_pool = fm_full[dd > CONTROL_DEG]
    control_pool = control_pool[control_pool.index.isin(qc_ids)]
    ctrl_idx = []
    rng = np.random.default_rng(0)
    for _, row in dam_reaches.iterrows():
        cand = control_pool[(control_pool.continent == row.continent)
                            & (abs(control_pool.y - row.y) <= 5)]
        if len(cand) >= 5:
            ctrl_idx.append(rng.choice(cand.index.values))
    ctrl = control_pool.loc[pd.unique(np.asarray(ctrl_idx))]
    return dam_reaches, ctrl, dams, matched


def mde_enrich(n, p0, alpha=0.05, power=0.80):
    """精确二项：区域大小 n、基线 p0，双侧 α 下 80% 功效的最小富集倍数。"""
    cl = binom.ppf(alpha / 2, n, p0)          # X <= cl-1 拒绝（下尾）
    ch = binom.ppf(1 - alpha / 2, n, p0)      # X >= ch+1 拒绝（上尾）
    for r in np.arange(1.005, 50, 0.005):
        p1 = min(r * p0, 0.999999)
        pw = binom.cdf(cl - 1, n, p1) + binom.sf(ch, n, p1)
        if pw >= power:
            return float(r)
    return np.nan


def main():
    dam, ctrl, dams, matched = load_sets()
    n1, n0 = len(dam), len(ctrl)
    p0 = n1 / len(pd.read_parquet(BASE / "output" / "feature_matrix_v1_qc.parquet"))
    print(f"坝邻近 n1={n1}, 对照 n0={n0}, 全样本基线坝占比 p0={p0*100:.2f}%")

    # ---- (1) 富集 MDE：按诊断子样本中观测到的区域规模 ----
    regions = [("最小诊断区域 (UMAP 子样本)", 2678),
               ("簇0", 5485), ("簇2", 5955), ("簇1", 10159),
               ("最大诊断区域 (簇3)", 15723),
               ("假想小区域 n=1000", 1000),
               ("假想小区域 n=500", 500)]
    rows = []
    for name, n in regions:
        r05 = mde_enrich(n, p0, 0.05)
        r01 = mde_enrich(n, p0, 0.01)
        rows.append(dict(区域=name, n=n, MDE_富集倍数_a05=round(r05, 2),
                         MDE_富集倍数_a01=round(r01, 2)))
        print(f"  {name} (n={n}): MDE={r05:.2f}x (α=0.05), "
              f"{r01:.2f}x (α=0.01)")
    t = pd.DataFrame(rows)

    # ---- (2) 位移 MDE：f2 两组比较 ----
    a = dam.f2_rel_range.dropna().values
    b = ctrl.f2_rel_range.dropna().values
    za, zb = 1.959964, 0.841621   # z(0.975), z(0.80)
    for tag, xa, xb, unit in [
            ("raw", a, b, "f2"),
            ("log10", np.log10(np.clip(a, 1e-5, None)),
             np.log10(np.clip(b, 1e-5, None)), "log10 f2")]:
        sp = np.sqrt(((len(xa) - 1) * xa.var(ddof=1)
                      + (len(xb) - 1) * xb.var(ddof=1))
                     / (len(xa) + len(xb) - 2))
        mde = (za + zb) * sp * np.sqrt(1 / len(xa) + 1 / len(xb))
        obs = np.median(xa) - np.median(xb)
        print(f"  f2 ({tag}): SD={sp:.4f}, MDE={mde:.5f} {unit}, "
              f"观测中位差={obs:.4f} ({abs(obs)/mde:.0f}x MDE)")
        t.loc[len(t)] = dict(区域=f"f2位移({tag})", n=f"{n1}v{n0}",
                             MDE_富集倍数_a05=round(mde, 5),
                             MDE_富集倍数_a01=round(abs(obs) / mde, 1))
    # log10 口径换算为百分比位移
    la = np.log10(np.clip(a, 1e-5, None))
    lb = np.log10(np.clip(b, 1e-5, None))
    sp = np.sqrt(((len(la) - 1) * la.var(ddof=1)
                  + (len(lb) - 1) * lb.var(ddof=1)) / (len(la) + len(lb) - 2))
    mde_log = (za + zb) * sp * np.sqrt(1 / len(la) + 1 / len(lb))
    print(f"  即 80% 功效可检出 {((10**mde_log)-1)*100:.1f}% 的相对位移；"
          f"观测 −52.7% 约为其 {0.527/((10**mde_log)-1):.0f} 倍")

    # ---- (3) "新 regime 岛"情景 ----
    # 坝造新类 → 该区域坝占比应 ≫ 基线（例如 >50%，富集 ~9x）
    isl = mde_enrich(2678, p0, 0.01)
    print(f"  最小诊断区域 (n=2678, α=0.01) 可检富集 {isl:.2f}x；"
          f"坝专属类型岛（坝占比>50% ≈ {0.5/p0:.0f}x 富集）"
          f"超出功效边界 {(0.5/p0)/isl:.0f} 倍 → 不可能漏检")

    t.to_csv(OUT / "power_analysis_MDE.csv", index=False, encoding="utf-8-sig")
    print(f"输出 -> {OUT / 'power_analysis_MDE.csv'}")


if __name__ == "__main__":
    main()
