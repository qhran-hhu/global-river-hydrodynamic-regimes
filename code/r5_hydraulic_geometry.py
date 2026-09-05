# -*- coding: utf-8 -*-
"""R5：水力学几何标度桥 —— 站点水力几何指数 β 的全球图谱与标度律

物理模型：at-a-station hydraulic geometry w = a·h^β，配合 Manning（宽浅河
Q ∝ w·h^(5/3)·S^0.5）得 WSE 对流量的转换增益 dlogh/dlogQ = 1/(β+5/3)。
β 高的河道（宽度响应快于水深）水位对流量不敏感 → β 的跨河差异在机制上
稀释流量-水动力耦合。

计算：逐 reach 对 log(width)~log(h_rel) 做 OLS（h_rel = wse - P5(wse)），
要求 n≥30 且 h_rel 变程 >0.05 m（保证拟合可辨识）。
输出：output/regime_space/R5_hydraulic_geometry_beta.parquet（reach_id, beta, beta_r2,
      a, n, h_range, w_med, s_med）
用法：python r5_hydraulic_geometry.py [--s0 N --s1 M --tag _partN_M]
      python r5_hydraulic_geometry.py --analyze   （合并+标度律检验）
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SHARDS = BASE / "output" / "ts_shards_global"
OUT = BASE / "output" / "regime_space"
MIN_N = 30
MIN_HRANGE = 0.05  # m


def beta_one(doy, wse, wid, s_med):
    v = np.isfinite(wse) & np.isfinite(wid) & (wid > 0)
    if v.sum() < MIN_N:
        return None
    wse, wid = wse[v], wid[v]
    p5 = np.nanpercentile(wse, 5)
    h = wse - p5
    hv = h > 1e-3
    if hv.sum() < MIN_N:
        return None
    h, w = h[hv], wid[hv]
    h_range = np.nanpercentile(h, 90) - np.nanpercentile(h, 10)
    if h_range < MIN_HRANGE:
        return None
    lx, ly = np.log(h), np.log(w)
    X = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(beta=float(coef[1]), beta_r2=r2, a=float(np.exp(coef[0])),
                n=len(h), h_range=float(h_range),
                w_med=float(np.median(w)),
                s_med=float(s_med) if np.isfinite(s_med) else np.nan)


def run(s0, s1, tag):
    files = sorted(SHARDS.glob("shard_*.parquet"))
    if s1:
        files = files[s0:s1]
    print(f"分片 {len(files)} 个", flush=True)
    recs = {}
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["reach_id", "time_str", "wse",
                                         "width", "slope"])
        df = df[(df.time_str != "no_data") & df.time_str.notna()]
        df["date"] = pd.to_datetime(df.time_str, utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        for c in ("wse", "width", "slope"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df.wse <= -1e11, "wse"] = np.nan
        df.loc[df.width <= -1e11, "width"] = np.nan
        df["doy"] = df.date.dt.dayofyear.values / 365.25
        for rid, x in df.groupby("reach_id"):
            sv = x.slope[(x.slope > 0) & (x.slope < 1)]
            r = beta_one(x.doy.values, x.wse.values, x.width.values,
                         sv.median() if len(sv) >= 5 else np.nan)
            if r:
                recs[rid] = r
        if (i + 1) % 100 == 0:
            print(f"  shard {i + 1}/{len(files)}, 已收 {len(recs)}", flush=True)
    R = pd.DataFrame(recs).T
    R.index.name = "reach_id"
    R.to_parquet(OUT / f"R5_hydraulic_geometry_beta{tag}.parquet")
    print(f"有效 beta: {len(R)}", flush=True)


def analyze():
    from scipy.stats import spearmanr
    parts = sorted(OUT.glob("R5_hydraulic_geometry_beta_part*.parquet"))
    R = pd.concat([pd.read_parquet(p) for p in parts])
    R = R[~R.index.duplicated(keep="first")]
    R.to_parquet(OUT / "R5_hydraulic_geometry_beta.parquet")
    print(f"合并 {len(R)} reach 有 beta", flush=True)

    qc = pd.read_parquet(BASE / "output" / "feature_matrix_v1_qc.parquet")
    qc = qc.reset_index()
    d = qc.merge(R, on="reach_id", how="inner")
    good = d[d.beta_r2 >= 0.3]  # 拟合质量过滤
    print(f"QC 内且有 beta: {len(d)}; beta_r2>=0.3: {len(good)}", flush=True)

    rep = ["R5 水力学几何标度桥（2026-09-03）", "",
           f"β 可辨识 reach（全球）: {len(R)}; QC 内 {len(d)}, "
           f"其中拟合 R²≥0.3 的 {len(good)}",
           f"β 分布（R²≥0.3）: 中位 {good.beta.median():.2f}, "
           f"IQR [{good.beta.quantile(.25):.2f}, {good.beta.quantile(.75):.2f}], "
           f"P5-P95 [{good.beta.quantile(.05):.2f}, {good.beta.quantile(.95):.2f}]",
           ""]
    # β 与坡度/河宽的标度关系
    for c, lab in [("f6_slope", "坡度"), ("w_med", "河宽"),
                   ("f2_rel_range", "WSE变幅 f2"), ("f3_event_resp", "宽度响应 f3")]:
        v = np.isfinite(good.beta) & np.isfinite(good[c])
        rep.append(f"β ~ {lab}: Spearman ρ = "
                   f"{spearmanr(good.beta[v], good[c][v]).statistic:+.3f} "
                   f"(n={v.sum()})")
    rep.append("")
    # 标度律检验（GSIM 配对）：f2 ≈ q2/(β+5/3)？
    g = pd.read_csv(OUT / "gsim_discharge_hydrodynamic_pairs.csv", encoding="utf-8-sig")
    m = g.merge(R.reset_index()[["reach_id", "beta", "beta_r2"]],
                on="reach_id", how="inner")
    m = m[m.beta_r2 >= 0.3]
    gain = 1.0 / (m.beta + 5.0 / 3.0)
    pred = m.q2 * gain
    r_raw = spearmanr(m.f2_rel_range, m.q2).statistic
    r_law = spearmanr(m.f2_rel_range, pred).statistic
    rep.append(f"=== 标度律检验（GSIM 配对, β 可辨识 n={len(m)}） ===")
    rep.append(f"f2 ~ q2（无形态校正）: ρ = {r_raw:+.3f}")
    rep.append(f"f2 ~ q2/(β+5/3)（标度律预测）: ρ = {r_law:+.3f}")
    # β 对 regime 坐标的解释力
    pc = pd.read_parquet(OUT / "driver_data.parquet").reset_index()
    dd = good.merge(pc[["reach_id", "PC1", "PC2", "PC3"]], on="reach_id")
    for c in ("PC1", "PC2", "PC3"):
        rep.append(f"β ~ {c}: ρ = "
                   f"{spearmanr(dd.beta, dd[c]).statistic:+.3f}")
    txt = "\n".join(rep)
    (OUT / "R5_scaling_law.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=int, default=0)
    ap.add_argument("--s1", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    else:
        run(a.s0, a.s1, a.tag or f"_part{a.s0}_{a.s1}")
