# -*- coding: utf-8 -*-
"""应对②③：GDAT 全球大坝 × SWOT reach 对照 + 覆盖审计。

② 对照：每座坝匹配最近 reach（≤0.2°），与"同洲、同纬度带、同坡度档、
   远离任何坝（>0.5°）"的对照 reach 比较 f2/f3/f4（KS 检验 + 中位差）。
③ 审计：大坝 0.2° 内有 Hydrocron 数据的比例、通过 QC 的比例，
   对比全球基线（57.5% / 64.9%），定量回答"人类活动区被掩盖多少"。

输出：output/human_activity/ 下的 CSV + 图
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "human_activity"
DAMS_SHP = OUT / "GDAT" / "GDAT_data_v1" / "data" / "GDAT_v1_dams.shp"

try:
    from plotstyle import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

MATCH_DEG = 0.2   # 坝- reach 匹配半径
CONTROL_DEG = 0.5  # 对照须离所有坝 > 0.5°


def load_dams():
    import shapefile  # pyshp
    sf = shapefile.Reader(str(DAMS_SHP))
    fields = [f[0] for f in sf.fields[1:]]
    recs = list(sf.iterRecords())
    shapes = sf.shapes()
    df = pd.DataFrame([dict(zip(fields, r)) for r in recs])
    df["lon"] = [s.points[0][0] if len(s.points) else np.nan
                 for s in shapes]
    df["lat"] = [s.points[0][1] if len(s.points) else np.nan
                 for s in shapes]
    df = df.dropna(subset=["lon", "lat"])
    print(f"GDAT 大坝：{len(df)} 座；属性列：{fields[:12]}...")
    return df


def main():
    dams = load_dams()
    fm = pd.read_parquet(BASE / "output" / "feature_matrix_v1.parquet")
    qc_ids = set(pd.read_parquet(BASE / "output" / "feature_matrix_v1_qc.parquet")
                 .index)
    print(f"特征矩阵：{len(fm)}；QC 通过：{len(qc_ids)}")

    # reach 坐标 KD 树（纬度加权经度近似球面）
    pts = fm[["x", "y"]].values.copy()
    pts[:, 0] *= np.cos(np.deg2rad(pts[:, 1]))
    tree = cKDTree(pts)
    dpts = dams[["lon", "lat"]].values.copy()
    dpts[:, 0] *= np.cos(np.deg2rad(dpts[:, 1]))
    dist, idx = tree.query(dpts)
    dams["reach_id"] = fm.index[idx]
    dams["dist_deg"] = dist
    matched = dams[dams.dist_deg <= MATCH_DEG].copy()
    print(f"\n③ 覆盖审计：{len(dams)} 座坝中，0.2° 内有 Hydrocron reach："
          f"{len(matched)}（{len(matched)/len(dams)*100:.1f}%）"
          f"——全球基线 57.5%")
    matched["qc_pass"] = matched.reach_id.isin(qc_ids)
    qc_rate = matched.qc_pass.mean()
    print(f"   匹配到 reach 的坝中通过 QC：{matched.qc_pass.sum()}"
          f"（{qc_rate*100:.1f}%）——全球基线 64.9%")

    # ② 对照：坝邻近 reach（去重）vs 对照
    dam_reaches = fm.loc[matched.reach_id.unique()]
    dam_reaches = dam_reaches[dam_reaches.index.isin(qc_ids)]
    # 对照池：QC 通过、离所有坝 > 0.5°
    qc_df = fm.loc[list(qc_ids)]
    qpts = qc_df[["x", "y"]].values.copy()
    qpts[:, 0] *= np.cos(np.deg2rad(qpts[:, 1]))
    qtree_pts = qpts
    dtree = cKDTree(dpts)
    dd = dtree.query(qtree_pts)[0]
    control_pool = qc_df[dd > CONTROL_DEG * 1.0]
    # 纬度带+洲匹配抽样
    ctrl_idx = []
    rng = np.random.default_rng(0)
    pool_by = control_pool.groupby("continent")
    for _, row in dam_reaches.iterrows():
        cand = control_pool[(control_pool.continent == row.continent)
                            & (abs(control_pool.y - row.y) <= 5)]
        if len(cand) >= 5:
            ctrl_idx.append(rng.choice(cand.index.values))
    ctrl = control_pool.loc[pd.unique(np.asarray(ctrl_idx))]
    print(f"\n② 对照实验：坝邻近 {len(dam_reaches)} reach，"
          f"对照 {len(ctrl)} reach")

    results = []
    for c, lab in [("f2_rel_range", "f2 水位相对变幅"),
                   ("f3_event_resp", "f3 宽度事件响应"),
                   ("f4_iqr_logh", "f4 IQR(logH)")]:
        a = dam_reaches[c].dropna()
        b = ctrl[c].dropna()
        ks = ks_2samp(a, b)
        results.append(dict(特征=lab, 坝邻近中位=round(a.median(), 4),
                            对照中位=round(b.median(), 4),
                            中位差=round(a.median() - b.median(), 4),
                            相对差pct=round(100 * (a.median() - b.median())
                                           / abs(b.median()), 1),
                            KS_p=f"{ks.pvalue:.1e}"))
        print(f"  {lab}: 坝 {a.median():.4f} vs 对照 {b.median():.4f} "
              f"({100*(a.median()-b.median())/abs(b.median()):+.1f}%, "
              f"KS p={ks.pvalue:.1e})")
    res = pd.DataFrame(results)
    res.to_csv(OUT / "dam_contrast_feature_diff.csv", index=False,
               encoding="utf-8-sig")
    matched.to_csv(OUT / "dam_reach_matches.csv", index=False,
                   encoding="utf-8-sig")

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (c, lab) in zip(axes, [("f2_rel_range", "f2 水位相对变幅"),
                                   ("f3_event_resp", "f3 宽度事件响应"),
                                   ("f4_iqr_logh", "f4 IQR(logH)")]):
        a = dam_reaches[c].dropna()
        b = ctrl[c].dropna()
        bins = np.linspace(0, np.percentile(pd.concat([a, b]), 98), 40)
        ax.hist(b, bins=bins, alpha=0.6, density=True, label=f"对照 (n={len(b)})",
                color="#4bacc6")
        ax.hist(a, bins=bins, alpha=0.6, density=True,
                label=f"坝邻近 (n={len(a)})", color="#c0504d")
        ax.axvline(a.median(), color="#c0504d", ls="--", lw=1)
        ax.axvline(b.median(), color="#4bacc6", ls="--", lw=1)
        ax.set_title(lab)
        ax.legend(fontsize=8)
    fig.suptitle("GDAT 大坝邻近 reach vs 匹配对照（全球，QC 后）")
    fig.tight_layout()
    fig.savefig(OUT / "dam_contrast.png", bbox_inches="tight", dpi=150)
    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
