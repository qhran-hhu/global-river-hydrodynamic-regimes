# -*- coding: utf-8 -*-
"""特征矩阵 v0：用 Hydrocron 试拉的 369 个 reach 时间序列检验 f1–f7。

内容：
  1. 构建特征矩阵（features.py 路径甲；f6 坡度 = momma_slope 缺省时用
     Hydrocron 观测中位坡度）
  2. 物理合理性端到端检查：f5 谐波相位 → 峰值月份 vs 纬度
     （北半球峰值应在 6–9 月，南半球在 12–3 月）
  3. 快速聚类（Ward, k=4/6）+ 全球分布图
  4. 特征相关性热图

输出：code/output/feature_matrix_v0/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from features import build_feature_matrix, clean_series, MIN_OBS

try:
    from plotstyle import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

PILOT = BASE / "output" / "ts_shards_pilot"
META = BASE / "output" / "global_reach_list.csv"
OUT = BASE / "output" / "feature_matrix_v0"
OUT.mkdir(exist_ok=True)


def main():
    # 1. 载入试拉时间序列
    ts = pd.concat([pd.read_parquet(f)
                    for f in sorted(PILOT.glob("shard_*.parquet"))],
                   ignore_index=True)
    print(f"时间序列：{ts.reach_id.nunique()} reach，{len(ts)} 行")

    # 2. reach 表：坐标 + 坡度先验（momma_slope 缺省→Hydrocron 中位坡度）
    meta = pd.read_csv(META, usecols=["reach_id", "continent", "x", "y",
                                      "river_name", "momma_width",
                                      "momma_slope", "ice_clim_f"])
    meta = meta[meta.reach_id.isin(ts.reach_id.unique())].copy()
    slopes = {}
    for rid, g in ts.groupby("reach_id"):
        s = pd.to_numeric(g.slope, errors="coerce")
        s = s[(s > 0) & (s < 1)]
        slopes[rid] = float(s.median()) if len(s) >= 5 else np.nan
    meta["slope_hydro"] = meta.reach_id.map(slopes)
    meta["slope"] = meta.momma_slope.where(
        meta.momma_slope.notna() & (meta.momma_slope > 0),
        meta.slope_hydro)
    print(f"坡度先验覆盖：momma {meta.momma_slope.notna().sum()}，"
          f"Hydrocron 补充 {meta.slope.notna().sum() - meta.momma_slope.notna().sum()}，"
          f"仍缺 {meta.slope.isna().sum()}")

    # 3. 特征矩阵（路径甲）
    fm = build_feature_matrix(ts, meta[["reach_id", "slope"]],
                              slope_col="slope", path="A")
    fm = fm.join(meta.set_index("reach_id")[["continent", "x", "y",
                                             "river_name", "momma_width",
                                             "ice_clim_f"]])
    print(f"\n特征矩阵：{fm.shape[0]} reach × {fm.shape[1]} 列")
    print("各特征有效率：")
    for c in ["f1_level_rank", "f2_rel_range", "f3_event_resp",
              "f4_iqr_logh", "f5_phase", "f6_slope", "f7_width_slope"]:
        print(f"  {c}: {fm[c].notna().mean()*100:.0f}%")
    fm.to_csv(OUT / "feature_matrix_v0.csv", encoding="utf-8-sig")

    # 4. 物理检查：峰值月份 vs 纬度（phase=0 对应 1 月 1 日峰值）
    #    只信 R²≥0.3 的相位（年循环信号弱的 reach 相位是噪声）
    fm["peak_month"] = fm.f5_phase / (2 * np.pi) * 12 + 1
    R2_MIN = 0.3
    valid = fm.dropna(subset=["f5_phase", "y"])
    strong = valid[valid.f5_r2 >= R2_MIN]
    nh = strong[strong.y > 15]
    sh = strong[strong.y < -15]
    print(f"\n相位检查（R²≥{R2_MIN}，{len(strong)}/{len(valid)} reach 通过）："
          f"北半球(n={len(nh)})峰值月中位 {nh.peak_month.median():.1f}，"
          f"南半球(n={len(sh)})峰值月中位 {sh.peak_month.median():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(valid[valid.f5_r2 < R2_MIN].y,
               valid[valid.f5_r2 < R2_MIN].peak_month,
               s=10, alpha=0.25, c="gray", label=f"R²<{R2_MIN}（相位不可信）")
    ax.scatter(strong.y, strong.peak_month, s=16, alpha=0.75,
               c=np.where(strong.y > 0, "#c0504d", "#4bacc6"))
    ax.axvline(0, color="k", lw=0.5)
    for m in (7, 1.5):
        ax.axhline(m, color="g", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("纬度")
    ax.set_ylabel("WSE 峰值月份（f5 谐波相位）")
    ax.set_title("端到端物理检查：北半球应≈6–9月，南半球应≈12–3月")
    ax.set_ylim(0.5, 12.5)
    ax.legend(loc="upper right", fontsize=8)

    # 5. 快速聚类 + 全球分布（f5 仅对 R²≥0.3 的 reach 启用，其余置中性值）
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.preprocessing import StandardScaler
    fcols = ["f1_level_rank", "f2_rel_range", "f3_event_resp",
             "f4_iqr_logh", "f6_slope"]
    d = fm.dropna(subset=fcols).copy()
    # f5 相位转周期编码；R² 不足时置 0（中性）
    use5 = d.f5_r2 >= R2_MIN
    d["f5_sin"] = np.where(use5, np.sin(d.f5_phase), 0.0)
    d["f5_cos"] = np.where(use5, np.cos(d.f5_phase), 0.0)
    feats = d[["f1_level_rank", "f2_rel_range", "f3_event_resp",
               "f4_iqr_logh", "f5_sin", "f5_cos",
               "f6_slope", "f7_width_slope"]].copy()
    feats["f6_slope"] = np.log10(d.f6_slope.clip(lower=1e-6))
    feats["f7_width_slope"] = np.log10(
        d.f7_width_slope.clip(lower=1e-12).fillna(1e-12))
    feats = feats.fillna(feats.median())
    X = StandardScaler().fit_transform(feats)
    k = 6
    d["cluster"] = fcluster(linkage(X, method="ward"), k,
                            criterion="maxclust")
    ax = axes[1]
    sc = ax.scatter(d.x, d.y, c=d.cluster, s=16, cmap="tab10", alpha=0.8)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title(f"Ward 聚类（k={k}）全球分布（n={len(d)}）")
    plt.colorbar(sc, ax=ax, label="类")
    fig.tight_layout()
    fig.savefig(OUT / "v0_physical_check_clustering.png", bbox_inches="tight", dpi=150)

    d.to_csv(OUT / "feature_matrix_v0_clustered.csv", encoding="utf-8-sig")
    print("\n各类规模：", d.cluster.value_counts().sort_index().to_dict())
    print(f"\n输出目录：{OUT}")


if __name__ == "__main__":
    main()
