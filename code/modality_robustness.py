# -*- coding: utf-8 -*-
"""D1 构造无关的单峰稳健性（评审 R1）：

(a) 各原始特征的边缘分布单峰性：dip 统计量单调变换不变 → 直接对原始
    f2/f3/f4/f5amp/f6/f7 做偶极检验，回答"分位数变换是否消除了多峰性"。
(b) 线性稳健缩放（winsorize 0.5/99.5% + median/IQR，保边缘分布形状）
    重做 PCA + 全轴 dip + 众数持续分析，替代分位数正态变换。
(c) 主空间（分位数正态）dip 从 PC1/PC2 扩展到 PC1–PC6。
(d) f5 门控阈值扫描 R²=0.1–0.5：门控率连续变化 + f5_r2 分布本身无自然断点。
(e) HDBSCAN 直接在 PC 空间（不经 UMAP）：无稳定聚类。

输出：output/regime_space/构造无关稳健性.csv
  python modality_robustness.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "regime_space"
RNG = 0


def dip_full(x, tag, rows, n_sub=5):
    import diptest
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    stat = float(diptest.dipstat(x))
    rng = np.random.default_rng(RNG)
    ps = [float(diptest.diptest(rng.choice(x, min(5000, len(x)), replace=False),
                                boot_pval=True, n_boot=300, seed=RNG)[1])
          for _ in range(n_sub)]
    rows.append(dict(检验=tag, dip=round(stat, 4),
                     p中位=round(float(np.median(ps)), 3)))
    print(f"{tag}: dip={stat:.4f}, p中位={np.median(ps):.3f}", flush=True)


def main():
    from scipy.ndimage import gaussian_filter, maximum_filter
    from sklearn.decomposition import PCA

    d = pd.read_parquet(BASE / "output" / "特征矩阵_v1_qc.parquet")
    rows = []

    # (a) 原始边缘分布单峰性（dip 单调不变）
    print("== (a) 原始特征边缘 dip（变换不变，直接回答构造性质疑） ==")
    f5amp = np.log1p(d.f5_amp.clip(lower=0))
    for c, lab in [("f2_rel_range", "f2 原始"), ("f3_event_resp", "f3 原始"),
                   ("f4_iqr_logh", "f4 原始"), ("f6_slope", "f6 原始(log10)"),
                   ("f7_width_slope", "f7 原始(log10)")]:
        v = d[c].values.astype(float)
        if "log10" in lab:
            v = np.log10(np.clip(v, 1e-12, None))
        dip_full(v, lab, rows)
    dip_full(f5amp.values, "f5amp 原始(log1p)", rows)

    # (b) 线性稳健缩放空间：PCA + 全轴 dip + 众数
    print("== (b) winsorize+median/IQR 线性缩放（保边缘形状）PCA ==")
    f5x = (np.log1p(d.f5_amp.clip(lower=0)) * np.cos(d.f5_phase)).fillna(0.0)
    f5y = (np.log1p(d.f5_amp.clip(lower=0)) * np.sin(d.f5_phase)).fillna(0.0)
    X = pd.DataFrame({
        "f2": d.f2_rel_range.fillna(d.f2_rel_range.median()),
        "f3": d.f3_event_resp.fillna(d.f3_event_resp.median()),
        "f4": d.f4_iqr_logh.fillna(d.f4_iqr_logh.median()),
        "f5x": f5x, "f5y": f5y,
        "f6": np.log10(d.f6_slope.fillna(d.f6_slope.median())
                       .clip(lower=1e-8)),
        "f7": np.log10(d.f7_width_slope.fillna(d.f7_width_slope.median())
                       .clip(lower=1e-12))})
    Xw = X.copy()
    for c in X.columns:
        lo, hi = X[c].quantile([0.005, 0.995])
        Xw[c] = X[c].clip(lo, hi)
        med = Xw[c].median()
        iqr = Xw[c].quantile(0.75) - Xw[c].quantile(0.25)
        Xw[c] = (Xw[c] - med) / max(iqr, 1e-12)
    Z = PCA(n_components=7, random_state=RNG).fit_transform(Xw.values)
    ev = PCA(n_components=7, random_state=RNG).fit(Xw.values) \
        .explained_variance_ratio_
    print("方差贡献率:", np.round(ev, 3))
    for i in range(4):
        dip_full(Z[:, i], f"稳健缩放 PC{i+1}", rows)
    H, _, _ = np.histogram2d(Z[:, 0], Z[:, 1], bins=200)
    H = H / H.sum()
    cnts = []
    for sig in [1, 2, 3, 6, 12]:
        S = gaussian_filter(H, sig)
        mx = maximum_filter(S, size=9)
        cnts.append(int(((S == mx) & (S > S.max() * 0.05)).sum()))
    rows.append(dict(检验="稳健缩放 众数持续", dip=np.nan,
                     p中位=f"σ=[1,2,3,6,12] -> {cnts}"))
    print("众数: σ=[1,2,3,6,12] ->", cnts, flush=True)

    # (c) 主空间 dip 扩展 PC1–PC6
    print("== (c) 主分位数空间 dip PC1–PC6 ==")
    pc = pd.read_parquet(OUT / "pc_scores.parquet")
    for i in range(6):
        dip_full(pc[f"PC{i+1}"].values, f"主空间 PC{i+1}", rows)

    # (d) 门控阈值扫描 + f5_r2 分布无断点
    print("== (d) 门控阈值扫描 ==")
    r2 = d.f5_r2
    frac = []
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        g = (r2 < thr) | r2.isna()
        frac.append(f"{thr}: {g.mean()*100:.1f}%")
    print("门控率随阈值:", "; ".join(frac))
    rows.append(dict(检验="门控阈值扫描", dip=np.nan, p中位="; ".join(frac)))
    dip_full(r2.dropna().values, "f5_r2 分布(显著者)", rows)

    # (e) PC 空间直接 HDBSCAN
    print("== (e) PC 空间 HDBSCAN（不经 UMAP） ==")
    from sklearn.cluster import HDBSCAN
    cols = [f"PC{i+1}" for i in range(6)]
    rng = np.random.default_rng(42)
    sub = rng.choice(len(pc), 40_000, replace=False)
    lab = HDBSCAN(min_cluster_size=300, min_samples=30).fit_predict(
        pc[cols].values[sub])
    n_clu = len(set(lab) - {-1})
    noise = float((lab == -1).mean() * 100)
    sizes = pd.Series(lab[lab >= 0]).value_counts()
    top = sizes.head(3).tolist()
    print(f"HDBSCAN(PC空间): {n_clu} 簇, 噪声 {noise:.1f}%, 前3大 {top}")
    rows.append(dict(检验="PC空间HDBSCAN", dip=np.nan,
                     p中位=f"{n_clu}簇, 噪声{noise:.1f}%, 前3大{top}"))

    pd.DataFrame(rows).to_csv(OUT / "构造无关稳健性.csv", index=False,
                              encoding="utf-8-sig")
    print(f"输出 -> {OUT / '构造无关稳健性.csv'}")


if __name__ == "__main__":
    main()
