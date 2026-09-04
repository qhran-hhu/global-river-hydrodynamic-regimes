# -*- coding: utf-8 -*-
"""regime 空间主线：低维嵌入 + 多峰性检验（连续谱 vs 离散类的正式证据链）

数据：output/特征矩阵_v1_qc.parquet（97,566 reach，QC 修复后）
特征块（f2-f7 动力学特征，§8.4 决策：水平维度不参与）：
  f2_rel_range, f3_event_resp, f4_iqr_logh,
  f5 = log1p(amp)*(cos phase, sin phase)  —— 相位为圆周量，用振幅加权向量；
      R²<0.3 被门控的 reach 取向量 (0,0)，物理含义"无显著年循环"，不丢样本
  log10(f6_slope), log10(f7_width_slope)
异常值极胖尾（f2 min=-3e4, max=1.9e4）→ 全块分位数正态变换（QuantileTransformer）
三步运行（各自 <300 s）：
  python regime_space.py pca   # 变换 + PCA + dip 检验
  python regime_space.py gmm   # GMM BIC 真实 vs 单峰参照（同边缘分布全协方差高斯）
  python regime_space.py umap  # UMAP 嵌入 + HDBSCAN + 众数持续分析 + 出图出表
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "regime_space"
OUT.mkdir(exist_ok=True)

try:
    from plotstyle import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

RNG = 0
NPC = 6  # 嵌入/检验用的 PC 数


def build_block():
    """构造特征块并分位数正态变换，返回 (Xn, meta DataFrame)。"""
    from sklearn.preprocessing import QuantileTransformer

    d = pd.read_parquet(BASE / "output" / "特征矩阵_v1_qc.parquet")
    f5x = np.log1p(d.f5_amp.clip(lower=0)) * np.cos(d.f5_phase)
    f5y = np.log1p(d.f5_amp.clip(lower=0)) * np.sin(d.f5_phase)
    X = pd.DataFrame({
        "f2": d.f2_rel_range.fillna(d.f2_rel_range.median()),
        "f3": d.f3_event_resp.fillna(d.f3_event_resp.median()),
        "f4": d.f4_iqr_logh.fillna(d.f4_iqr_logh.median()),
        "f5x": f5x.fillna(0.0),
        "f5y": f5y.fillna(0.0),
        "f6": np.log10(d.f6_slope.fillna(d.f6_slope.median()).clip(lower=1e-8)),
        "f7": np.log10(d.f7_width_slope.fillna(
            d.f7_width_slope.median()).clip(lower=1e-12)),
    }, index=d.index)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200_000, random_state=RNG)
    Xn = qt.fit_transform(X.values)
    meta = d[["continent", "x", "y", "f2_rel_range"]].copy()
    return Xn, meta


def step_pca():
    import diptest
    from sklearn.decomposition import PCA

    Xn, meta = build_block()
    pca = PCA(n_components=7, random_state=RNG)
    Z = pca.fit_transform(Xn)
    cols = [f"PC{i+1}" for i in range(7)]
    pc = pd.DataFrame(Z, index=meta.index, columns=cols)
    pc[["continent", "x", "y", "f2_rel_range"]] = meta
    pc.to_parquet(OUT / "pc_scores.parquet")
    ev = pca.explained_variance_ratio_
    print("PCA 方差贡献率:", np.round(ev, 3), "累计:", np.round(np.cumsum(ev), 3))

    # Hartigan 偶极检验：PC1/PC2 全量统计量 + 5 组 5000 子样本 p 值
    dip = {}
    for ax in ["PC1", "PC2"]:
        full = pc[ax].values
        stat_full = float(diptest.dipstat(full))
        ps, stats = [], []
        rng = np.random.default_rng(RNG)
        for _ in range(5):
            s = rng.choice(full, 5000, replace=False)
            st, p = diptest.diptest(s, boot_pval=True, n_boot=300, seed=RNG)
            ps.append(float(p))
            stats.append(float(st))
        dip[ax] = dict(dip_full=stat_full,
                       dip_sub_median=float(np.median(stats)),
                       p_median=float(np.median(ps)) if ps else None)
        print(ax, dip[ax])
    json.dump(dict(explained=ev.tolist(), dip=dip),
              open(OUT / "pca_dip.json", "w"), ensure_ascii=False, indent=1)


def step_gmm():
    from sklearn.mixture import GaussianMixture

    pc = pd.read_parquet(OUT / "pc_scores.parquet")
    cols = [f"PC{i+1}" for i in range(NPC)]
    rng = np.random.default_rng(RNG)
    sub = rng.choice(len(pc), 50_000, replace=False)
    X = pc[cols].values[sub]
    # 单峰参照：与数据同均值同全协方差的多元高斯（真单峰，同边缘结构）
    mu, cov = X.mean(0), np.cov(X.T)
    Xref = rng.multivariate_normal(mu, cov, size=len(X))

    rows = []
    for tag, Xd in [("real", X), ("unimodal_ref", Xref)]:
        for k in range(1, 9):
            g = GaussianMixture(k, covariance_type="full", max_iter=100,
                                n_init=1, tol=1e-3, random_state=RNG)
            g.fit(Xd)
            rows.append(dict(data=tag, k=k, bic=g.bic(Xd)))
            print(tag, k, f"BIC={rows[-1]['bic']:.0f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "gmm_bic.csv", index=False, encoding="utf-8-sig")
    # 每个 k 相对 k=1 的改善率
    for tag in ["real", "unimodal_ref"]:
        b = t[t.data == tag].set_index("k").bic
        print(tag, "BIC 相对 k=1 改善%:", ((b[1] - b) / b[1] * 100).round(1).to_dict())


def step_umap():
    import umap
    from scipy.ndimage import gaussian_filter, maximum_filter
    from sklearn.cluster import HDBSCAN

    pc = pd.read_parquet(OUT / "pc_scores.parquet")
    cols = [f"PC{i+1}" for i in range(NPC)]
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(pc), 40_000, replace=False)
    samp = pc.iloc[sub_idx]

    emb = umap.UMAP(n_neighbors=50, min_dist=0.05, n_components=2,
                    random_state=42).fit_transform(samp[cols].values)
    samp = samp.assign(U1=emb[:, 0], U2=emb[:, 1])

    lab = HDBSCAN(min_cluster_size=300, min_samples=30).fit_predict(emb)
    n_clu = len(set(lab) - {-1})
    noise = float((lab == -1).mean() * 100)
    sizes = pd.Series(lab[lab >= 0]).value_counts().sort_values(ascending=False)
    print(f"HDBSCAN: {n_clu} 个簇, 噪声点 {noise:.1f}%")
    print("前 10 大簇规模:", sizes.head(10).tolist())

    # 众数持续分析（scale-space）：PC1-PC2 二维直方图逐步加宽 KDE，数局部峰
    H, _, _ = np.histogram2d(pc.PC1, pc.PC2, bins=200)
    H = H / H.sum()
    modes = []
    for sig in [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]:
        S = gaussian_filter(H, sig)
        mx = maximum_filter(S, size=9)
        n_mode = int(((S == mx) & (S > S.max() * 0.05)).sum())
        modes.append(dict(sigma=sig, n_modes=n_mode))
    modes = pd.DataFrame(modes)
    print(modes.to_string(index=False))

    # 大坝邻近 reach 叠加
    dam = pd.read_csv(BASE / "output" / "human_activity" / "大坝_reach匹配.csv",
                      encoding="utf-8-sig")
    dam_ids = set(dam.loc[dam.qc_pass == True, "reach_id"])
    samp_dam = samp.index.isin(dam_ids)
    print(f"UMAP 子样本中坝邻近 reach: {samp_dam.sum()} / {len(samp)}")

    # ---------- 汇总表 ----------
    pca_dip = json.load(open(OUT / "pca_dip.json"))
    bic = pd.read_csv(OUT / "gmm_bic.csv")
    with open(OUT / "多峰性检验_汇总.txt", "w", encoding="utf-8") as f:
        f.write("全球河流水动力 regime 空间：多峰性检验汇总\n")
        f.write(f"样本: {len(pc)} reach (QC 后), 特征块 f2-f7 (7 维分位数正态)\n\n")
        f.write(f"[1] PCA 方差贡献率: {np.round(pca_dip['explained'], 3).tolist()}\n")
        for ax, r in pca_dip["dip"].items():
            f.write(f"[2] Hartigan 偶极检验 {ax}: dip={r['dip_full']:.4f}, "
                    f"子样本中位 p={r['p_median']}\n")
        f.write(f"[3] HDBSCAN (UMAP 2D, 40k 子样本): {n_clu} 簇, "
                f"噪声 {noise:.1f}%, 最大簇 {sizes.max() if len(sizes) else 0}\n")
        f.write("[4] 众数持续分析 (PC1-PC2 KDE):\n" + modes.to_string(index=False) + "\n")
        f.write("[5] GMM BIC (50k 子样本):\n" + bic.to_string(index=False) + "\n")

    # ---------- 图 ----------
    cont_colors = {"as": "#c0504d", "sa": "#9bbb59", "na": "#4bacc6",
                   "eu": "#8064a2", "af": "#e8a33d", "oc": "#4d3b2f"}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    ax = axes[0, 0]
    for c, g in samp.groupby("continent"):
        ax.scatter(g.U1, g.U2, s=0.3, alpha=0.35, color=cont_colors.get(c, "gray"),
                   label=c, rasterized=True)
    ax.legend(markerscale=8, fontsize=9, title="洲", title_fontsize=9)
    ax.set_title("(a) UMAP 嵌入 · 按洲着色")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    ax = axes[0, 1]
    v = np.log10(samp.f2_rel_range.clip(lower=1e-5))
    sc = ax.scatter(samp.U1, samp.U2, s=0.3, c=v, cmap="viridis", alpha=0.5,
                    rasterized=True)
    fig.colorbar(sc, ax=ax, label="log10 f2 水位相对变幅")
    ax.set_title("(b) UMAP · 按 f2 着色（检验是否梯度而非孤岛）")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    ax = axes[0, 2]
    ax.scatter(samp.U1, samp.U2, s=0.3, color="#cccccc", alpha=0.4, rasterized=True)
    ax.scatter(samp.U1[samp_dam], samp.U2[samp_dam], s=1.2, color="#c00000",
               alpha=0.7, rasterized=True,
               label=f"坝邻近 (n={samp_dam.sum()})")
    ax.legend(markerscale=5, fontsize=9)
    ax.set_title("(c) UMAP · GDAT 大坝邻近 reach 叠加")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    ax = axes[1, 0]
    ax.hist(pc.PC1, bins=200, density=True, color="#4bacc6", alpha=0.8)
    d1 = pca_dip["dip"]["PC1"]
    ax.set_title(f"(d) PC1 分布（方差贡献 {pca_dip['explained'][0]*100:.0f}%）\n"
                 f"偶极统计量={d1['dip_full']:.4f}, p≈{d1['p_median']}")
    ax.set_xlabel("PC1"); ax.set_ylabel("密度")

    ax = axes[1, 1]
    for tag, lab_, col in [("real", "真实数据", "#c0504d"),
                           ("unimodal_ref", "单峰高斯参照", "#999999")]:
        b = bic[bic.data == tag]
        b0 = b.bic.iloc[0]
        ax.plot(b.k, (b0 - b.bic) / b0 * 100, "o-", color=col, label=lab_)
    ax.set_xlabel("GMM 组分个数 k"); ax.set_ylabel("BIC 相对 k=1 改善（%）")
    ax.set_title("(e) GMM BIC：有无拐点？")
    ax.legend(fontsize=9)

    ax = axes[1, 2]
    ax.plot(modes.sigma, modes.n_modes, "o-", color="#8064a2")
    ax.set_xscale("log")
    ax.set_xlabel("KDE 带宽 σ（直方图格）"); ax.set_ylabel("局部众数个数")
    ax.set_title("(f) 众数持续分析：平滑尺度 vs 众数")
    ax.axhline(1, color="gray", ls=":", lw=0.8)

    fig.suptitle("全球河流水动力 regime 空间：连续谱证据链"
                 f"（n={len(pc):,} reach）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "连续谱证据.png", bbox_inches="tight", dpi=150)
    samp_out = samp.assign(hdb=lab)
    samp_out.to_parquet(OUT / "umap_sample.parquet")
    print(f"输出 -> {OUT}")


def step_blob():
    """剖析 UMAP/HDBSCAN 各团块的成分：是真 regime 岛还是数据/填补伪影。"""
    samp = pd.read_parquet(OUT / "umap_sample.parquet")
    full = pd.read_parquet(BASE / "output" / "特征矩阵_v1_qc.parquet")
    dam = pd.read_csv(BASE / "output" / "human_activity" / "大坝_reach匹配.csv",
                      encoding="utf-8-sig")
    dam_ids = set(dam.loc[dam.qc_pass == True, "reach_id"])
    m = samp.join(full[["n_obs", "f4_iqr_logh", "f6_slope", "f5_r2",
                        "dark_frac", "ice_clim_f", "river_name"]])
    m["dam"] = m.index.isin(dam_ids)

    rows = []
    for c, g in m.groupby("hdb"):
        top_cont = g.continent.value_counts(normalize=True).head(2)
        rows.append(dict(
            cluster=c, n=len(g),
            dam_pct=round(g.dam.mean() * 100, 1),
            f5_r2_med=round(g.f5_r2.median(), 3),
            f2_med=round(g.f2_rel_range.median(), 4),
            f4_med=round(g.f4_iqr_logh.median(), 3),
            slope_med=round(g.f6_slope.median(), 6),
            n_obs_med=int(g.n_obs.median()),
            dark_med=round(g.dark_frac.replace(-999, np.nan).median(), 2),
            ice_med=round(g.ice_clim_f.replace(-999, np.nan).median(), 2),
            top_continent=f"{top_cont.index[0]} {top_cont.iloc[0]*100:.0f}%",
        ))
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(t.to_string(index=False))
    t.to_csv(OUT / "团块成分剖析.csv", index=False, encoding="utf-8-sig")

    # 全样本 vs 各团块的坝密度富集倍数
    base = m.dam.mean()
    print(f"\n全样本坝邻近占比 {base*100:.1f}%")
    for c, g in m.groupby("hdb"):
        print(f"  簇{c}: {g.dam.mean()/base:.1f}x 富集")
    # 簇内代表性河流名（非 NODATA）
    for c, g in m.groupby("hdb"):
        names = g.river_name[(g.river_name != "NODATA") & g.river_name.notna()]
        if len(names):
            print(f"  簇{c} 代表河流:", names.value_counts().head(3).index.tolist())


def step_nof5():
    """稳健性：去掉 f5 相位向量（及 (0,0) 填补），仅用 f2,f3,f4,f6,f7 重做
    PCA + 偶极检验 + 众数持续分析，验证连续谱结论不依赖 f5 处理方式。"""
    import diptest
    from scipy.ndimage import gaussian_filter, maximum_filter
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer

    d = pd.read_parquet(BASE / "output" / "特征矩阵_v1_qc.parquet")
    X = pd.DataFrame({
        "f2": d.f2_rel_range.fillna(d.f2_rel_range.median()),
        "f3": d.f3_event_resp.fillna(d.f3_event_resp.median()),
        "f4": d.f4_iqr_logh.fillna(d.f4_iqr_logh.median()),
        "f6": np.log10(d.f6_slope.fillna(d.f6_slope.median()).clip(lower=1e-8)),
        "f7": np.log10(d.f7_width_slope.fillna(
            d.f7_width_slope.median()).clip(lower=1e-12)),
    }, index=d.index)
    Xn = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             random_state=RNG).fit_transform(X.values)
    Z = PCA(n_components=5, random_state=RNG).fit_transform(Xn)
    pca = PCA(n_components=5, random_state=RNG).fit(Xn)
    print("无 f5 PCA 方差贡献率:", np.round(pca.explained_variance_ratio_, 3))
    for i, ax in enumerate(["PC1", "PC2"]):
        full = Z[:, i]
        stat_full = float(diptest.dipstat(full))
        rng = np.random.default_rng(RNG)
        ps = [float(diptest.diptest(rng.choice(full, 5000, replace=False),
                                    boot_pval=True, n_boot=300, seed=RNG)[1])
              for _ in range(5)]
        print(f"无f5 {ax}: dip={stat_full:.4f}, 子样本中位 p={np.median(ps)}")
    H, _, _ = np.histogram2d(Z[:, 0], Z[:, 1], bins=200)
    H = H / H.sum()
    for sig in [1, 2, 3, 6, 12]:
        S = gaussian_filter(H, sig)
        mx = maximum_filter(S, size=9)
        print(f"无f5 众数: sigma={sig} -> "
              f"{int(((S == mx) & (S > S.max() * 0.05)).sum())}")


VARIANTS = {
    # 特征剔除稳健性矩阵（意见书问题 2：f4/f7 均含坡度 S，坡度自由集只剩 f2/f3/f5）
    "nof5":    ["f2", "f3", "f4", "f6", "f7"],
    "nof4":    ["f2", "f3", "f5x", "f5y", "f6", "f7"],
    "noslope": ["f2", "f3", "f5x", "f5y"],
}


def _raw_block(d):
    """与 build_block 同口径的原始特征块（未变换）。"""
    f5x = np.log1p(d.f5_amp.clip(lower=0)) * np.cos(d.f5_phase)
    f5y = np.log1p(d.f5_amp.clip(lower=0)) * np.sin(d.f5_phase)
    return pd.DataFrame({
        "f2": d.f2_rel_range.fillna(d.f2_rel_range.median()),
        "f3": d.f3_event_resp.fillna(d.f3_event_resp.median()),
        "f4": d.f4_iqr_logh.fillna(d.f4_iqr_logh.median()),
        "f5x": f5x.fillna(0.0),
        "f5y": f5y.fillna(0.0),
        "f6": np.log10(d.f6_slope.fillna(d.f6_slope.median()).clip(lower=1e-8)),
        "f7": np.log10(d.f7_width_slope.fillna(
            d.f7_width_slope.median()).clip(lower=1e-12)),
    }, index=d.index)


def step_robust():
    """特征剔除稳健性矩阵：nof5 / nof4 / noslope 三变体重跑
    PCA + Hartigan 偶极 + GMM BIC(真实 vs 单峰参照) + 众数持续分析，
    并用 noslope PC 重估坡度归因 R²（此时坡度完全外生，检验循环性影响）。
      python regime_space.py robust   # ~10 min
    """
    import diptest
    from scipy.ndimage import gaussian_filter, maximum_filter
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import QuantileTransformer

    d = pd.read_parquet(BASE / "output" / "特征矩阵_v1_qc.parquet")
    raw = _raw_block(d)
    rng = np.random.default_rng(RNG)
    rows, noslope_pc = [], None

    for tag, cols in VARIANTS.items():
        print(f"\n===== 变体 {tag}: {cols} =====", flush=True)
        Xn = QuantileTransformer(n_quantiles=1000,
                                 output_distribution="normal",
                                 random_state=RNG).fit_transform(raw[cols].values)
        pca = PCA(n_components=len(cols), random_state=RNG)
        Z = pca.fit_transform(Xn)
        ev = pca.explained_variance_ratio_
        print("方差贡献率:", np.round(ev, 3))
        if tag == "noslope":
            noslope_pc = pd.DataFrame(Z[:, :3], index=d.index,
                                      columns=["PC1", "PC2", "PC3"])

        # Hartigan 偶极检验 PC1/PC2
        for i, ax in enumerate(["PC1", "PC2"]):
            full = Z[:, i]
            stat_full = float(diptest.dipstat(full))
            ps = [float(diptest.diptest(rng.choice(full, 5000, replace=False),
                                        boot_pval=True, n_boot=300,
                                        seed=RNG)[1])
                  for _ in range(5)]
            rows.append(dict(variant=tag, test=f"dip_{ax}",
                             stat=round(stat_full, 4),
                             detail=f"子样本中位p={np.median(ps):.3f}"))
            print(f"{ax}: dip={stat_full:.4f}, 子样本中位p={np.median(ps):.3f}",
                  flush=True)

        # GMM BIC：真实 vs 同均值同协方差单峰参照（50k 子样本）
        sub = rng.choice(len(Z), 50_000, replace=False)
        Xs = Z[sub][:, :min(4, Z.shape[1])]
        mu, cov = Xs.mean(0), np.cov(Xs.T)
        Xref = rng.multivariate_normal(mu, cov, size=len(Xs))
        for dtag, Xd in [("real", Xs), ("unimodal_ref", Xref)]:
            bics = {}
            for k in range(1, 7):
                g = GaussianMixture(k, covariance_type="full", max_iter=100,
                                    n_init=1, tol=1e-3, random_state=RNG)
                g.fit(Xd)
                bics[k] = g.bic(Xd)
            imp2 = (bics[1] - bics[2]) / bics[1] * 100
            imp6 = (bics[1] - bics[6]) / bics[1] * 100
            rows.append(dict(variant=tag, test=f"gmm_{dtag}",
                             stat=round(imp2, 2),
                             detail=f"k=2改善{imp2:.2f}%, k=6改善{imp6:.2f}%"))
            print(f"GMM {dtag}: k=2 改善 {imp2:.2f}%, k=6 改善 {imp6:.2f}%",
                  flush=True)

        # 众数持续分析
        H, _, _ = np.histogram2d(Z[:, 0], Z[:, 1], bins=200)
        H = H / H.sum()
        cnts = []
        for sig in [1, 2, 3, 6, 12]:
            S = gaussian_filter(H, sig)
            mx = maximum_filter(S, size=9)
            cnts.append(int(((S == mx) & (S > S.max() * 0.05)).sum()))
        rows.append(dict(variant=tag, test="modes",
                         stat=cnts[0],
                         detail=f"σ=[1,2,3,6,12] -> {cnts}"))
        print(f"众数: σ=[1,2,3,6,12] -> {cnts}", flush=True)

    # ---- f5 门控点质量诊断：noslope 空间的表观多峰是否来自 (0,0) 填补 ----
    valid = d.f5_r2.notna()
    print(f"\nf5 门控比例: {(~valid).mean()*100:.1f}% reach 取向量 (0,0)",
          flush=True)
    cols = VARIANTS["noslope"]
    Xn = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             random_state=RNG).fit_transform(raw[cols].values)
    Zns = PCA(n_components=4, random_state=RNG).fit_transform(Xn)
    sub = rng.choice(len(Zns), 50_000, replace=False)
    g2 = GaussianMixture(2, covariance_type="full", max_iter=100, n_init=1,
                         tol=1e-3, random_state=RNG).fit(Zns[sub])
    lab2 = g2.predict(Zns[sub])
    for c in (0, 1):
        frac_valid = valid.values[sub][lab2 == c].mean()
        print(f"  组分{c}: n={int((lab2==c).sum())}, "
              f"非门控占比 {frac_valid*100:.1f}%", flush=True)
    rows.append(dict(variant="noslope_gatecheck", test="gmm2_vs_gate",
                     stat=round(float(valid.values[sub][lab2 == 0].mean()), 3),
                     detail=f"组分0非门控{valid.values[sub][lab2==0].mean()*100:.0f}%"
                            f"/组分1非门控{valid.values[sub][lab2==1].mean()*100:.0f}%"))

    # 仅非门控子集（f5 全部为实测谐波）重跑 noslope 全套
    print("\n===== 变体 noslope_nogate（仅 f5 非门控 reach） =====", flush=True)
    rv = raw[cols][valid.values]
    Xv = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             random_state=RNG).fit_transform(rv.values)
    Zv = PCA(n_components=4, random_state=RNG).fit_transform(Xv)
    print("n =", len(rv), "方差贡献率:",
          np.round(PCA(n_components=4, random_state=RNG).fit(Xv)
                   .explained_variance_ratio_, 3), flush=True)
    for i, ax in enumerate(["PC1", "PC2"]):
        full = Zv[:, i]
        stat_full = float(diptest.dipstat(full))
        ps = [float(diptest.diptest(rng.choice(full, 5000, replace=False),
                                    boot_pval=True, n_boot=300, seed=RNG)[1])
              for _ in range(5)]
        rows.append(dict(variant="noslope_nogate", test=f"dip_{ax}",
                         stat=round(stat_full, 4),
                         detail=f"子样本中位p={np.median(ps):.3f}"))
        print(f"{ax}: dip={stat_full:.4f}, 子样本中位p={np.median(ps):.3f}",
              flush=True)
    subv = rng.choice(len(Zv), min(50_000, len(Zv)), replace=False)
    Xs = Zv[subv]
    mu, cov = Xs.mean(0), np.cov(Xs.T)
    Xref = rng.multivariate_normal(mu, cov, size=len(Xs))
    for dtag, Xd in [("real", Xs), ("unimodal_ref", Xref)]:
        bics = {}
        for k in range(1, 7):
            g = GaussianMixture(k, covariance_type="full", max_iter=100,
                                n_init=1, tol=1e-3, random_state=RNG)
            g.fit(Xd)
            bics[k] = g.bic(Xd)
        imp2 = (bics[1] - bics[2]) / bics[1] * 100
        imp6 = (bics[1] - bics[6]) / bics[1] * 100
        rows.append(dict(variant="noslope_nogate", test=f"gmm_{dtag}",
                         stat=round(imp2, 2),
                         detail=f"k=2改善{imp2:.2f}%, k=6改善{imp6:.2f}%"))
        print(f"GMM {dtag}: k=2 改善 {imp2:.2f}%, k=6 改善 {imp6:.2f}%",
              flush=True)
    H, _, _ = np.histogram2d(Zv[:, 0], Zv[:, 1], bins=200)
    H = H / H.sum()
    cnts = []
    for sig in [1, 2, 3, 6, 12]:
        S = gaussian_filter(H, sig)
        mx = maximum_filter(S, size=9)
        cnts.append(int(((S == mx) & (S > S.max() * 0.05)).sum()))
    rows.append(dict(variant="noslope_nogate", test="modes",
                     stat=cnts[0], detail=f"σ=[1,2,3,6,12] -> {cnts}"))
    print(f"众数: σ=[1,2,3,6,12] -> {cnts}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "特征剔除稳健性矩阵.csv", index=False, encoding="utf-8-sig")

    # ---- 坡度归因循环性检验：noslope PC 下坡度完全外生 ----
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    drv = pd.read_parquet(OUT / "驱动因子数据.parquet")
    drv = drv.drop(columns=["PC1", "PC2", "PC3"]).join(noslope_pc)
    GROUPS_V = {"气候": ["bio1_年均温", "bio4_温度季节性",
                         "bio12_年降水", "bio15_降水季节性", "abs_lat"],
                "地形": ["slope"], "人类": ["dam"]}
    feats = sum(GROUPS_V.values(), [])
    sub = rng.choice(len(drv), 50_000, replace=False)
    dd = drv.iloc[sub]
    tr, te = train_test_split(np.arange(len(dd)), test_size=0.2, random_state=0)
    subsets = {"气候": GROUPS_V["气候"], "地形": GROUPS_V["地形"],
               "人类": GROUPS_V["人类"], "全部": feats}
    ar = []
    for tgt in ["PC1", "PC2", "PC3"]:
        y = dd[tgt].values.astype(float)
        for stag, cols in subsets.items():
            m = HistGradientBoostingRegressor(random_state=0)
            m.fit(dd[cols].values[tr], y[tr])
            r2 = float(r2_score(y[te], m.predict(dd[cols].values[te])))
            ar.append(dict(target=tgt, subset=stag, r2=round(r2, 3)))
            print(f"noslope归因 {tgt} {stag}: R²={r2:.3f}", flush=True)
    pd.DataFrame(ar).to_csv(OUT / "noslope归因_r2.csv", index=False,
                            encoding="utf-8-sig")
    print(f"输出 -> {OUT / '特征剔除稳健性矩阵.csv'}, noslope归因_r2.csv")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "pca"
    {"pca": step_pca, "gmm": step_gmm, "umap": step_umap,
     "blob": step_blob, "nof5": step_nof5, "robust": step_robust}[step]()
