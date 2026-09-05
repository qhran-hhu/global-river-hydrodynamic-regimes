# -*- coding: utf-8 -*-
"""流量 regime（GSIM 观测）vs 水动力 regime（SWOT）：双空间正面对比

审稿问题 2 的对决：同一地点，流量 regime 与水动力 regime 是否一回事？
数据：GSIM 月指数（30,959 站，.mon 内存解析，不落盘）
     × 特征矩阵 v1 QC（97,566 reach）
匹配：cKDTree 最近邻 ≤0.1°；站点质量：≥10 年、缺测 ≤20%、剔除可疑坐标
特征（镜像设计）：
  q2=(P90-P10)/P50(Q) ~ f2;  q_iqr=IQR(log10Q) ~ f4;
  q_ev=(P95-P50)/P50 ~ f3;   q_phase/amp/r2 年谐波 ~ f5
分析：秩相关 / 相位差圆统计 / Q 空间偶极检验 / 差异地图
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from regime_map import robinson, draw_graticule
OUT = BASE / "output" / "regime_space"
GSIM = OUT / "gsim"

try:
    from plotstyle import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def harmonic_monthly(dates, q):
    """月均值年谐波：.mon 的 date 是月末，必须按月中赋值
    t = 年 + (月-0.5)/12（曾用 dayofyear+14，引入 +0.96 月系统偏移）。"""
    t = (dates.dt.year + (dates.dt.month - 0.5) / 12).values
    y = q.astype(float)
    m = np.isfinite(y)
    if m.sum() < 24:
        return np.nan, np.nan, np.nan
    t, y = t[m], y[m]
    X = np.column_stack([np.ones_like(t), np.cos(2 * np.pi * t),
                         np.sin(2 * np.pi * t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ph = float(np.arctan2(coef[2], coef[1]) % (2 * np.pi))
    amp = float(np.hypot(coef[1], coef[2]))
    yhat = X @ coef
    ss = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return ph, amp, float(ss)


def main():
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    import diptest
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer

    # ---------- 1. 站点筛选与匹配 ----------
    meta = pd.read_csv(GSIM / "catalog" / "GSIM_metadata.csv",
                       low_memory=False)
    susp = pd.read_csv(GSIM / "catalog" /
                       "GSIM_suspect_coordinates_stations.csv")
    meta = meta[~meta["gsim.no"].isin(susp["gsim.no"])]
    good = meta[(meta["year.no"] >= 10) &
                (meta["frac.missing.days"] <= 0.2)].copy()
    print(f"站点: {len(meta)} -> 质量过滤后 {len(good)}")

    full = pd.read_parquet(BASE / "output" / "feature_matrix_v1_qc.parquet")
    tree = cKDTree(full[["x", "y"]].values)
    dist, idx = tree.query(good[["longitude", "latitude"]].values)
    good["reach_id"] = full.index.values[idx]
    good["dist_deg"] = dist
    matched = good[good.dist_deg <= 0.1].copy()
    print(f"匹配到 reach (≤0.1°): {len(matched)} 站 -> "
          f"{matched.reach_id.nunique()} 个 reach")

    # ---------- 2. 解析 .mon 计算流量 regime 特征 ----------
    z = zipfile.ZipFile(GSIM / "GSIM_indices.zip")
    rows = []
    need = set(matched["gsim.no"])
    import io
    for i, gs in enumerate(matched["gsim.no"]):
        try:
            with z.open(f"GSIM_indices/TIMESERIES/monthly/{gs}.mon") as f:
                txt = f.read().decode("utf-8", "replace")
            txt = txt.replace('"', "").replace("\t", ",")
            t = pd.read_csv(io.StringIO(txt), comment="#",
                            parse_dates=["date"])
        except KeyError:
            continue
        q = t.MEAN.values
        q = q[np.isfinite(q) & (q > 0)]
        if len(q) < 24:
            continue
        p = np.percentile(q, [10, 25, 50, 75, 90, 95])
        ph, amp, r2 = harmonic_monthly(t.date, t.MEAN.values)
        rows.append(dict(gsim=gs, n_months=len(q),
                         q2=(p[4] - p[0]) / p[2],
                         q_iqr=np.log10(p[3]) - np.log10(p[1]),
                         q_ev=(p[5] - p[2]) / p[2],
                         q_phase=ph, q_amp=amp, q_r2=r2))
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i+1}/{len(matched)}", flush=True)
    qf = pd.DataFrame(rows).set_index("gsim")
    m = matched.set_index("gsim.no").join(qf).dropna(subset=["q2"])
    r = full[["f2_rel_range", "f3_event_resp", "f4_iqr_logh", "f5_phase",
              "f5_r2", "f6_slope", "continent", "x", "y"]]
    m = m.join(r, on="reach_id")
    m.to_csv(OUT / "gsim_discharge_hydrodynamic_pairs.csv", encoding="utf-8-sig")
    print(f"有效配对: {len(m)}")

    # ---------- 3. 对比分析 ----------
    def rho(a, b):
        v = np.isfinite(a) & np.isfinite(b)
        return spearmanr(a[v], b[v]).statistic, v.sum()

    r_f2_q2, n1 = rho(m.f2_rel_range, m.q2)
    r_f4_qi, n2 = rho(m.f4_iqr_logh, m.q_iqr)
    r_f3_qe, n3 = rho(m.f3_event_resp, m.q_ev)
    # 相位：双方 R² 都达标才比（圆差，月）
    both = m[(m.f5_r2 >= 0.3) & (m.q_r2 >= 0.3)]
    dph = ((both.f5_phase - both.q_phase + np.pi) % (2 * np.pi) - np.pi)
    dph_m = dph / (2 * np.pi) * 12
    within1 = float((dph_m.abs() <= 1).mean() * 100)
    print(f"Spearman: f2~q2 {r_f2_q2:.3f} (n={n1}); "
          f"f4~q_iqr {r_f4_qi:.3f} (n={n2}); f3~q_ev {r_f3_qe:.3f} (n={n3})")
    print(f"相位差 |Δ|≤1月: {within1:.1f}% (n={len(both)}), "
          f"中位差 {dph_m.median():.2f} 月")

    # Q 空间连续谱检验
    Q = m[["q2", "q_iqr", "q_ev"]].copy()
    Q["qx"] = np.log1p(m.q_amp) * np.cos(m.q_phase)
    Q["qy"] = np.log1p(m.q_amp) * np.sin(m.q_phase)
    Q[["qx", "qy"]] = Q[["qx", "qy"]].fillna(0)
    Qn = QuantileTransformer(n_quantiles=1000,
                             output_distribution="normal",
                             random_state=0).fit_transform(Q.values)
    Z = PCA(n_components=5, random_state=0).fit_transform(Qn)
    ev = PCA(n_components=5, random_state=0).fit(Qn).explained_variance_ratio_
    rng = np.random.default_rng(0)
    ndip = min(4000, len(Z))
    ps = [float(diptest.diptest(rng.choice(Z[:, 0], ndip, replace=False),
                                boot_pval=True, n_boot=300, seed=0)[1])
          for _ in range(5)]
    print(f"Q 空间 PCA 方差: {np.round(ev, 3)}; "
          f"Q-PC1 dip p 中位={np.median(ps):.3f}")

    # 差异度量：水动力变幅秩 - 流量变幅秩
    m["f4_rank"] = m.f4_iqr_logh.rank(pct=True)
    m["qi_rank"] = m.q_iqr.rank(pct=True)
    m["rank_diff"] = m.f4_rank - m.qi_rank

    # ---------- 4. 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    ax = axes[0, 0]
    hb = ax.hexbin(m.q_iqr.rank(pct=True), m.f4_iqr_logh.rank(pct=True),
                   gridsize=40, cmap="viridis", mincnt=1)
    ax.plot([0, 1], [0, 1], "r:", lw=1)
    fig.colorbar(hb, ax=ax, label="站点数")
    ax.set_xlabel("流量变幅秩 IQR(log Q)"); ax.set_ylabel("水动力变幅秩 IQR(log H)")
    ax.set_title(f"(a) 变幅：流量秩 vs 水动力秩\nSpearman ρ={r_f4_qi:.2f}")

    ax = axes[0, 1]
    hb = ax.hexbin(m.q2.rank(pct=True), m.f2_rel_range.rank(pct=True),
                   gridsize=40, cmap="viridis", mincnt=1)
    ax.plot([0, 1], [0, 1], "r:", lw=1)
    fig.colorbar(hb, ax=ax, label="站点数")
    ax.set_xlabel("流量相对变幅秩 (P90-P10)/P50")
    ax.set_ylabel("水位相对变幅秩 f2")
    ax.set_title(f"(b) 相对变幅：ρ={r_f2_q2:.2f}")

    ax = axes[0, 2]
    bins = np.linspace(-6, 6, 49)
    ax.hist(dph_m, bins=bins, color="#4bacc6", edgecolor="w", lw=0.3)
    ax.axvline(0, color="r", ls=":", lw=1)
    ax.set_xlabel("水位峰值月 - 流量峰值月（月）")
    ax.set_ylabel("站点数")
    ax.set_title(f"(c) 季节相位一致性（双方 R²≥0.3, n={len(both):,}）\n"
                 f"|Δ|≤1 月占 {within1:.0f}%")

    ax = axes[1, 0]
    draw_graticule(ax)
    sc = ax.scatter(*robinson(m.longitude.values, m.latitude.values),
                    c=m.rank_diff, cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                    s=4, lw=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label="水动力变幅秩 - 流量变幅秩")
    ax.set_title(f"(d) 差异地图：何处水动力≠流量（n={len(m):,} 站）")

    ax = axes[1, 1]
    ax.hist(Z[:, 0], bins=100, color="#8064a2", alpha=0.8)
    ax.set_title(f"(e) 流量 regime 空间 PC1（方差 {ev[0]*100:.0f}%）\n"
                 f"偶极检验 p≈{np.median(ps):.2f}——流量格局也是连续谱")
    ax.set_xlabel("Q-PC1"); ax.set_ylabel("站点数")

    ax = axes[1, 2]
    mm = m.dropna(subset=["bio"]) if "bio" in m else m
    cold = m[m.f6_slope <= m.f6_slope.median()]
    hot = m[m.f6_slope > m.f6_slope.median()]
    for g, lab, c in [(cold, f"平缓河道 (n={len(cold):,})", "#4bacc6"),
                      (hot, f"陡峻河道 (n={len(hot):,})", "#c0504d")]:
        v = np.isfinite(g.q_iqr) & np.isfinite(g.f4_iqr_logh)
        rr = spearmanr(g.q_iqr[v], g.f4_iqr_logh[v]).statistic
        ax.scatter(g.q_iqr.rank(pct=True), g.f4_iqr_logh.rank(pct=True),
                   s=2, alpha=0.15, color=c, rasterized=True,
                   label=f"{lab}, ρ={rr:.2f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.legend(markerscale=6, fontsize=9)
    ax.set_xlabel("流量变幅秩"); ax.set_ylabel("水动力变幅秩")
    ax.set_title("(f) 河道形态调节流量→水动力转换\n（平缓 vs 陡峻分组）")

    fig.suptitle("流量 regime（GSIM 观测）× 水动力 regime（SWOT）："
                 "同一地点的双空间对比", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "discharge_vs_hydrodynamic.png", bbox_inches="tight", dpi=150)

    with open(OUT / "discharge_vs_hydrodynamic_summary.txt", "w", encoding="utf-8") as f:
        f.write("流量 regime × 水动力 regime 对比汇总\n")
        f.write(f"站点质量过滤后 {len(good)}, 匹配 reach {len(matched)}, "
                f"有效配对 {len(m)}\n")
        f.write(f"Spearman f2~q2: {r_f2_q2:.3f}; f4~q_iqr: {r_f4_qi:.3f}; "
                f"f3~q_ev: {r_f3_qe:.3f}\n")
        f.write(f"相位 |Δ|<=1月: {within1:.1f}% (n={len(both)}), "
                f"中位 {dph_m.median():.2f} 月\n")
        f.write(f"Q 空间 PCA: {np.round(ev, 3).tolist()}, "
                f"dip p={np.median(ps):.3f}\n")
        cold_r = spearmanr(cold.q_iqr, cold.f4_iqr_logh,
                           nan_policy="omit").statistic
        hot_r = spearmanr(hot.q_iqr, hot.f4_iqr_logh,
                          nan_policy="omit").statistic
        f.write(f"分组: 平缓 ρ={cold_r:.3f}, 陡峻 ρ={hot_r:.3f}\n")
    print(f"输出 -> {OUT / 'discharge_vs_hydrodynamic.png'}")


if __name__ == "__main__":
    main()
