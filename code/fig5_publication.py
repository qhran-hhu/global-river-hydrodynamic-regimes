# -*- coding: utf-8 -*-
"""Figure 5 出版级：流量 regime × 水动力 regime——同一地点的双空间对比。

(a) 变幅解耦：IQR(logQ) 秩 vs IQR(logH) 秩（hexbin，ρ=0.13）
(b) 季节相位一致性：双方 R²≥0.3 子集相位差直方图（中位 0.04 月）
(c) 差异地图：水动力秩 − 流量秩（Robinson + 海岸线）
(d) 流量 regime 空间也是连续谱（Q-PC1 分布 + 偶极检验 p=0.55）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from plotstyle import setup_plot
setup_plot()
from regime_map import robinson, draw_graticule, rank01
from fig1_publication import coast_lines, draw_coast, robinson_frame

BASE = Path(__file__).parent
OUT = BASE / "output" / "regime_space"

plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "font.size": 9})


def panel_label(ax, s, dx=-0.13):
    ax.text(dx, 1.04, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")


def main():
    m = pd.read_csv(OUT / "gsim_discharge_hydrodynamic_pairs.csv", low_memory=False)

    fig = plt.figure(figsize=(11.4, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                          left=0.08, right=0.97, top=0.94, bottom=0.08)

    # (a) 变幅解耦 hexbin
    ax = fig.add_subplot(gs[0, 0])
    d = m[["q_iqr", "f4_iqr_logh"]].dropna()
    rq, rh = rank01(d.q_iqr), rank01(d.f4_iqr_logh)
    rho = spearmanr(d.q_iqr, d.f4_iqr_logh).statistic
    hb = ax.hexbin(rq, rh, gridsize=42, cmap="viridis", mincnt=1,
                   rasterized=True)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.9, ls="--")
    ax.set_xlabel("discharge variability rank — IQR(log Q)", fontsize=9)
    ax.set_ylabel("hydrodynamic variability rank — IQR(log H)", fontsize=9)
    ax.set_title(f"Amplitude decoupling: Spearman ρ = {rho:.2f} "
                 f"(n = {len(d):,})", fontsize=9)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("stations", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    panel_label(ax, "a")

    # (b) 相位差直方图
    ax_b = fig.add_subplot(gs[0, 1])
    ax = ax_b
    sub = m[(m.q_r2 >= 0.3) & (m.f5_r2 >= 0.3)]
    # 圆差（弧度→月），与 compare_q_regime.py 口径一致
    dph = ((sub.f5_phase - sub.q_phase + np.pi) % (2 * np.pi) - np.pi)
    ph = (dph / (2 * np.pi) * 12).dropna()
    ax.hist(ph, bins=np.arange(-6.25, 6.5, 0.5), color="#4a90b8",
            alpha=0.9, lw=0)
    ax.axvline(0, color="#888888", lw=0.9, ls=":")
    med = ph.median()
    within = (ph.abs() <= 1).mean() * 100
    ax.set_xlabel("WSE peak month − discharge peak month (months)",
                  fontsize=9)
    ax.set_ylabel("stations", fontsize=9)
    ax.set_title(f"Phase agreement: median Δ = {med:.2f} mo, "
                 f"{within:.0f}% within ±1 mo\n(both R² ≥ 0.3, "
                 f"n = {len(ph)}; high-latitude ice caveat applies)",
                 fontsize=8.8)
    panel_label(ax, "b")

    # (c) 差异地图
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor("#f4f7fa")
    draw_graticule(ax)
    segs = coast_lines()
    draw_coast(ax, segs, lw=0.3)
    dm = m[["x", "y", "q_iqr", "f4_iqr_logh"]].dropna()
    diff = rank01(dm.f4_iqr_logh) - rank01(dm.q_iqr)
    xs, ys = robinson(dm.x.values, dm.y.values)
    order = np.random.RandomState(7).permutation(len(dm))
    sc = ax.scatter(xs[order], ys[order], c=diff[order],
                    cmap="RdBu_r", vmin=-0.6, vmax=0.6, s=3.2, lw=0,
                    rasterized=True)
    robinson_frame(ax)
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, pad=0.015, shrink=0.85)
    cb.set_label("hydrodynamic rank − discharge rank", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_title(f"Where hydrodynamics ≠ discharge "
                 f"(n = {len(dm):,} paired stations)", fontsize=9.5, pad=4)
    panel_label(ax, "c", dx=-0.035)

    # Q 空间连续谱结论以文本锚在 (b) 右侧空白区
    ax_b.text(0.97, 0.97,
              "Flow-regime space itself is also a continuum:\n"
              "Q-space PC1 (58% var.) dip test p = 0.55",
              transform=ax_b.transAxes, fontsize=7.8, va="top", ha="right",
              bbox=dict(fc="white", ec="#bbbbbb", lw=0.6, alpha=0.95,
                        boxstyle="round,pad=0.3"))

    for ext in ("png", "pdf"):
        p = OUT / f"Fig5_流量对决_出版级.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    facecolor="white")
        print("saved:", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
