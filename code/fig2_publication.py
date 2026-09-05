# -*- coding: utf-8 -*-
"""Figure 2 出版级：全球水动力 regime 空间是连续谱（四联证据）。

(a) PC1 分布 + 偶极检验（单峰）
(b) PC1–PC2 二维密度（单一连通质量体，无孤立盆地）
(c) GMM BIC 改善曲线：真实数据无拐点 vs 单峰参照
(d) 众数持续分析：平滑尺度 σ≥3 后众数归 1
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from plotstyle import setup_plot
setup_plot()

BASE = Path(__file__).parent
OUT = BASE / "output" / "regime_space"

plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "font.size": 9})

# 众数持续分析结果（源自 multimodality_summary.txt，regime_space.py 实证输出）
MODE_PERSIST = [(0.5, 85), (1.0, 15), (1.5, 7), (2.0, 3), (3.0, 1),
                (4.0, 1), (6.0, 1), (8.0, 1), (12.0, 1), (16.0, 1),
                (24.0, 1)]


def panel_label(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")


def main():
    pc = pd.read_parquet(OUT / "pc_scores.parquet")
    dip = json.loads((OUT / "pca_dip.json").read_text(encoding="utf-8"))
    bic = pd.read_csv(OUT / "gmm_bic.csv", encoding="utf-8-sig")
    var_exp = np.array(dip["explained"]) * 100

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        left=0.09, right=0.97, top=0.94, bottom=0.09)

    # (a) PC1 分布 + 偶极检验
    ax = axes[0, 0]
    ax.hist(pc.PC1, bins=220, density=True, color="#4a90b8", alpha=0.85,
            lw=0)
    xs = np.linspace(pc.PC1.min(), pc.PC1.max(), 500)
    from scipy.stats import norm
    ax.plot(xs, norm.pdf(xs, pc.PC1.mean(), pc.PC1.std()),
            color="#c0392b", lw=1.4, ls="--", label="unimodal Gaussian fit")
    d1 = dip["dip"]["PC1"]
    ax.set_xlabel(f"PC1 — regime structure ({var_exp[0]:.1f}% var.)",
                  fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"Hartigan dip = {d1['dip_full']:.4f}, p ≈ 1.0 "
                 f"(unimodal)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    panel_label(ax, "a")

    # (b) PC1–PC2 二维密度
    ax = axes[0, 1]
    h = ax.hist2d(pc.PC1, pc.PC2, bins=260, cmap="viridis",
                  norm=matplotlib.colors.LogNorm(), rasterized=True)
    ax.set_xlabel(f"PC1 — regime structure ({var_exp[0]:.1f}% var.)",
                  fontsize=9)
    ax.set_ylabel(f"PC2 — water-level variability ({var_exp[1]:.1f}% var.)",
                  fontsize=9)
    ax.set_title("One contiguous probability mass — no isolated basins",
                 fontsize=9)
    cb = fig.colorbar(h[3], ax=ax, pad=0.02)
    cb.set_label("reaches per cell (log)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    panel_label(ax, "b")

    # (c) GMM BIC：收敛拟合（n_init=10）真实 vs 高斯参照 + 重尾 t 参照
    ax = axes[1, 0]
    for data, color, lab, sty in [
            ("real", "#c0392b", "observed regime space", "o-"),
            ("unimodal_ref", "#999999",
             "unimodal Gaussian reference", "^-"),
            ("t_ref", "#4a90b8",
             "kurtosis-matched unimodal t reference", "s-")]:
        sub = bic[bic.data == data].sort_values("k")
        imp = (sub.bic.iloc[0] - sub.bic) / sub.bic.iloc[0] * 100
        ax.plot(sub.k, imp, sty, color=color, ms=4, lw=1.4, label=lab)
    ax.set_xlabel("number of GMM components k", fontsize=9)
    ax.set_ylabel("BIC improvement over k = 1 (%)", fontsize=9)
    ax.set_title("BIC gains rise smoothly, no elbow —\n"
                 "nested overlapping components, not separated modes",
                 fontsize=9)
    ax.set_xticks(range(1, 9))
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    panel_label(ax, "c")

    # (d) 众数持续分析
    ax = axes[1, 1]
    sig = [s for s, _ in MODE_PERSIST]
    nm = [n for _, n in MODE_PERSIST]
    ax.plot(sig, nm, "o-", color="#6a51a3", ms=4.5, lw=1.4)
    ax.axhline(1, color="#bbbbbb", lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("KDE bandwidth σ (histogram-bin units)", fontsize=9)
    ax.set_ylabel("number of local modes", fontsize=9)
    ax.set_title("Mode persistence: a single mode survives\n"
                 "for all σ ≥ 3", fontsize=9)
    ax.annotate("1 mode", xy=(3, 1), xytext=(0.7, 42),
                fontsize=8.5, color="#6a51a3",
                arrowprops=dict(arrowstyle="->", color="#6a51a3", lw=1))
    panel_label(ax, "d")

    fig.suptitle("")
    for ext in ("png", "pdf"):
        p = OUT / f"Fig2_continuum_evidence_出版级.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    facecolor="white")
        print("saved:", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
