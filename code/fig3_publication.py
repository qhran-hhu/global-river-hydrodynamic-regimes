# -*- coding: utf-8 -*-
"""Figure 3 出版级：driver_attribution——循环性校正后，气候（而非地形）组织连续谱、人类≈0。

(a) 分组条：各驱动组单独 R²，全特征集（地形部分循环）vs 无坡度特征集（坡度外生）
(b) 单变量置换重要性（7 预测因子 × 3 PC）
(c) 气候空间（bio1 × bio15）中的 PC2（水位变幅轴）
(d) 事件主导型占比随降水季节性/年均温十分位的变化
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

C_CLIM, C_TERR, C_HUM, C_SHARED = "#4c9ed9", "#8fbf60", "#d95f4c", "#c8c8c8"

VAR_LABELS = {"bio1_年均温": "mean T (bio1)",
              "bio4_温度季节性": "T seasonality (bio4)",
              "bio12_年降水": "annual P (bio12)",
              "bio15_降水季节性": "P seasonality (bio15)",
              "abs_lat": "|latitude|",
              "slope": "slope",
              "dam": "dam proximity"}


def panel_label(ax, s):
    ax.text(-0.15, 1.04, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")


def main():
    r2 = json.loads((OUT / "driver_attribution_r2.json").read_text(encoding="gbk"))
    imp = pd.read_csv(OUT / "driver_attribution_permutation_importance.csv", encoding="utf-8-sig")
    drv = pd.read_parquet(OUT / "driver_data.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    fig.subplots_adjust(hspace=0.55, wspace=0.30,
                        left=0.10, right=0.97, top=0.93, bottom=0.10)

    # (a) 分组条：各驱动组单独 R²，全特征集（地形部分循环）vs 无坡度特征集（坡度外生）
    ax = axes[0, 0]
    pcs = ["PC1", "PC2", "PC3"]
    r2_ns = pd.read_csv(OUT / "noslope_attribution_r2.csv", encoding="utf-8-sig")

    def g_full(p, tag):
        return r2[p]["r2"][tag]

    def g_ns(p, tag):
        return float(r2_ns[(r2_ns.target == p) &
                           (r2_ns.subset == tag)].r2.iloc[0])

    groups = [("气候", "climate", C_CLIM), ("地形", "terrain", C_TERR),
              ("人类", "human", C_HUM)]
    w = 0.13
    for i, p in enumerate(pcs):
        for j, (tag, lab, col) in enumerate(groups):
            x0 = i + (j - 1) * 2.4 * w
            v_full = g_full(p, tag)
            v_ns = g_ns(p, tag)
            ax.bar(x0 - w / 2 - 0.008, v_full, w, color=col, alpha=0.35,
                   hatch="//", edgecolor="white", lw=0,
                   label=f"{lab}, full set" if i == 0 else None)
            ax.bar(x0 + w / 2 + 0.008, v_ns, w, color=col,
                   label=f"{lab}, slope-free" if i == 0 else None)
    ax.annotate("terrain's apparent R² is circular\n(slope enters f6/f7 "
                "algebraically);\nslope-free set makes it exogenous",
                xy=(0.16, 0.79), xytext=(0.42, 0.66), fontsize=7.3,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"))
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{p}\n({v}% var.)" for p, v in
                        zip(pcs, [34.7, 23.5, 16.4])], fontsize=8.5)
    ax.set_ylabel("standalone R² (held-out)", fontsize=9)
    ax.set_ylim(0, 0.95)
    ax.set_title("Circularity-corrected attribution:\nclimate, not terrain, "
                 "organizes the continuum", fontsize=9)
    handles, labels_ = ax.get_legend_handles_labels()
    order = [1, 0, 3, 2, 5, 4]
    ax.legend([handles[k] for k in order], [labels_[k] for k in order],
              fontsize=6.8, frameon=False, loc="upper right", ncol=2)
    panel_label(ax, "a")

    # (b) 置换重要性
    ax = axes[0, 1]
    vars_ = list(VAR_LABELS)
    w = 0.26
    for j, (pc, c) in enumerate(zip(pcs, ["#1f77b4", "#ff7f0e", "#2ca02c"])):
        vals = [imp[(imp.target == pc) & (imp["var"] == v)].imp.iloc[0]
                for v in vars_]
        ax.bar(np.arange(len(vars_)) + (j - 1) * w, vals, w,
               color=c, label=pc)
    ax.axvspan(5.5, 6.5, color="#f5d5d0", zorder=0)
    ax.text(6, 1.30, "dam ≈ 0", ha="center", fontsize=8, color="#b03a2e")
    ax.set_xticks(range(len(vars_)))
    ax.set_xticklabels([VAR_LABELS[v] for v in vars_], rotation=18,
                       ha="right", fontsize=7.6)
    ax.set_ylabel("permutation importance (ΔR²)", fontsize=9)
    ax.set_title("Single-predictor permutation importance", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    panel_label(ax, "b")

    # (c) 气候空间中的 PC2
    ax = axes[1, 0]
    t = drv[["bio1_年均温", "bio15_降水季节性", "PC2"]].dropna()
    t = t[(t.bio1_年均温 > -25) & (t.bio1_年均温 < 32)]
    tb = t.copy()
    tb["tbin"] = pd.cut(tb.bio1_年均温, bins=np.arange(-25, 33, 4))
    tb["pbin"] = pd.cut(tb.bio15_降水季节性,
                        bins=np.arange(10, 185, 12.5))
    piv = tb.groupby(["pbin", "tbin"], observed=True).PC2.median().unstack()
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r",
                   vmin=-1.6, vmax=1.6, origin="lower", rasterized=True)
    ax.set_xticks(range(0, piv.shape[1], 3))
    ax.set_xticklabels([f"{int(iv.left)}" for iv in
                        piv.columns[::3]], fontsize=7.5)
    ax.set_yticks(range(0, piv.shape[0], 2))
    ax.set_yticklabels([f"{int(iv.left)}" for iv in
                        piv.index[::2]], fontsize=7.5)
    ax.set_xlabel("mean annual temperature bio1 (°C)", fontsize=9)
    ax.set_ylabel("P seasonality bio15 (CV)", fontsize=9)
    ax.set_title("PC2 (water-level variability axis) organised\n"
                 "by climate space", fontsize=9)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("median PC2", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    panel_label(ax, "c")

    # (d) 事件主导型预测因子
    ax = axes[1, 1]
    g = drv[["bio15_降水季节性", "bio1_年均温", "f5_gated"]].dropna()
    frac = []
    for col, color, lab, ls in [("bio15_降水季节性", "#c0392b",
                                 "P seasonality bio15 (decile)", "-"),
                                ("bio1_年均温", "#4c9ed9",
                                 "mean T bio1 (decile)", "--")]:
        q = pd.qcut(g[col], 10, duplicates="drop")
        f = g.groupby(q, observed=True).f5_gated.mean() * 100
        mid = [iv.mid for iv in f.index]
        frac.append((mid, f.values, color, lab, ls))
    ax2 = ax.twiny()
    for mid, fv, color, lab, ls in frac:
        target = ax if "bio15" in lab else ax2
        target.plot(mid, fv, "o" + ls, color=color, ms=4, lw=1.3,
                    label=lab)
    ax.set_xlabel("P seasonality bio15 (decile midpoint, CV)",
                  fontsize=9, color="#c0392b")
    ax2.set_xlabel("mean T bio1 (decile midpoint, °C)",
                   fontsize=9, color="#4c9ed9")
    ax.set_ylabel("weak-annual-cycle fraction (%)", fontsize=9)
    ax.tick_params(axis="x", colors="#c0392b")
    ax2.tick_params(axis="x", colors="#4c9ed9")
    ax.set_title("Weak-annual-cycle regimes (72%) tracked by\n"
                 "climate, not dams", fontsize=9)
    ax.text(0.97, 0.97,
            "AUC: climate 0.79 / terrain 0.65 / human 0.51",
            transform=ax.transAxes, fontsize=7.5, ha="right",
            va="top", color="#333333",
            bbox=dict(fc="white", ec="#bbbbbb", lw=0.6, alpha=0.95,
                      boxstyle="round,pad=0.3"))
    panel_label(ax, "d")

    for ext in ("png", "pdf"):
        p = OUT / f"Fig3_driver_attribution_出版级.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    facecolor="white")
        print("saved:", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
