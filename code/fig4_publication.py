# -*- coding: utf-8 -*-
"""Figure 4 出版级：人类指纹——大坝改变水动力，但沿谱位移而非出谱。

(a) tgd_fingerprint：四特征建坝后变化随距坝里程（长江六站实测）
(b) 三峡季节相位移动（建坝后峰值月 − 建坝前，负=提前）
(c) GDAT 全球坝邻近 reach 特征对比（坝邻近 vs 对照中位相对差）
(d) 坝邻近 reach 在 HDBSCAN 各簇的占比 vs 基线 5.4%（无大坝 regime 岛）
"""
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
DAM = BASE / "output" / "human_activity"

plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "font.size": 9})

STATION_EN = {"朱沱站": "Zhutuo", "宜昌站": "Yichang", "监利站": "Jianli",
              "汉口站": "Hankou", "九江站": "Jiujiang", "大通站": "Datong"}


def panel_label(ax, s):
    ax.text(-0.15, 1.04, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")


def main():
    tgd = pd.read_csv(DAM / "tgd_fingerprint_displacement.csv", encoding="utf-8-sig")
    tgd["station_en"] = tgd.station.map(STATION_EN)
    gdat = pd.read_csv(DAM / "dam_contrast_feature_diff.csv", encoding="utf-8-sig")
    blob = pd.read_csv(OUT / "cluster_composition.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    fig.subplots_adjust(hspace=0.46, wspace=0.30,
                        left=0.10, right=0.97, top=0.93, bottom=0.10)

    # (a) 三峡特征位移
    ax = axes[0, 0]
    series = [("var_Z", "WSE range (P90−P10)/P50", "#c0392b"),
              ("var_Q", "discharge range (P90−P10)/P50", "#4c9ed9"),
              ("ev_Q", "event response (P95−P50)/P50", "#8fbf60"),
              ("iqr_H", "IQR(log$_{10}$H)", "#6a51a3")]
    for col, lab, c in series:
        ax.plot(tgd.dist_km, tgd[col], "o-", color=c, ms=4.5, lw=1.3,
                label=lab)
    ax.axvline(0, color="#888888", lw=0.9, ls=":")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.text(8, 47, "Three Gorges Dam", fontsize=7.5, rotation=90,
            color="#666666", va="top")
    for _, r in tgd.iterrows():
        if not np.isnan(r.iqr_H):
            ax.annotate(r.station_en, (r.dist_km, r.iqr_H),
                        textcoords="offset points", xytext=(4, -9),
                        fontsize=6.8, color="#555555")
    ax.set_xlabel("distance from dam (km; negative = upstream control)",
                  fontsize=9)
    ax.set_ylabel("post-dam change (%)", fontsize=9)
    ax.set_title("TGD fingerprint: suppression of hydrodynamic\n"
                 "variability peaks 330–500 km downstream,\n"
                 "buffered near the dam, dissolving at the tidal limit",
                 fontsize=8.6)
    ax.legend(fontsize=6.8, frameon=False, loc="lower left", ncols=2)
    panel_label(ax, "a")

    # (b) 相位移动
    ax = axes[0, 1]
    d = tgd.dropna(subset=["phase_shift_months"])
    colors = ["#c0392b" if v < 0 else "#4c9ed9"
              for v in d.phase_shift_months]
    ax.bar(d.dist_km.astype(str), d.phase_shift_months, 0.55,
           color=colors)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i, r.phase_shift_months - 0.02 if r.phase_shift_months < 0
                else r.phase_shift_months + 0.01,
                f"{r.phase_shift_months:+.2f}", ha="center",
                va="top" if r.phase_shift_months < 0 else "bottom",
                fontsize=7.6)
        ax.text(i, 0.05, r.station_en, ha="center", fontsize=7,
                color="#555555")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("distance from dam (km)", fontsize=9)
    ax.set_ylabel("peak-month shift after dam (months)", fontsize=9)
    ax.set_title("Seasonal phase advances 0.3–0.4 months at\n"
                 "40–500 km downstream", fontsize=9)
    panel_label(ax, "b")

    # (c) GDAT 全球对比：原匹配 vs 坡度校正（两臂）
    ax = axes[1, 0]
    labels = ["WSE relative\nrange (f2)", "width event\nresponse (f3)",
              "IQR(log$_{10}$H)\n(f4)"]
    g3 = pd.read_csv(DAM / "dam_contrast_slope_matched.csv", encoding="utf-8-sig")
    arms = [("arm1 原匹配(洲+纬度)", "climate-latitude match\n(naive)",
             "#e8a33d", "//"),
            ("arm2 坡度匹配(洲+纬度+坡度)", "+ slope matched",
             "#c0392b", ""),
            ("arm3 回归标准化", "+ regression\nstandardized",
             "#7b2d26", "..")]
    w = 0.26
    for j, (arm, lab, col, hatch) in enumerate(arms):
        sub = g3[g3.匹配臂 == arm]
        vals = sub["相对差pct"].values
        xs = np.arange(3) + (j - 1) * (w + 0.03)
        ax.bar(xs, vals, w, color=col, hatch=hatch, alpha=0.9, label=lab,
               edgecolor="white", lw=0.4)
        for x, v, p in zip(xs, vals, sub["KS_p"].values):
            pv = float(p)
            star = "n.s." if pv >= 0.05 else (
                "*" if pv >= 0.01 else ("**" if pv >= 1e-3 else "***"))
            ax.text(x, v + 2.5 if v > 0 else v - 2.5, f"{v:+.0f}% {star}",
                    ha="center", va="bottom" if v > 0 else "top",
                    fontsize=6.6)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("dam-adjacent vs control, median diff. (%)", fontsize=9)
    ax.set_title("Global GDAT contrast, siting-corrected: dam reaches are\n"
                 "2.4× steeper (selection); with depth-relative f2 the\n"
                 "WSE-range contrast is only −0.3 to −1.2% — no global\n"
                 "suppression of relative variability", fontsize=8.4)
    ax.legend(fontsize=6.8, frameon=False, loc="upper right")
    ax.set_ylim(-22, 25)
    panel_label(ax, "c")

    # (d) 簇内坝富集
    ax = axes[1, 1]
    base = 5.4
    bd = blob[blob.cluster != -1]  # 噪声点不是簇
    ax.bar(range(len(bd)), bd.dam_pct, 0.6,
           color="#7fb3d5", edgecolor="#2e6b99", lw=0.8)
    ax.axhline(base, color="#c0392b", lw=1.2, ls="--")
    ax.text(len(bd) - 0.4, base + 0.3, f"baseline {base}%",
            color="#c0392b", fontsize=8, ha="right")
    for i, (_, r) in enumerate(bd.iterrows()):
        ax.text(i, r.dam_pct + 0.25, f"×{r.dam_pct/base:.1f}",
                ha="center", fontsize=6.2, rotation=90)
    ax.set_xticks(range(len(bd)))
    ax.set_xticklabels(bd.cluster.astype(str), fontsize=7)
    ax.set_xlabel("HDBSCAN cluster", fontsize=9)
    ax.set_ylabel("dam-adjacent reach share (%)", fontsize=9)
    ax.set_title("No dam-regime island: the regime space is one dominant\n"
                 "mass; its only satellite cluster (n = 322) is 2.7×\n"
                 "dam-enriched, not dam-created", fontsize=8.6)
    ax.set_ylim(0, 16.5)
    panel_label(ax, "d")

    for ext in ("png", "pdf"):
        p = OUT / f"Fig4_人类指纹_出版级.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    facecolor="white")
        print("saved:", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
