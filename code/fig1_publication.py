# -*- coding: utf-8 -*-
"""Figure 1 出版级：全球水动力 regime 格局图（Robinson 投影 + 海岸线 + 色钥匙）。

输出两版：
  A) 全分辨率散点版  Fig1_格局图_出版级_fullres.png/.pdf
  B) 0.5° 网格聚合版 Fig1_格局图_出版级_grid.png/.pdf（PC 空间中位聚合后重算秩次 RGB）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapefile  # pyshp

sys.path.insert(0, str(Path(__file__).parent))
from regime_map import robinson, draw_graticule, rank01

BASE = Path(__file__).parent
OUT = BASE / "output" / "regime_space"
COAST = OUT / "basemap" / "ne_110m_coastline.shp"

VAR_EXP = (34.7, 23.5, 16.4)  # PC1/2/3 解释方差 %（f2 深度相对修正后）


def load_data():
    df = pd.read_parquet(OUT / "regime_map_data.parquet")
    return df


def coast_lines():
    """读取海岸线并投影，返回线段列表（处理跨 180° 截断）。"""
    sf = shapefile.Reader(str(COAST))
    segs = []
    for shp in sf.shapes():
        pts = np.array(shp.points)
        # 同一 shape 可能有多个 part
        parts = list(shp.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            p = pts[parts[i]:parts[i + 1]]
            lon, lat = p[:, 0], p[:, 1]
            # 跨日界线处断开
            jump = np.where(np.abs(np.diff(lon)) > 180)[0]
            xs, ys = robinson(lon, lat)
            start = 0
            for j in jump:
                segs.append((xs[start:j + 1], ys[start:j + 1]))
                start = j + 1
            segs.append((xs[start:], ys[start:]))
    return segs


def draw_coast(ax, segs, lw=0.35, color="#9a9a9a"):
    for xs, ys in segs:
        if len(xs) > 1:
            ax.plot(xs, ys, color=color, lw=lw, zorder=1,
                    solid_capstyle="round")


def robinson_frame(ax):
    """Robinson 外框：±180° 子午线 + 两极线。"""
    la = np.linspace(-90, 90, 361)
    xl, yl = robinson(np.full_like(la, -180.0), la)
    xr, yr = robinson(np.full_like(la, 180.0), la)
    xt, yt = robinson(np.linspace(-180, 180, 361), np.full(361, 90.0))
    xb, yb = robinson(np.linspace(-180, 180, 361), np.full(361, -90.0))
    xs = np.concatenate([xl, xt, xr[::-1], xb[::-1]])
    ys = np.concatenate([yl, yt, yr[::-1], yb[::-1]])
    ax.plot(xs, ys, color="#666666", lw=0.8, zorder=2)
    ax.set_xlim(xs.min() - 0.05, xs.max() + 0.05)
    ax.set_ylim(ys.min() - 0.05, ys.max() + 0.05)


def grid_aggregate(df, res=0.5):
    """0.5° 格内 PC 取中位，再对格元重算秩次 RGB。"""
    gx = (df.x / res).round() * res
    gy = (df.y / res).round() * res
    g = df.groupby([gx, gy])[["PC1", "PC2", "PC3"]].median()
    rgb = np.column_stack([rank01(g.PC1), rank01(g.PC2), rank01(g.PC3)])
    xs, ys = robinson(g.index.get_level_values(0).values,
                      g.index.get_level_values(1).values)
    return xs, ys, rgb, len(g)


def add_color_key(fig, anchor, df, size=0.235):
    """PC1–PC2 空间 RGB 色钥匙（嵌在主图右下）。"""
    axk = fig.add_axes(anchor)
    sub = df.sample(min(30000, len(df)), random_state=42)
    axk.scatter(sub.PC1, sub.PC2, c=sub[["R", "G", "B"]].values,
                s=1.2, marker="s", lw=0, rasterized=True)
    axk.set_xlabel(f"PC1 — regime structure ({VAR_EXP[0]}%)",
                   fontsize=7.5)
    axk.set_ylabel(f"PC2 — water-level variability ({VAR_EXP[1]}%)",
                   fontsize=7.5)
    axk.set_title("Colour key (RGB = ranked PC1/PC2/PC3)",
                  fontsize=8, pad=3)
    # 物理含义角注（新 f2 口径：PC1=坡度/约束-宽度结构轴，PC2=水位变幅轴）
    kw = dict(fontsize=6.3, style="italic", color="#222222", zorder=5)
    xl = axk.get_xlim(); yl = axk.get_ylim()
    axk.text(xl[1] * 0.97, yl[0] + 0.06 * (yl[1] - yl[0]),
             "steep, confined,\nstable levels", ha="right",
             va="bottom", **kw)
    axk.text(xl[0] + 0.03 * (xl[1] - xl[0]), yl[1] * 0.97,
             "gentle lowland,\nhighly variable", ha="left", va="top", **kw)
    axk.text(xl[1] * 0.97, yl[1] * 0.97,
             "steep, confined,\nhighly variable", ha="right",
             va="top", **kw)
    axk.text(xl[0] + 0.03 * (xl[1] - xl[0]), yl[0] + 0.06 * (yl[1] - yl[0]),
             "gentle, stable\nlowland", ha="left", va="bottom", **kw)
    axk.tick_params(labelsize=6, length=2)
    for s in axk.spines.values():
        s.set_linewidth(0.6)
    return axk


def render(mode="fullres"):
    df = load_data()
    segs = coast_lines()

    fig = plt.figure(figsize=(13.6, 7.4))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.97])
    ax.set_facecolor("#f4f7fa")  # 极浅海洋底色
    draw_graticule(ax)
    draw_coast(ax, segs)

    if mode == "fullres":
        order = np.random.RandomState(42).permutation(len(df))
        ax.scatter(df.rx.values[order], df.ry.values[order],
                   c=df[["R", "G", "B"]].values[order],
                   s=0.55, lw=0, rasterized=True)
        n_txt = f"n = {len(df):,} river reaches"
    else:
        xs, ys, rgb, ncell = grid_aggregate(df, res=0.5)
        order = np.random.RandomState(42).permutation(len(xs))
        ax.scatter(xs[order], ys[order], c=rgb[order],
                   s=4.2, marker="s", lw=0, rasterized=True)
        n_txt = f"n = {len(df):,} reaches (0.5° median, {ncell:,} cells)"

    robinson_frame(ax)
    ax.set_aspect("equal")

    # 年循环主导度注记（修复后口径：R²≥0.3 为季节主导）
    seasonal_frac = (1 - df.f5_gated.mean()) * 100
    ax.text(0.005, 0.012,
            f"{n_txt}\nReaches with dominant annual harmonic "
            f"(R²≥0.3): {seasonal_frac:.0f}%",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            color="#333333",
            bbox=dict(fc="white", ec="#bbbbbb", lw=0.6, alpha=0.9,
                      boxstyle="round,pad=0.35"))

    add_color_key(fig, [0.715, 0.045, 0.275, 0.30], df)

    tag = "fullres" if mode == "fullres" else "grid"
    for ext in ("png", "pdf"):
        p = OUT / f"Fig1_格局图_出版级_{tag}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
        print("saved:", p)
    plt.close(fig)


if __name__ == "__main__":
    for m in ("fullres", "grid"):
        render(m)
