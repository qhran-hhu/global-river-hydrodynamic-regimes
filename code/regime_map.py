# -*- coding: utf-8 -*-
"""全球水动力格局图：把 regime 连续谱映射回地理空间

每个 reach 的 PC1/PC2/PC3 秩次 → RGB 三色合成（相似 regime = 相似颜色）。
图版：
  (a) 全球 RGB regime 格局图（Robinson 投影，97,566 reach）
  (b) PC1-PC2 regime 空间（同配色，即图例）
  (c) "弱/无年循环"宏区（f5 门控）地理分布
  (d) PC1-PC3 载荷（颜色的物理含义）
输出: output/regime_space/全球水动力格局图.png, 格局图数据.parquet
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
from regime_space import build_block, OUT, RNG

try:
    from plotstyle import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

# Robinson 投影系数表（每 5° 纬度）
_LAT = np.arange(0, 95, 5.0)
_RX = np.array([1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600,
                0.9427, 0.9216, 0.8962, 0.8679, 0.8350, 0.7986, 0.7597,
                0.7186, 0.6732, 0.6213, 0.5722, 0.5322])
_RY = np.array([0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720,
                0.4340, 0.4958, 0.5571, 0.6176, 0.6769, 0.7346, 0.7903,
                0.8435, 0.8936, 0.9394, 0.9761, 1.0000])


def robinson(lon, lat):
    a = np.abs(lat)
    x = np.interp(a, _LAT, _RX)
    y = np.interp(a, _LAT, _RY)
    return (0.8487 * x * np.radians(lon),
            1.3523 * y * np.sign(lat))


def draw_graticule(ax):
    for lon in range(-180, 181, 30):
        la = np.linspace(-90, 90, 181)
        xs, ys = robinson(np.full_like(la, lon), la)
        ax.plot(xs, ys, color="#dddddd", lw=0.4, zorder=0)
    for lat in range(-60, 61, 30):
        lo = np.linspace(-180, 180, 361)
        xs, ys = robinson(lo, np.full_like(lo, lat))
        ax.plot(xs, ys, color="#dddddd", lw=0.4, zorder=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def rank01(v):
    return pd.Series(v).rank(pct=True).values


def main():
    from sklearn.decomposition import PCA

    Xn, meta = build_block()
    pca = PCA(n_components=7, random_state=RNG)
    Z = pca.fit_transform(Xn)
    pc = pd.DataFrame(Z[:, :3], index=meta.index, columns=["PC1", "PC2", "PC3"])
    pc[["continent", "x", "y"]] = meta[["continent", "x", "y"]]

    # f5 弱年循环（R²<0.3）与大坝标记
    full = pd.read_parquet(BASE / "output" / "特征矩阵_v1_qc.parquet",
                           columns=["f5_r2"])
    pc["f5_gated"] = (full.f5_r2 < 0.3) | full.f5_r2.isna()
    dam = pd.read_csv(BASE / "output" / "human_activity" / "大坝_reach匹配.csv",
                      encoding="utf-8-sig", low_memory=False)
    pc["dam"] = pc.index.isin(
        set(dam.loc[dam.qc_pass == True, "reach_id"]))

    rgb = np.column_stack([rank01(pc.PC1), rank01(pc.PC2), rank01(pc.PC3)])
    xs, ys = robinson(pc.x.values, pc.y.values)
    pc["rx"], pc["ry"] = xs, ys
    pc[["R", "G", "B"]] = rgb
    pc.to_parquet(OUT / "格局图数据.parquet")

    # 分洲概况
    print("=== 分洲：f5 门控占比 / PC 中位 ===")
    print(pc.groupby("continent").agg(
        n=("PC1", "size"), gated_pct=("f5_gated", lambda s: s.mean() * 100),
        dam_pct=("dam", lambda s: s.mean() * 100),
        PC1=("PC1", "median"), PC2=("PC2", "median"),
        PC3=("PC3", "median")).round(2).to_string())

    fig = plt.figure(figsize=(17, 11))
    order = np.random.default_rng(1).permutation(len(pc))  # 打乱防覆盖偏置

    ax = fig.add_subplot(2, 2, 1)
    draw_graticule(ax)
    ax.scatter(xs[order], ys[order], s=0.6, c=rgb[order], lw=0, rasterized=True)
    ax.set_title("(a) 全球水动力 regime 格局图（颜色=PC1/2/3 秩次 RGB 合成，"
                 "相似颜色=相似 regime）", fontsize=11)

    ax = fig.add_subplot(2, 2, 2)
    sub = np.random.default_rng(2).choice(len(pc), 30_000, replace=False)
    ax.scatter(pc.PC1.values[sub], pc.PC2.values[sub], s=1.2, c=rgb[sub],
               lw=0, rasterized=True)
    ax.set_xlabel("PC1 → 红"); ax.set_ylabel("PC2 → 绿")
    ax.set_title("(b) regime 空间（图 (a) 的配色钥匙；蓝=PC3 未在此平面）",
                 fontsize=11)

    ax = fig.add_subplot(2, 2, 3)
    draw_graticule(ax)
    g0 = ~pc.f5_gated.values
    ax.scatter(xs[g0], ys[g0], s=0.4, color="#cccccc", lw=0, rasterized=True)
    g1 = pc.f5_gated.values
    ax.scatter(xs[g1], ys[g1], s=0.8, color="#c00000", lw=0, rasterized=True)
    ax.set_title(f"(c) 弱年循环宏区地理分布（R²<0.3, 红, n={g1.sum():,}, "
                 f"{g1.mean()*100:.0f}%）", fontsize=11)

    ax = fig.add_subplot(2, 2, 4)
    feats = ["f2 水位变幅", "f3 事件响应", "f4 IQR(logH)", "f5 相位x",
             "f5 相位y", "f6 坡度", "f7 宽坡比"]
    w = 0.27
    xp = np.arange(7)
    for i, (c, lab) in enumerate([("#c0504d", "PC1 (34%)"),
                                  ("#9bbb59", "PC2 (20%)"),
                                  ("#4bacc6", "PC3 (17%)")]):
        ax.bar(xp + (i - 1) * w, pca.components_[i], w, color=c, label=lab)
    ax.set_xticks(xp); ax.set_xticklabels(feats, fontsize=8, rotation=20)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=9)
    ax.set_title("(d) PC 载荷：颜色的物理含义", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT / "全球水动力格局图.png", bbox_inches="tight", dpi=200)
    print(f"输出 -> {OUT / '全球水动力格局图.png'}")


if __name__ == "__main__":
    main()
