# -*- coding: utf-8 -*-
"""驱动因子归因：气候 / 地形 / 人类活动各自解释 regime 空间位置多少方差

预测因子（7 个，全部外生于动力学特征，坡度除外——f6 参与 PC 构造，
其 R² 存在轻度循环性，解读时已标注）：
  气候: bio1 年均温, bio4 温度季节性, bio12 年降水, bio15 降水季节性, |lat|
  地形: slope (f6, SWORD 先验)
  人类: dam_adjacent (GDAT 0.2° 匹配)
目标: PC1, PC2, PC3 (回归, R²) + f5_gated 事件主导型 (分类, AUC)
方法: HistGradientBoosting + 全子集方差分解（共同性分析）
  python driver_attribution.py climate  # 栅格采样（~1 min）
  python driver_attribution.py model    # 拟合 + 出图出表（~3 min）
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
CLIM = OUT / "climate"

try:
    from plotstyle import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

BIOS = {1: "bio1_年均温", 4: "bio4_温度季节性", 12: "bio12_年降水",
        15: "bio15_降水季节性"}
GROUPS = {"气候": ["bio1_年均温", "bio4_温度季节性", "bio12_年降水",
                   "bio15_降水季节性", "abs_lat"],
          "地形": ["slope"],
          "人类": ["dam"]}


def step_climate():
    import rasterio

    pc = pd.read_parquet(OUT / "regime_map_data.parquet")
    coords = list(zip(pc.x.values, pc.y.values))
    out = pc[["PC1", "PC2", "PC3", "f5_gated", "dam"]].copy()
    out["abs_lat"] = pc.y.abs()
    for num, name in BIOS.items():
        with rasterio.open(CLIM / f"wc2.1_10m_bio_{num}.tif") as ds:
            nodata = ds.nodata
            v = np.array([r[0] for r in ds.sample(coords)], dtype=float)
        if nodata is not None:
            v[v == nodata] = np.nan
        v[v < -9999] = np.nan
        out[name] = v
        print(name, "NaN%", round(out[name].isna().mean() * 100, 2))
    full = pd.read_parquet(BASE / "output" / "feature_matrix_v1_qc.parquet",
                           columns=["f6_slope"])
    out["slope"] = full.f6_slope
    out.to_parquet(OUT / "driver_data.parquet")
    print("saved", OUT / "driver_data.parquet", out.shape)


def step_model():
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    d = pd.read_parquet(OUT / "driver_data.parquet")
    feats = sum(GROUPS.values(), [])
    rng = np.random.default_rng(0)
    sub = rng.choice(len(d), 50_000, replace=False)
    d = d.iloc[sub]
    tr, te = train_test_split(np.arange(len(d)), test_size=0.2,
                              random_state=0)

    res, imp_rows = {}, []
    for target in ["PC1", "PC2", "PC3", "f5_gated"]:
        y = d[target].values.astype(float)
        is_clf = target == "f5_gated"
        r2 = {}
        subsets = {"气候": GROUPS["气候"], "地形": GROUPS["地形"],
                   "人类": GROUPS["人类"],
                   "气候+地形": GROUPS["气候"] + GROUPS["地形"],
                   "气候+人类": GROUPS["气候"] + GROUPS["人类"],
                   "地形+人类": GROUPS["地形"] + GROUPS["人类"],
                   "全部": feats}
        for tag, cols in subsets.items():
            Xt, Xv = d[cols].values[tr], d[cols].values[te]
            if is_clf:
                m = HistGradientBoostingClassifier(random_state=0)
                m.fit(Xt, y[tr])
                r2[tag] = float(roc_auc_score(y[te],
                                              m.predict_proba(Xv)[:, 1]))
            else:
                m = HistGradientBoostingRegressor(random_state=0)
                m.fit(Xt, y[tr])
                r2[tag] = float(r2_score(y[te], m.predict(Xv)))
            print(target, tag, round(r2[tag], 3), flush=True)
        # 共同性分析（对分类目标用 AUC 的相对提升近似，仅作参考）
        allv = r2["全部"]
        uniq = {g: allv - r2["".join(x) if False else
                             {"气候": "地形+人类", "地形": "气候+人类",
                              "人类": "气候+地形"}[g]]
                for g in GROUPS}
        shared = allv - sum(max(u, 0) for u in uniq.values())
        res[target] = dict(r2=r2,
                           unique={g: round(max(u, 0), 4)
                                   for g, u in uniq.items()},
                           shared=round(max(shared, 0), 4))
        # 单变量置换重要性（全部因子模型）
        m_full = HistGradientBoostingRegressor(random_state=0)
        if is_clf:
            m_full = HistGradientBoostingClassifier(random_state=0)
        m_full.fit(d[feats].values[tr], y[tr])
        pi = permutation_importance(m_full, d[feats].values[te], y[te],
                                    n_repeats=3, random_state=0,
                                    scoring="roc_auc" if is_clf else "r2")
        for f_, v_ in zip(feats, pi.importances_mean):
            imp_rows.append(dict(target=target, var=f_, imp=float(v_)))

    imp = pd.DataFrame(imp_rows)
    json.dump(res, open(OUT / "driver_attribution_r2.json", "w"),
              ensure_ascii=False, indent=1)
    imp.to_csv(OUT / "driver_attribution_permutation_importance.csv", index=False,
               encoding="utf-8-sig")

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    targets = ["PC1", "PC2", "PC3"]
    w = 0.25
    xp = np.arange(3)
    bottoms = np.zeros(3)
    for g, col in [("气候", "#4bacc6"), ("地形", "#9bbb59"),
                   ("人类", "#c0504d"), ("shared", "#bbbbbb")]:
        vals = np.array([res[t]["shared" if g == "shared" else "unique"]
                         [g] if g != "shared" else res[t]["shared"]
                         for t in targets])
        lab = {"气候": "气候 独特", "地形": "地形 独特", "人类": "人类 独特",
               "shared": "共同/交互"}[g]
        ax.bar(xp, vals, 0.55, bottom=bottoms, color=col, label=lab)
        bottoms += vals
    for i, t in enumerate(targets):
        ax.text(i, bottoms[i] + 0.01, f"总R²={res[t]['r2']['全部']:.2f}",
                ha="center", fontsize=9)
    ax.set_xticks(xp); ax.set_xticklabels(targets)
    ax.set_ylabel("R² 贡献")
    ax.set_title("(a) 方差分解：谁解释 regime 空间位置？")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    vars_order = ["bio1_年均温", "bio4_温度季节性", "bio12_年降水",
                  "bio15_降水季节性", "abs_lat", "slope", "dam"]
    w = 0.27
    xp = np.arange(len(vars_order))
    for i, t in enumerate(targets):
        v = imp[imp.target == t].set_index("var").imp.reindex(vars_order)
        ax.bar(xp + (i - 1) * w, v.values, w, label=t)
    ax.set_xticks(xp)
    ax.set_xticklabels(["年均温", "温度季节性", "年降水", "降水季节性",
                        "|纬度|", "坡度", "大坝"], fontsize=8, rotation=15)
    ax.set_ylabel("置换重要性（ΔR²）")
    ax.set_title("(b) 单变量置换重要性")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    dd = d.dropna(subset=["bio1_年均温", "bio15_降水季节性"])
    b1 = pd.cut(dd.bio1_年均温, 12)
    b15 = pd.cut(dd.bio15_降水季节性, 12)
    piv = dd.pivot_table(values="PC2", index=b15, columns=b1,
                         observed=True, aggfunc="median")
    im = ax.imshow(piv.values, aspect="auto", origin="lower",
                   cmap="RdBu_r")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c.mid:.0f}" for c in piv.columns], fontsize=7,
                       rotation=45)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{c.mid:.0f}" for c in piv.index], fontsize=7)
    ax.set_xlabel("年均温 bio1（°C）"); ax.set_ylabel("降水季节性 bio15（CV）")
    fig.colorbar(im, ax=ax, label="PC2 中位")
    ax.set_title("(c) 气候空间中的 regime（PC2 季节-变幅轴）")

    ax = axes[1, 1]
    q = pd.qcut(dd.bio15_降水季节性, 10, duplicates="drop")
    g = dd.groupby(q, observed=True).agg(
        x=("bio15_降水季节性", "median"), rate=("f5_gated", "mean"))
    ax.plot(g.x, g.rate * 100, "o-", color="#c0504d", lw=1.5,
            label="事件主导型比例")
    q2 = pd.qcut(dd.bio1_年均温, 10, duplicates="drop")
    g2 = dd.groupby(q2, observed=True).agg(
        x=("bio1_年均温", "median"), rate=("f5_gated", "mean"))
    ax2 = ax.twiny()
    ax2.plot(g2.x, g2.rate * 100, "s--", color="#4bacc6", lw=1.2,
             label="对年均温")
    ax.set_xlabel("降水季节性 bio15（十分位中值，红）")
    ax2.set_xlabel("年均温 bio1（十分位中值，蓝）", color="#4bacc6")
    ax.set_ylabel("事件主导型 regime 占比（%）")
    auc = res["f5_gated"]["r2"]
    ax.text(0.03, 0.97,
            f"AUC: 气候 {auc['气候']:.2f} / 地形 {auc['地形']:.2f} / "
            f"人类 {auc['人类']:.2f} / 全部 {auc['全部']:.2f}",
            transform=ax.transAxes, fontsize=9, va="top")
    ax.set_title("(d) 事件主导型 regime 由什么预测？")

    fig.suptitle("驱动因子归因：气候 / 地形 / 人类活动对全球水动力 "
                 "regime 的解释力（n=50,000 子样本）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "driver_attribution.png", bbox_inches="tight", dpi=150)
    print(f"输出 -> {OUT / 'driver_attribution.png'}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "climate"
    {"climate": step_climate, "model": step_model}[step]()
