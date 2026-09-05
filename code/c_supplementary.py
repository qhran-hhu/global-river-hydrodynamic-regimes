# -*- coding: utf-8 -*-
"""C 档补充材料新算两项：QC 阈值敏感性 + 端元三亚组画像

C1 QC 阈值敏感性：在全量矩阵（138,856 reach，QC 前）上变化
   ice_clim_f / dark_frac / n_obs 阈值，重建 7 特征空间，
   偶极检验 + GMM k=2 改善——连续谱结论对 QC 口径不敏感。
C2 端元三亚组画像：弱年循环端（f5_r2<0.3）拆为冰冻影响 / 坝调控 /
   干旱事件型 / 其他过渡，与强季节端（R²>=0.6）对照画像。

输出：output/regime_space/C1_qc_threshold_sensitivity.csv
      output/regime_space/C2_endmember_subgroups.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "regime_space"
MAT_FULL = BASE / "output" / "feature_matrix_v1.parquet"
DRV = OUT / "driver_data.parquet"

FEATS = ["f2_rel_range", "f3_event_resp", "f4_iqr_logh",
         "f6_slope", "f7_width_slope"]


def build_space(df):
    """7 维特征空间（f5 振幅加权相位向量）→ 分位数正态 + PCA。"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer
    X = df.copy()
    X["f5x"] = np.log1p(df.f5_amp) * np.cos(df.f5_phase)
    X["f5y"] = np.log1p(df.f5_amp) * np.sin(df.f5_phase)
    X = X[["f2_rel_range", "f3_event_resp", "f4_iqr_logh",
           "f5x", "f5y", "f6_slope", "f7_width_slope"]].dropna()
    if len(X) < 5000:
        return None, None
    Z = PCA(n_components=7, random_state=0).fit_transform(
        QuantileTransformer(n_quantiles=1000,
                            output_distribution="normal",
                            random_state=0).fit_transform(X.values))
    return X, Z


def modality(df, seed=0):
    """dip PC1/PC2（5×5000 子样本中位 p）+ GMM k=2 改善（50k 子样本）。"""
    import diptest
    from sklearn.mixture import GaussianMixture
    X, Z = build_space(df)
    if X is None:
        return None
    rng = np.random.default_rng(seed)
    out = {}
    for j, nm in [(0, "pc1"), (1, "pc2")]:
        ps = []
        for _ in range(5):
            s = Z[rng.choice(len(Z), 5000, replace=False), j]
            _, p = diptest.diptest(s, boot_pval=True, n_boot=300,
                                   seed=int(rng.integers(1 << 31)))
            ps.append(float(p))
        out[f"dip_{nm}_p"] = float(np.median(ps))
    idx = rng.choice(len(Z), min(50000, len(Z)), replace=False)
    Zs = Z[idx]
    bics = []
    for k in (1, 2):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=0, max_iter=300).fit(Zs)
        bics.append(gm.bic(Zs))
    out["gmm_k2_impr_pct"] = float((bics[0] - bics[1]) / bics[0] * 100)
    out["n"] = len(X)
    return out


def c1():
    df = pd.read_parquet(MAT_FULL)
    rows = []
    # 基线（正式 QC 口径）
    base = df[(df.ice_clim_f <= 0.2) & (df.dark_frac <= 0.5)]
    for lab, sub in [
            ("baseline ice<=0.2, dark<=0.5", base),
            ("ice<=0.1", df[(df.ice_clim_f <= 0.1) & (df.dark_frac <= 0.5)]),
            ("ice<=0.3", df[(df.ice_clim_f <= 0.3) & (df.dark_frac <= 0.5)]),
            ("dark<=0.3", df[(df.ice_clim_f <= 0.2) & (df.dark_frac <= 0.3)]),
            ("dark<=0.7", df[(df.ice_clim_f <= 0.2) & (df.dark_frac <= 0.7)]),
            ("n_obs>=50", base[base.n_obs >= 50]),
            ("n_obs>=100", base[base.n_obs >= 100]),
            ("n_obs>=150", base[base.n_obs >= 150]),
            ("no QC at all", df)]:
        r = modality(sub)
        if r:
            r["variant"] = lab
            rows.append(r)
        print(f"  {lab}: n={r['n'] if r else 'NA'}", flush=True)
    R = pd.DataFrame(rows)[["variant", "n", "dip_pc1_p", "dip_pc2_p",
                            "gmm_k2_impr_pct"]]
    R.to_csv(OUT / "C1_qc_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(R.to_string(index=False))


def c2():
    qc = pd.read_parquet(BASE / "output" / "feature_matrix_v1_qc.parquet")
    drv = pd.read_parquet(DRV).reset_index()
    if "reach_id" not in drv.columns:
        drv = drv.rename(columns={drv.columns[0]: "reach_id"})
    d = qc.merge(drv[["reach_id", "dam", "abs_lat", "bio1_年均温",
                      "bio12_年降水", "bio15_降水季节性"]],
                 on="reach_id", how="left")
    weak = d[d.f5_r2 < 0.3].copy()
    # 三亚组（互斥，优先级：寒冷 > 坝 > 干旱）
    # 注：矩阵 ice_clim_f 全为 0/-999（源缺失），冰冻以气候代理 bio1<0°C 定义
    weak["subgroup"] = "其他过渡"
    weak.loc[weak.bio12_年降水 < 500, "subgroup"] = "干旱事件型"
    weak.loc[weak.dam == 1, "subgroup"] = "坝调控"
    weak.loc[weak.bio1_年均温 < 0, "subgroup"] = "寒冷/冰冻影响"
    strong = d[d.f5_r2 >= 0.6].copy()
    strong["subgroup"] = "强季节端（对照）"
    allx = pd.concat([weak, strong])

    rows = []
    for sg, x in allx.groupby("subgroup"):
        cont = x.continent.value_counts(normalize=True)
        rows.append(dict(
            亚组=sg, n=len(x), 占比_pct=round(100 * len(x) / len(d), 1),
            f2_中位=round(x.f2_rel_range.median(), 3),
            f3_中位=round(x.f3_event_resp.median(), 3),
            f4_中位=round(x.f4_iqr_logh.median(), 3),
            f5_r2_中位=round(x.f5_r2.median(), 3),
            纬度中位=round(x.abs_lat.median(), 1),
            bio1_年均温=round(x.bio1_年均温.median(), 1),
            bio12_年降水=round(x.bio12_年降水.median(), 0),
            bio15_降水季节性=round(x.bio15_降水季节性.median(), 0),
            冰冻比_中位=round(x.ice_clim_f.median(), 3),
            坝邻近_pct=round(100 * (x.dam == 1).mean(), 1),
            前两大洲=" ".join(f"{k} {v * 100:.0f}%" for k, v in
                             cont.head(2).items())))
    R = pd.DataFrame(rows).sort_values("n", ascending=False)
    R.to_csv(OUT / "C2_endmember_subgroups.csv", index=False, encoding="utf-8-sig")
    print(R.to_string(index=False))


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("c1", "all"):
        print("=== C1 QC 阈值敏感性 ===", flush=True)
        c1()
    if which in ("c2", "all"):
        print("\n=== C2 端元三亚组画像 ===", flush=True)
        c2()
