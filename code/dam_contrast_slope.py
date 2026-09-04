# -*- coding: utf-8 -*-
"""A4 大坝对照加坡度（意见书问题 4）：选址效应拆解

坝址河流天然更陡、更靠上游，而坡度直接进入 f2/f4/f7 的自然基线——
−52.7% 抑制里可能混入选址效应。三臂对比：
  arm1 原匹配：同洲 + 纬度带 ±5°（复现 −52.7%）
  arm2 坡度匹配：arm1 候选内再按 log10(slope) 最近邻（卡尺 0.5 dex，不足放宽）
  arm3 回归标准化：对照池拟合 log10 f2 ~ log10 slope + |lat|（分洲），
       预测坝邻近 reach 的"自然期望值"，比较观测 vs 期望
另输出选址诊断：坝邻近 vs 对照池坡度分布对比。
输出：output/human_activity/大坝对照_坡度匹配.csv
  python dam_contrast_slope.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "human_activity"

MATCH_DEG = 0.2
CONTROL_DEG = 0.5
FEATS = [("f2_rel_range", "f2 水位相对变幅"),
         ("f3_event_resp", "f3 宽度事件响应"),
         ("f4_iqr_logh", "f4 IQR(logH)")]


def load_sets():
    import shapefile
    from scipy.spatial import cKDTree

    fm_full = pd.read_parquet(BASE / "output" / "特征矩阵_v1.parquet")
    fm_qc = pd.read_parquet(BASE / "output" / "特征矩阵_v1_qc.parquet")
    qc_ids = set(fm_qc.index)
    sf = shapefile.Reader(str(OUT / "GDAT" / "GDAT_data_v1" / "data"
                              / "GDAT_v1_dams.shp"))
    fields = [f[0] for f in sf.fields[1:]]
    dams = pd.DataFrame([dict(zip(fields, r)) for r in sf.iterRecords()])
    dams["lon"] = [s.points[0][0] if len(s.points) else np.nan
                   for s in sf.shapes()]
    dams["lat"] = [s.points[0][1] if len(s.points) else np.nan
                   for s in sf.shapes()]
    dams = dams.dropna(subset=["lon", "lat"])

    pts = fm_full[["x", "y"]].values.copy()
    pts[:, 0] *= np.cos(np.deg2rad(pts[:, 1]))
    dpts = dams[["lon", "lat"]].values.copy()
    dpts[:, 0] *= np.cos(np.deg2rad(dpts[:, 1]))
    dist, idx = cKDTree(pts).query(dpts)
    dams["reach_id"] = fm_full.index[idx]
    matched = dams[dist <= MATCH_DEG]
    dam_reaches = fm_full.loc[matched.reach_id.unique()]
    dam_reaches = dam_reaches[dam_reaches.index.isin(qc_ids)]

    qpts = fm_full[["x", "y"]].values.copy()
    qpts[:, 0] *= np.cos(np.deg2rad(qpts[:, 1]))
    dd = cKDTree(dpts).query(qpts)[0]
    pool = fm_full[dd > CONTROL_DEG]
    pool = pool[pool.index.isin(qc_ids)]
    # 特征值统一取 QC 矩阵（含 kNN 坡度填补，无 NaN）
    dam_reaches = fm_qc.loc[dam_reaches.index]
    pool = fm_qc.loc[pool.index]
    return dam_reaches, pool


def contrast(dam, ctrl, tag, rows):
    for c, lab in FEATS:
        a = dam[c].dropna()
        b = ctrl[c].dropna()
        ks = ks_2samp(a, b)
        rel = 100 * (a.median() - b.median()) / abs(b.median())
        rows.append(dict(匹配臂=tag, 特征=lab, n_坝=len(a), n_对照=len(b),
                         坝邻近中位=round(a.median(), 4),
                         对照中位=round(b.median(), 4),
                         相对差pct=round(rel, 1),
                         KS_p=f"{ks.pvalue:.1e}"))
        print(f"  [{tag}] {lab}: {a.median():.4f} vs {b.median():.4f} "
              f"({rel:+.1f}%, p={ks.pvalue:.1e})", flush=True)


def main():
    dam, pool = load_sets()
    # 残余 63 个 NaN 坡度：以对照池中位数填补
    s_med = pool.f6_slope.median()
    dam = dam.copy(); pool = pool.copy()
    dam["f6_slope"] = dam.f6_slope.fillna(s_med)
    pool["f6_slope"] = pool.f6_slope.fillna(s_med)
    print(f"坝邻近 n={len(dam)}, 对照池 n={len(pool)}")
    rows = []
    rng = np.random.default_rng(0)

    # ---- 选址诊断：坡度分布 ----
    sd = np.log10(dam.f6_slope.clip(lower=1e-8))
    sp = np.log10(pool.f6_slope.clip(lower=1e-8))
    print(f"坡度诊断: 坝邻近 log10S 中位 {sd.median():.3f} vs 对照池 "
          f"{sp.median():.3f} (差 {sd.median()-sp.median():+.3f} dex, "
          f"KS p={ks_2samp(sd, sp).pvalue:.1e})")

    # ---- arm1 原匹配（同洲 + 纬度带 ±5°）----
    idx1 = []
    for _, row in dam.iterrows():
        cand = pool[(pool.continent == row.continent)
                    & (abs(pool.y - row.y) <= 5)]
        if len(cand) >= 5:
            idx1.append(rng.choice(cand.index.values))
    ctrl1 = pool.loc[pd.unique(np.asarray(idx1))]
    contrast(dam, ctrl1, "arm1 原匹配(洲+纬度)", rows)

    # ---- arm2 加坡度匹配（候选内 log10 slope 最近邻，卡尺 0.5 dex）----
    idx2 = []
    ls_pool = np.log10(pool.f6_slope.clip(lower=1e-8))
    for _, row in dam.iterrows():
        cand = pool[(pool.continent == row.continent)
                    & (abs(pool.y - row.y) <= 5)]
        if len(cand) < 5:
            continue
        ls_d = np.log10(max(row.f6_slope, 1e-8))
        d_ls = (ls_pool.loc[cand.index] - ls_d).abs()
        for cal in (0.5, 0.75, 1.0, 99):
            ok = d_ls[d_ls <= cal]
            if len(ok) >= 5:
                idx2.append(ok.sample(1, random_state=rng).index[0])
                break
    ctrl2 = pool.loc[pd.unique(np.asarray(idx2))]
    print(f"arm2 匹配成功率 {len(idx2)}/{len(dam)}")
    contrast(dam, ctrl2, "arm2 坡度匹配(洲+纬度+坡度)", rows)

    # ---- arm3 回归标准化（对照池分洲拟合 log10 f ~ log10 S + |lat|，
    #      非线性 GBM + 对照 5 折交叉拟合残差作参照）----
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict
    print("arm3 回归标准化(GBM, 对照交叉拟合残差参照):")
    for c, lab in FEATS:
        ycol = np.log10(pool[c].clip(lower=1e-6))
        okp = ycol.notna()
        X = pd.DataFrame({"ls": np.log10(pool.f6_slope.clip(lower=1e-8)),
                          "alat": pool.y.abs(), "cont": pool.continent})
        gbm = lambda: HistGradientBoostingRegressor(random_state=0)
        yhat_d = np.full(len(dam), np.nan)
        for cont in dam.continent.unique():
            mtr = (X.cont == cont) & okp
            if mtr.sum() < 100:
                continue
            reg = gbm().fit(X.loc[mtr, ["ls", "alat"]], ycol[mtr])
            md = dam.continent == cont
            Xd = pd.DataFrame({
                "ls": np.log10(dam.loc[md, "f6_slope"].clip(lower=1e-8)),
                "alat": dam.loc[md].y.abs()})
            yhat_d[md.values] = reg.predict(Xd)
        obs = np.log10(dam[c].clip(lower=1e-6))
        ok = (~np.isnan(yhat_d)) & obs.notna().values
        res_d = (obs[ok] - yhat_d[ok]).values
        # 对照参照：交叉拟合残差（防止在样本内过拟合偏向 0）
        res_p_all = np.full(len(pool), np.nan)
        for cont in pool.continent.unique():
            mtr = (X.cont == cont) & okp
            if mtr.sum() < 100:
                continue
            yhat_cv = cross_val_predict(
                gbm(), X.loc[mtr, ["ls", "alat"]], ycol[mtr], cv=5)
            res_p_all[np.where(mtr.values)[0]] = ycol[mtr].values - yhat_cv
        res_p = res_p_all[~np.isnan(res_p_all)]
        med_d = np.median(res_d)
        med_p = np.median(res_p)
        rel = 100 * (10 ** (med_d - med_p) - 1)
        ks = ks_2samp(res_d, res_p)
        rows.append(dict(匹配臂="arm3 回归标准化", 特征=lab, n_坝=int(ok.sum()),
                         n_对照=len(res_p),
                         坝邻近中位=round(10 ** med_d, 4),
                         对照中位=round(10 ** med_p, 4),
                         相对差pct=round(rel, 1), KS_p=f"{ks.pvalue:.1e}"))
        print(f"  [arm3] {lab}: 调整后 {rel:+.1f}% "
              f"(坝残差中位 {med_d:+.3f} vs 对照 {med_p:+.3f} dex, "
              f"p={ks.pvalue:.1e})", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "大坝对照_坡度匹配.csv", index=False, encoding="utf-8-sig")
    print(f"输出 -> {OUT / '大坝对照_坡度匹配.csv'}")


if __name__ == "__main__":
    main()
