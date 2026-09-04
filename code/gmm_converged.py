# -*- coding: utf-8 -*-
"""GMM BIC 收敛修复重跑（一致性通读发现主流程 n_init=1 严重欠收敛）。

用法: python gmm_converged.py full | ablate | c1
- full   : 主空间 PC1-6, k=1..8, 真实 + 高斯参照 + 峰度匹配 t 参照
- ablate : nof5/nof4/noslope/noslope_nogate 四变体, k=1..6, 真实+高斯参照
- c1     : C1 QC 阈值九变体, k=1..2 收敛版
统一 n_init=10, max_iter=500, tol=1e-4。
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer

BASE = "output"
OUT = f"{BASE}/regime_space"
RNG = 0
KW = dict(covariance_type="full", max_iter=500, n_init=10, tol=1e-4,
          random_state=RNG)


def bic_curve(X, ks):
    bics = {}
    for k in ks:
        g = GaussianMixture(k, **KW).fit(X)
        bics[k] = g.bic(X)
        print(f"    k={k}: BIC={bics[k]:.0f}", flush=True)
    return bics


def raw_block(d):
    f5x = np.log1p(d.f5_amp.clip(lower=0)) * np.cos(d.f5_phase)
    f5y = np.log1p(d.f5_amp.clip(lower=0)) * np.sin(d.f5_phase)
    return pd.DataFrame({
        "f2": d.f2_rel_range.fillna(d.f2_rel_range.median()),
        "f3": d.f3_event_resp.fillna(d.f3_event_resp.median()),
        "f4": d.f4_iqr_logh.fillna(d.f4_iqr_logh.median()),
        "f5x": f5x.fillna(0.0), "f5y": f5y.fillna(0.0),
        "f6": np.log10(d.f6_slope.fillna(d.f6_slope.median()).clip(lower=1e-8)),
        "f7": np.log10(d.f7_width_slope.fillna(
            d.f7_width_slope.median()).clip(lower=1e-12)),
    }, index=d.index)


def refs(X, rng):
    mu, cov = X.mean(0), np.cov(X.T)
    Xg = rng.multivariate_normal(mu, cov, size=len(X))
    kurt = max(stats.kurtosis(X[:, 0]), 0.05)
    df_t = 4 + 6 / kurt
    g = rng.standard_gamma(df_t / 2, size=len(X)) / (df_t / 2)
    Z = rng.multivariate_normal(np.zeros(X.shape[1]), cov, size=len(X))
    Xt = mu + Z / np.sqrt(g)[:, None]
    return Xg, Xt, df_t


def run_full():
    pc = pd.read_parquet(f"{OUT}/pc_scores.parquet")
    cols = [f"PC{i+1}" for i in range(6)]
    rng = np.random.default_rng(RNG)
    X = pc[cols].values[rng.choice(len(pc), 50_000, replace=False)]
    Xg, Xt, df_t = refs(X, rng)
    rows = []
    for tag, Xd in [("real", X), ("unimodal_ref", Xg), ("t_ref", Xt)]:
        print(f"== {tag} ==", flush=True)
        bics = bic_curve(Xd, range(1, 9))
        for k, b in bics.items():
            rows.append(dict(data=tag, k=k, bic=b,
                             impr=(bics[1] - b) / bics[1] * 100))
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/gmm_bic_converged.csv", index=False,
             encoding="utf-8-sig")
    print("t_ref df:", round(df_t, 1))
    print(t.pivot(index="k", columns="data", values="impr").round(2))


def run_ablate():
    VARIANTS = {
        "nof5":    ["f2", "f3", "f4", "f6", "f7"],
        "nof4":    ["f2", "f3", "f5x", "f5y", "f6", "f7"],
        "noslope": ["f2", "f3", "f5x", "f5y"],
    }
    d = pd.read_parquet(f"{BASE}/特征矩阵_v1_qc.parquet")
    raw = raw_block(d)
    rows = []
    for tag, cols in {**VARIANTS}.items():
        rng = np.random.default_rng(RNG)
        Xn = QuantileTransformer(n_quantiles=1000,
                                 output_distribution="normal",
                                 random_state=RNG).fit_transform(
                                     raw[cols].values)
        Z = PCA(n_components=len(cols), random_state=RNG).fit_transform(Xn)
        Xs = Z[rng.choice(len(Z), 50_000, replace=False)][:, :min(4, Z.shape[1])]
        mu, cov = Xs.mean(0), np.cov(Xs.T)
        Xref = rng.multivariate_normal(mu, cov, size=len(Xs))
        for dtag, Xd in [("real", Xs), ("unimodal_ref", Xref)]:
            print(f"== {tag}/{dtag} ==", flush=True)
            bics = bic_curve(Xd, range(1, 7))
            rows.append(dict(variant=tag, test=f"gmm_{dtag}",
                             stat=round((bics[1]-bics[2])/bics[1]*100, 2),
                             detail=f"k=2改善{(bics[1]-bics[2])/bics[1]*100:.2f}%, "
                                    f"k=6改善{(bics[1]-bics[6])/bics[1]*100:.2f}%"))
    # noslope_nogate：仅 f5 实测（非门控）reach
    rng = np.random.default_rng(RNG)
    cols = VARIANTS["noslope"]
    rv = raw[cols][d.f5_r2.notna().values]
    Xv = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             random_state=RNG).fit_transform(rv.values)
    Zv = PCA(n_components=4, random_state=RNG).fit_transform(Xv)
    Xs = Zv[rng.choice(len(Zv), min(50_000, len(Zv)), replace=False)]
    mu, cov = Xs.mean(0), np.cov(Xs.T)
    Xref = rng.multivariate_normal(mu, cov, size=len(Xs))
    for dtag, Xd in [("real", Xs), ("unimodal_ref", Xref)]:
        print(f"== noslope_nogate/{dtag} (n={len(rv)}) ==", flush=True)
        bics = bic_curve(Xd, range(1, 7))
        rows.append(dict(variant="noslope_nogate", test=f"gmm_{dtag}",
                         stat=round((bics[1]-bics[2])/bics[1]*100, 2),
                         detail=f"k=2改善{(bics[1]-bics[2])/bics[1]*100:.2f}%, "
                                f"k=6改善{(bics[1]-bics[6])/bics[1]*100:.2f}%"))
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/gmm_bic_ablate_converged.csv", index=False,
             encoding="utf-8-sig")
    print(t.to_string(index=False))


if __name__ == "__main__":
    {"full": run_full, "ablate": run_ablate}[sys.argv[1]]()
