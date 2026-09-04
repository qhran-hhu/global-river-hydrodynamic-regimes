# -*- coding: utf-8 -*-
"""聚类与双路径 ARI 终检模块（阶段 1 再决策点专用）。

设计（方案 §6）：
  - Ward + GMM 双方法互验
  - k 由 GMM BIC + bootstrap ARI 稳定性联合选择
  - 聚类前稳健缩放（median/IQR）+ 1/99 分位截尾（v0 教训：单点离群类）
  - f5 相位仅对 R²≥0.3 的 reach 启用（周期编码 sin/cos，否则置中性 0）

终检逻辑：
  路径甲 f1 来自纯观测 H_proxy（Manning 型）；
  路径乙 f1 来自 SoS consensus_q 经河宽分层校正后的 H（hproxy_path_b）。
  其余特征 f2–f7 两路径相同（均来自 Hydrocron 观测）。
  两路径分别聚类后在公共 reach 上算 ARI —— 全球尺度（~9 万）无小样本伪影。

main：用试拉 369 reach + SoS Q 做预览（样本小，仅验证代码通路）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from features import hproxy_path_b, width_bias_correction

R2_MIN = 0.3
FCOLS_BASE = ["f2_rel_range", "f3_event_resp", "f4_iqr_logh"]


# ---------- 路径乙 f1 ----------

def f1_path_b(reach_table):
    """路径乙水平秩次：SoS consensus_q（缺省回退 momma_q/metroman_q 中位）
    经河宽分层校正 → H = ρ(Q/WD)² → 全球分位数秩。
    返回 Series（index=reach_id），无法计算处为 NaN。"""
    rt = reach_table.set_index("reach_id")
    q = rt["consensus_q"].copy()
    q = q.where(q > 0)
    q = q.fillna(rt.get("momma_q")).fillna(rt.get("metroman_q"))
    w = rt["momma_width"].where(rt["momma_width"] > 0)
    d = rt["momma_depth"].where(rt["momma_depth"] > 0)
    h = pd.Series(hproxy_path_b(q.values, w.values, d.values),
                  index=rt.index)
    h = h.where(np.isfinite(h) & (h > 0))
    return np.log10(h).rank(pct=True)


# ---------- 特征准备 ----------

def prepare_X(fm, f1_col="f1_level_rank"):
    """特征矩阵 → 标准化设计矩阵。返回 (X, index)。"""
    d = fm.dropna(subset=[f1_col] + FCOLS_BASE).copy()
    use5 = d.f5_r2 >= R2_MIN if "f5_r2" in d.columns else False
    d["f5_sin"] = np.where(use5, np.sin(d.f5_phase), 0.0)
    d["f5_cos"] = np.where(use5, np.cos(d.f5_phase), 0.0)
    d["f6_log"] = np.log10(d.f6_slope.clip(lower=1e-6))
    d["f7_log"] = np.log10(d.f7_width_slope.clip(lower=1e-12)
                           .fillna(1e-12))
    cols = [f1_col] + FCOLS_BASE + ["f5_sin", "f5_cos", "f6_log", "f7_log"]
    X = d[cols].copy()
    # 1/99 分位截尾（离群修剪，v0 教训）+ 中位数填补
    for c in cols:
        lo, hi = X[c].quantile([0.01, 0.99])
        X[c] = X[c].clip(lo, hi).fillna(X[c].median())
    # 稳健缩放（median/IQR）
    med, iqr = X.median(), X.quantile(0.75) - X.quantile(0.25)
    iqr = iqr.replace(0, 1)
    return ((X - med) / iqr).values, d.index


# ---------- 聚类 ----------

def cluster_ward(X, k):
    from scipy.cluster.hierarchy import linkage, fcluster
    return fcluster(linkage(X, method="ward"), k, criterion="maxclust")


def cluster_gmm(X, k, seed=0):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(k, covariance_type="full",
                           random_state=seed).fit_predict(X)


def select_k(X, k_range=range(3, 11), n_boot=10, seed=0, n_max=20000):
    """GMM BIC + bootstrap ARI 稳定性联合选 k。
    n > n_max 时先在随机子样本上选 k（大样本标准做法）。
    返回 DataFrame（k, bic, ari_boot 中位）。"""
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        X = X[rng.choice(len(X), n_max, replace=False)]
    rows = []
    n = len(X)
    for k in k_range:
        gmm = GaussianMixture(k, covariance_type="full",
                              random_state=seed).fit(X)
        bic = gmm.bic(X)
        # bootstrap：80% 子样本 vs 全样本标签的 ARI（重复 n_boot 次）
        full_lab = gmm.predict(X)
        aris = []
        for b in range(n_boot):
            idx = rng.choice(n, int(n * 0.8), replace=False)
            g2 = GaussianMixture(k, covariance_type="full",
                                 random_state=seed + b + 1).fit(X[idx])
            aris.append(adjusted_rand_score(full_lab[idx],
                                            g2.predict(X[idx])))
        rows.append(dict(k=k, bic=bic,
                         ari_boot=float(np.median(aris)),
                         ari_p5=float(np.percentile(aris, 5))))
    return rows and pd.DataFrame(rows)


# ---------- 双路径终检 ----------

def dual_path_ari(fm, reach_table, k=None, seed=0, n_max_ward=20000):
    """fm: 特征矩阵（含 f1_level_rank 路径甲、f5_r2 等）；
    reach_table: SoS reach 表（含 consensus_q/momma_width/momma_depth）。
    Ward 在 ≤n_max_ward 随机子样本上跑（linkage 为 O(n²)，全量不可行）；
    GMM 全量跑。ARI 在各自样本集上计算。
    返回 dict(k, ari_ward, ari_gmm, n_common, pathB_coverage)。"""
    from sklearn.metrics import adjusted_rand_score
    f1b = f1_path_b(reach_table)
    fm = fm.copy()
    fm["f1_path_b"] = fm.index.map(f1b)
    both = fm.dropna(subset=["f1_level_rank", "f1_path_b"])
    if len(both) < 30:
        return dict(error=f"双路径公共 reach 太少（{len(both)}）")
    Xa, idx_a = prepare_X(both, "f1_level_rank")
    Xb, idx_b = prepare_X(both, "f1_path_b")
    assert (idx_a == idx_b).all()
    if k is None:
        ks = select_k(Xa)
        k = int(ks.loc[ks.bic.idxmin(), "k"])
    # GMM：全量
    ca_g, cb_g = cluster_gmm(Xa, k, seed), cluster_gmm(Xb, k, seed)
    ari_gmm = float(adjusted_rand_score(ca_g, cb_g))
    # Ward：子样本
    n = len(Xa)
    if n > n_max_ward:
        idx = np.random.default_rng(seed).choice(n, n_max_ward,
                                                 replace=False)
    else:
        idx = np.arange(n)
    ca_w = cluster_ward(Xa[idx], k)
    cb_w = cluster_ward(Xb[idx], k)
    ari_ward = float(adjusted_rand_score(ca_w, cb_w))
    return dict(k=k, n_common=len(both), n_ward=len(idx),
                ari_ward=ari_ward, ari_gmm=ari_gmm,
                pathB_coverage=float(fm.f1_path_b.notna().mean()))


if __name__ == "__main__":
    # 全量终检：特征矩阵 v1（QC 后 ~9 万 reach）
    OUT = BASE / "output"
    fm = pd.read_parquet(OUT / "特征矩阵_v1_qc.parquet")
    meta = fm.reset_index()[["reach_id", "consensus_q", "momma_q",
                             "metroman_q", "momma_width", "momma_depth"]]
    print(f"QC 后样本：{len(fm)}")
    print("=== 选 k（路径甲特征，2 万子样本 GMM BIC + bootstrap）===")
    X, idx = prepare_X(fm)
    ks = select_k(X)
    print(ks.round(1).to_string(index=False))
    ks.to_csv(OUT / "v1_选k_BIC_bootstrap.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 双路径 ARI 终检（全球尺度）===")
    res = dual_path_ari(fm, meta)
    print(res)
    pd.DataFrame([res]).to_csv(OUT / "v1_双路径ARI终检.csv", index=False,
                               encoding="utf-8-sig")
    verdict = ("通过（≥0.6）：特征工程锁定，进入阶段 2 勘探"
               if min(res["ari_ward"], res["ari_gmm"]) >= 0.6
               else "未达 0.6：需回查特征设计")
    print(f"\n终检判定：{verdict}")
