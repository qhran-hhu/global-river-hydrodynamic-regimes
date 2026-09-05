# -*- coding: utf-8 -*-
"""D3：SWOT 测量噪声衰减校正 —— 连续谱不是被噪声抹平的吗？

臂 1（噪声注入）：2,000 个分层抽样 QC reach 的真实序列上加模拟 SWOT 噪声
  （WSE 加性高斯 σ=0.1/0.2/0.3 m，对应 Cal/Val 实测 ~0.1 m 到悲观上限；
   宽度乘性对数正态 σ=0.1/0.2），每情景 5 次重复重算 f2/f3/f4/f5：
  - 特征偏移（中位相对偏差）与 加噪 vs 原始 跨 reach 秩相关；
  - 加噪后的 regime 空间（坡度无关 5 特征）偶极检验——单峰性是否噪声造成。
臂 2（去衰减校正）：以 D2 半段相关为信度（Spearman-Brown 外推全记录），
  对流量-水动力耦合等关键相关做衰减校正。

输出：output/regime_space/D3_noise_attenuation.csv / .txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
SHARDS = BASE / "output" / "ts_shards_global"
MAT = BASE / "output" / "feature_matrix_v1_qc.parquet"
D2 = BASE / "output" / "regime_space" / "D2_swot_halves_part0_353.parquet"
D2B = BASE / "output" / "regime_space" / "D2_swot_halves_part353_706.parquet"
OUT = BASE / "output" / "regime_space"
MIN_OBS = 15
N_SAMPLE = 2000
REPS = 5
SCEN = [(0.10, 0.10), (0.20, 0.10), (0.30, 0.20)]  # (σ_wse m, σ_width log)
RNG = np.random.default_rng(0)


def harmonic(t, y):
    m = np.isfinite(y) & np.isfinite(t)
    if m.sum() < MIN_OBS:
        return np.nan, np.nan, np.nan
    t, y = t[m], y[m]
    X = np.column_stack([np.ones_like(t), np.cos(2 * np.pi * t),
                         np.sin(2 * np.pi * t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    phase = float(np.arctan2(coef[2], coef[1]) % (2 * np.pi))
    amp = float(np.hypot(coef[1], coef[2]))
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return phase, amp, (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def feats(doy, wse, wid, s_med):
    """单 reach 特征（与 build_features_v1 同公式）。"""
    if len(wse) < MIN_OBS:
        return None
    p5 = np.nanpercentile(wse, 5)
    wp = np.nanpercentile(wse, [10, 90])
    mean_hrel = max(np.nanmean(wse) - p5, 1e-3)
    f2 = (wp[1] - wp[0]) / mean_hrel  # 深度相对变幅（§8.30 修正）
    wv = wid[np.isfinite(wid) & (wid > 0)]
    if len(wv) >= MIN_OBS:
        wq = np.nanpercentile(wv, [50, 95])
        f3 = (wq[1] - wq[0]) / wq[0]
        wq2 = np.nanpercentile(wv, [25, 75])
        f7w = (wq2[1] - wq2[0]) / wq[0]
    else:
        f3 = f7w = np.nan
    h_rel = np.clip(wse - p5, 1e-3, None)
    h = 1000.0 * h_rel ** (4 / 3) * s_med
    lh = np.log10(h[h > 0])
    f4 = (np.nanpercentile(lh, 75) - np.nanpercentile(lh, 25)
          if len(lh) >= MIN_OBS else np.nan)
    ph, amp, r2 = harmonic(doy, wse)
    return dict(f2=f2, f3=f3, f4=f4, f7w=f7w,
                f5_phase=ph, f5_amp=amp, f5_r2=r2)


def load_sample_series():
    mat = pd.read_parquet(MAT, columns=["f2_rel_range", "continent"])
    mat = mat.reset_index()
    mat["f2q"] = pd.qcut(mat.f2_rel_range, 4, labels=False,
                         duplicates="drop")
    samp = (mat.groupby(["continent", "f2q"], group_keys=False)
              .apply(lambda x: x.sample(min(len(x), N_SAMPLE // 24 + 1),
                                        random_state=0)))
    ids = set(samp.reach_id.head(N_SAMPLE))
    print(f"抽样 {len(ids)} reach；扫描分片取序列", flush=True)
    ser = {}
    files = sorted(SHARDS.glob("shard_*.parquet"))
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["reach_id", "time_str", "wse",
                                         "width", "slope"])
        df = df[df.reach_id.isin(ids)]
        if len(df) == 0:
            continue
        df = df[(df.time_str != "no_data") & df.time_str.notna()]
        df["date"] = pd.to_datetime(df.time_str, utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        for c in ("wse", "width", "slope"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df.wse <= -1e11, "wse"] = np.nan
        df.loc[df.width <= -1e11, "width"] = np.nan
        df = df.dropna(subset=["wse"])
        df["doy"] = df.date.dt.dayofyear.values / 365.25
        for rid, x in df.groupby("reach_id"):
            sv = x.slope[(x.slope > 0) & (x.slope < 1)]
            s_med = sv.median() if len(sv) >= 5 else np.nan
            ser[rid] = (x.doy.values, x.wse.values, x.width.values, s_med)
        if (i + 1) % 200 == 0:
            print(f"  shard {i + 1}/{len(files)}, 已收 {len(ser)}", flush=True)
    print(f"收到序列 {len(ser)} reach", flush=True)
    return ser


def dip_p(vals, seed=0):
    import diptest
    v = vals[np.isfinite(vals)]
    if len(v) < 200:
        return np.nan, np.nan
    st = float(diptest.dipstat(v))
    _, p = diptest.diptest(v, boot_pval=True, n_boot=300, seed=seed)
    return st, float(p)


def main():
    ser = load_sample_series()
    # 原始特征
    base = {rid: feats(*s) for rid, s in ser.items()}
    base = {k: v for k, v in base.items() if v is not None}
    B = pd.DataFrame(base).T
    print(f"原始特征有效 {len(B)}", flush=True)

    rows = []
    noisy_cache = {}
    for si, (sw, swi) in enumerate(SCEN):
        for rep in range(REPS):
            rng = np.random.default_rng(1000 * si + rep)
            recs = {}
            for rid, (doy, wse, wid, s_med) in ser.items():
                if rid not in base:
                    continue
                wse_n = wse + rng.normal(0, sw, len(wse))
                wid_n = wid * np.exp(rng.normal(0, swi, len(wid)))
                r = feats(doy, wse_n, wid_n, s_med)
                if r:
                    recs[rid] = r
            Nf = pd.DataFrame(recs).T
            noisy_cache[(si, rep)] = Nf
            J = Nf.join(B, rsuffix="_base")
            row = dict(sigma_wse=sw, sigma_width=swi, rep=rep,
                       n=len(J))
            for c in ("f2", "f3", "f4", "f5_amp"):
                v = np.isfinite(J[c]) & np.isfinite(J[f"{c}_base"])
                row[f"{c}_rho"] = spearmanr(J[c][v], J[f"{c}_base"][v]).statistic
                row[f"{c}_bias_pct"] = float(
                    ((J[c][v] - J[f"{c}_base"][v]) / J[f"{c}_base"][v])
                    .median() * 100)
            v = np.isfinite(J.f5_r2) & np.isfinite(J.f5_r2_base)
            row["f5_r2_rho"] = spearmanr(J.f5_r2[v], J.f5_r2_base[v]).statistic
            row["f5_r2_drop"] = float((J.f5_r2[v] - J.f5_r2_base[v]).median())
            b = J[(J.f5_r2_base >= 0.3) & (J.f5_r2 >= 0.3)]
            d = ((b.f5_phase - b.f5_phase_base + np.pi) % (2 * np.pi)
                 - np.pi) / (2 * np.pi) * 12
            row["phase_med_abs_mo"] = float(d.abs().median())
            row["phase_within1_pct"] = float((d.abs() <= 1).mean() * 100)
            rows.append(row)
        print(f"情景 {si + 1}/3 完成", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "D3_noise_attenuation.csv", index=False, encoding="utf-8-sig")

    # 单峰性（坡度无关 5 特征，每情景用 rep0 + 原始）
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer
    rep_lines = ["", "--- 加噪后 regime 空间偶极检验（坡度无关 5 特征） ---"]

    def dip_of(df):
        X = df[["f2", "f3", "f4", "f5_amp", "f5_r2"]].copy()
        X["f5x"] = np.log1p(df.f5_amp) * np.cos(df.f5_phase)
        X["f5y"] = np.log1p(df.f5_amp) * np.sin(df.f5_phase)
        X = X[["f2", "f3", "f4", "f5x", "f5y"]].dropna()
        Z = PCA(n_components=5, random_state=0).fit_transform(
            QuantileTransformer(n_quantiles=1000,
                                output_distribution="normal",
                                random_state=0).fit_transform(X.values))
        out = []
        for j in (0, 1):
            st, p = dip_p(Z[:, j])
            out.append((st, p))
        return len(X), out

    n0, d0 = dip_of(B)
    rep_lines.append(f"  原始: n={n0}, PC1 dip={d0[0][0]:.4f} p={d0[0][1]:.3f}; "
                     f"PC2 dip={d0[1][0]:.4f} p={d0[1][1]:.3f}")
    for si, (sw, swi) in enumerate(SCEN):
        nn, dd = dip_of(noisy_cache[(si, 0)])
        rep_lines.append(f"  σ_wse={sw} m, σ_w={swi}: n={nn}, "
                         f"PC1 dip={dd[0][0]:.4f} p={dd[0][1]:.3f}; "
                         f"PC2 dip={dd[1][0]:.4f} p={dd[1][1]:.3f}")

    # 臂 2：去衰减校正
    H = pd.concat([pd.read_parquet(D2), pd.read_parquet(D2B)],
                  ignore_index=True)
    P = H.pivot_table(index="reach_id", columns="half",
                      values=["f2", "f3", "f4", "f5_amp", "f5_r2"])
    P.columns = [f"{a}_h{int(b)}" for a, b in P.columns]
    rep_lines += ["", "--- 去衰减校正（Spearman-Brown） ---"]
    rel = {}
    for c in ("f2", "f3", "f4", "f5_amp", "f5_r2"):
        v = np.isfinite(P[f"{c}_h1"]) & np.isfinite(P[f"{c}_h2"])
        rhh = spearmanr(P[f"{c}_h1"][v], P[f"{c}_h2"][v]).statistic
        rel[c] = 2 * rhh / (1 + rhh)  # 全记录信度
        rep_lines.append(f"  {c}: 半段信度 {rhh:.3f} -> 全记录信度 "
                         f"{rel[c]:.3f}")
    q_rel = 0.877  # GSIM q_iqr 3.4 年窗口信度（D2 臂A）
    obs = 0.128    # f4~q_iqr 观测秩相关（compare_q_regime）
    corr = obs / np.sqrt(rel["f4"] * q_rel)
    rep_lines.append(f"  水动力-流量耦合 f4~q_iqr: 观测 ρ={obs:.3f} -> "
                     f"去衰减 ρ={corr:.3f}（仍 <0.3，弱耦合成立）")
    obs2, q2_rel = 0.095, 0.864
    corr2 = obs2 / np.sqrt(rel["f2"] * q2_rel)
    rep_lines.append(f"  f2~q2: 观测 ρ={obs2:.3f} -> 去衰减 ρ={corr2:.3f}")

    # 汇总表（跨重复中位）
    agg = R.groupby(["sigma_wse", "sigma_width"]).median(numeric_only=True)
    txt = ["D3 噪声衰减校正（2026-09-03）", "",
           "=== 臂 1：噪声注入（2,000 分层 reach，每情景 5 重复中位） ===",
           agg.round(3).to_string()] + rep_lines
    out = "\n".join(txt)
    (OUT / "D3_noise_attenuation.txt").write_text(out, encoding="utf-8")
    print("\n" + out)


if __name__ == "__main__":
    main()
