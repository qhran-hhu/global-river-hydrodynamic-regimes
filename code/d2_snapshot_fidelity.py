# -*- coding: utf-8 -*-
"""D2：短记录保真度 —— "3.4 年够不够"的正面回答

臂 A（GSIM）：长记录流量站（≥20 年）上，3.4 年（41 个月）随机窗口算出的
  regime 特征 vs 全记录特征——检验"短记录能否抓住 regime"。
臂 B（SWOT）：每个 QC reach 的记录按中位日期劈成前后两半（各 ~1.7 年），
  分别重算 f2/f3/f4/f5，跨 reach 检验半段间一致性 + 半段 vs 全记录一致性。

输出：
  output/regime_space/D2_snapshot_fidelity.txt   （汇总，人读）
  output/regime_space/D2_gsim_windows.csv （窗口级明细）
  output/regime_space/D2_swot_halves.parquet
"""
from pathlib import Path
import argparse
import io
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
SHARDS = BASE / "output" / "ts_shards_global"
GSIM = BASE / "output" / "regime_space" / "gsim"
MAT = BASE / "output" / "feature_matrix_v1_qc.parquet"
OUT = BASE / "output" / "regime_space"
MIN_OBS = 12          # 半段/窗口最少观测数
WIN_M = 41            # 3.4 年 ≈ 41 个月
K_WIN = 5             # 每站随机窗口数
RNG = np.random.default_rng(0)


def harmonic(t, y, min_obs=MIN_OBS):
    """年谐波 NaN 掩膜 lstsq，返回 (phase, amp, r2)。t 单位：年（含小数）。"""
    m = np.isfinite(y) & np.isfinite(t)
    if m.sum() < min_obs:
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
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return phase, amp, r2


def harmonic_monthly(dates, q):
    """与 compare_q_regime 同口径：月中时间基。"""
    t = (dates.dt.year + (dates.dt.month - 0.5) / 12).values
    return harmonic(t, np.asarray(q, float))


def qfeats(tdf):
    """月流量序列 → 特征 dict。"""
    q = tdf.MEAN.values.astype(float)
    q = q[np.isfinite(q) & (q > 0)]
    if len(q) < MIN_OBS:
        return None
    p = np.percentile(q, [10, 25, 50, 75, 90, 95])
    ph, amp, r2 = harmonic_monthly(tdf.date[np.isfinite(tdf.MEAN.values)],
                                   tdf.MEAN.values[np.isfinite(tdf.MEAN.values)])
    return dict(q2=(p[4] - p[0]) / p[2],
                q_iqr=np.log10(p[3]) - np.log10(p[1]),
                q_ev=(p[5] - p[2]) / p[2],
                q_phase=ph, q_amp=amp, q_r2=r2, n=len(q))


def circ_months(dph_rad):
    return dph_rad / (2 * np.pi) * 12


# ================= 臂 A：GSIM =================
def arm_a():
    meta = pd.read_csv(GSIM / "catalog" / "GSIM_metadata.csv",
                       low_memory=False)
    susp = pd.read_csv(GSIM / "catalog" /
                       "GSIM_suspect_coordinates_stations.csv")
    meta = meta[(meta["year.no"] >= 20) &
                (meta["frac.missing.days"] <= 0.2) &
                (~meta["gsim.no"].isin(susp["gsim.no"]))]
    print(f"臂A 站点（≥20年）: {len(meta)}", flush=True)

    z = zipfile.ZipFile(GSIM / "GSIM_indices.zip")
    names = set(z.namelist())
    rows_full, rows_win = {}, []
    done = 0
    for gs in meta["gsim.no"]:
        key = f"GSIM_indices/TIMESERIES/monthly/{gs}.mon"
        if key not in names:
            continue
        try:
            with z.open(key) as f:
                txt = f.read().decode("utf-8", "replace")
            txt = txt.replace('"', "").replace("\t", ",")
            t = pd.read_csv(io.StringIO(txt), comment="#",
                            parse_dates=["date"])
        except Exception:
            continue
        t["MEAN"] = pd.to_numeric(t.MEAN, errors="coerce")
        if len(t) < 240:
            continue
        full = qfeats(t)
        if full is None:
            continue
        rows_full[gs] = full
        # K 个随机 41 月窗口
        latest_start = len(t) - WIN_M
        if latest_start < 1:
            continue
        for k in range(K_WIN):
            s = int(RNG.integers(0, latest_start))
            w = qfeats(t.iloc[s:s + WIN_M])
            if w is None:
                continue
            rows_win.append(dict(gsim=gs, win=k, **w))
        done += 1
        if done % 2000 == 0:
            print(f"  parsed {done}", flush=True)

    F = pd.DataFrame(rows_full).T
    W = pd.DataFrame(rows_win)
    W.to_csv(OUT / "D2_gsim_windows.csv", index=False, encoding="utf-8-sig")
    J = W.join(F, on="gsim", rsuffix="_full")
    print(f"臂A 有效站 {len(F)}，窗口 {len(W)}", flush=True)

    rep = ["===== 臂 A：GSIM 3.4 年窗口 vs 全记录 =====",
           f"站点 {len(F)}（≥20 年记录），随机窗口 {len(W)} 个（41 月/个）"]
    for c, lab in [("q2", "相对变幅 q2"), ("q_iqr", "IQR(logQ)"),
                   ("q_ev", "事件响应 q_ev"), ("q_amp", "年谐波振幅"),
                   ("q_r2", "年谐波 R²")]:
        v = np.isfinite(J[c]) & np.isfinite(J[f"{c}_full"])
        r = spearmanr(J[c][v], J[f"{c}_full"][v]).statistic
        rep.append(f"  {lab}: Spearman ρ = {r:.3f} (n={v.sum()})")
    # 相位一致性：双方 R²≥0.3
    b = J[(J.q_r2 >= 0.3) & (J.q_r2_full >= 0.3)]
    d = circ_months((b.q_phase - b.q_phase_full + np.pi) % (2 * np.pi) - np.pi)
    rep.append(f"  相位差（双方 R²≥0.3, n={len(b)}）: 中位 |Δ| = "
               f"{d.abs().median():.2f} 月; ≤1 月 {(d.abs() <= 1).mean() * 100:.1f}%; "
               f"≤2 月 {(d.abs() <= 2).mean() * 100:.1f}%")
    return rep


# ================= 臂 B：SWOT 前后半段 =================
def arm_b(s0=0, s1=0, tag=""):
    files = sorted(SHARDS.glob("shard_*.parquet"))
    if s1:
        files = files[s0:s1]
    print(f"臂B 分片 {len(files)} 个", flush=True)
    parts = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        df = df[(df.time_str != "no_data") & df.time_str.notna()]
        df["date"] = pd.to_datetime(df.time_str, utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        for c in ("wse", "width", "slope"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df.wse <= -1e11, "wse"] = np.nan
        df.loc[df.width <= -1e11, "width"] = np.nan
        df = df.dropna(subset=["wse"])
        df["doy"] = df.date.dt.dayofyear.values / 365.25
        sm = df[(df.slope > 0) & (df.slope < 1)].groupby("reach_id").slope.median()
        df["s_med"] = df.reach_id.map(sm)
        med = df.groupby("reach_id").date.transform("median")
        df["half"] = (df.date > med).astype(int) + 1
        df = df.groupby(["reach_id", "half"]).filter(lambda x: len(x) >= MIN_OBS)
        if len(df) == 0:
            continue
        g = df.groupby(["reach_id", "half"])
        # f2: (P90-P10)/mean(h_rel)，h_rel = wse - P5（深度相对变幅，§8.30 修正）
        wp = g.wse.quantile([0.05, 0.1, 0.9]).unstack()
        mean_hrel = (g.wse.mean() - wp[0.05]).clip(lower=1e-3)
        hf = pd.DataFrame(index=wp.index)
        hf["f2"] = (wp[0.9] - wp[0.1]) / mean_hrel
        # f3: (P95-P50)/P50 of width
        wq = g.width.quantile([0.5, 0.95]).unstack()
        hf["f3"] = (wq[0.95] - wq[0.5]) / wq[0.5]
        # f4: IQR(log10 h)，h_rel 用半段内 P5
        p5 = g.wse.quantile(0.05)
        key = pd.MultiIndex.from_frame(df[["reach_id", "half"]])
        h_rel = (df.wse - pd.Series(key.map(p5), index=df.index)).clip(lower=1e-3)
        sv = (df.slope > 0) & (df.slope < 1)
        s_use = np.where(sv, df.slope, df.s_med)
        h = 1000.0 * h_rel.values ** (4 / 3) * np.asarray(s_use, float)
        lh = pd.Series(np.log10(np.where(h > 0, h, np.nan)), index=df.index)
        lq = lh.groupby(key).quantile([0.25, 0.75]).unstack()
        hf["f4"] = lq[0.75] - lq[0.25]
        hf["n"] = g.size()
        # f5: 谐波（逐组 lstsq）
        harm = g.apply(lambda x: harmonic(x.doy.values, x.wse.values),
                       include_groups=False)
        hf["f5_phase"] = harm.apply(lambda t: t[0])
        hf["f5_amp"] = harm.apply(lambda t: t[1])
        hf["f5_r2"] = harm.apply(lambda t: t[2])
        parts.append(hf.reset_index())
        if (i + 1) % 50 == 0:
            print(f"  shard {i + 1}/{len(files)}", flush=True)
    H = pd.concat(parts, ignore_index=True)
    H.to_parquet(OUT / f"D2_swot_halves{tag}.parquet")
    return H


def report_b(H=None):
    """由半段特征表出报告（可单独对已有 parquet 重跑）。"""
    if H is None:
        files = sorted(OUT.glob("D2_swot_halves_part*.parquet"))
        H = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    P = H.pivot_table(index="reach_id", columns="half",
                      values=["f2", "f3", "f4", "f5_phase", "f5_amp",
                              "f5_r2", "n"])
    P.columns = [f"{a}_h{int(b)}" for a, b in P.columns]
    both = P.dropna(subset=["n_h1", "n_h2"])
    print(f"臂B 两半皆有效的 reach: {len(both)}", flush=True)

    rep = ["", "===== 臂 B：SWOT 前半段 vs 后半段（各 ~1.7 年） =====",
           f"两半皆 ≥{MIN_OBS} 次观测的 reach: {len(both)}"]
    for c, lab in [("f2", "WSE 相对变幅 f2"), ("f3", "宽度事件响应 f3"),
                   ("f4", "IQR(logH) f4"), ("f5_amp", "年谐波振幅"),
                   ("f5_r2", "年谐波 R²")]:
        v = np.isfinite(both[f"{c}_h1"]) & np.isfinite(both[f"{c}_h2"])
        r = spearmanr(both[f"{c}_h1"][v], both[f"{c}_h2"][v]).statistic
        rep.append(f"  {lab}: Spearman ρ = {r:.3f} (n={v.sum()})")
    b = both[(both.f5_r2_h1 >= 0.3) & (both.f5_r2_h2 >= 0.3)]
    d = circ_months((b.f5_phase_h1 - b.f5_phase_h2 + np.pi) % (2 * np.pi) - np.pi)
    rep.append(f"  相位差（两半 R²≥0.3, n={len(b)}）: 中位 |Δ| = "
               f"{d.abs().median():.2f} 月; ≤1 月 {(d.abs() <= 1).mean() * 100:.1f}%; "
               f"≤2 月 {(d.abs() <= 2).mean() * 100:.1f}%")

    # 半段 vs 全记录（补丁后矩阵）
    mat = pd.read_parquet(MAT, columns=["f2_rel_range",
                                        "f3_event_resp", "f4_iqr_logh",
                                        "f5_amp", "f5_r2", "f5_phase"])
    mat = mat.reset_index()  # reach_id 是 parquet 索引
    M = both.join(mat.set_index("reach_id"))
    rep.append("  --- 半段 vs 全记录（3.4 年） ---")
    for c, fc, lab in [("f2", "f2_rel_range", "f2"),
                       ("f3", "f3_event_resp", "f3"),
                       ("f4", "f4_iqr_logh", "f4"),
                       ("f5_amp", "f5_amp", "振幅"),
                       ("f5_r2", "f5_r2", "R²")]:
        for h in (1, 2):
            v = np.isfinite(M[f"{c}_h{h}"]) & np.isfinite(M[fc])
            r = spearmanr(M[f"{c}_h{h}"][v], M[fc][v]).statistic
            rep.append(f"    {lab} 半段{h}~全记录: ρ = {r:.3f} (n={v.sum()})")
    b2 = M[(M.f5_r2_h1 >= 0.3) & (M.f5_r2 >= 0.3)]
    d2 = circ_months((b2.f5_phase_h1 - b2.f5_phase + np.pi) % (2 * np.pi) - np.pi)
    rep.append(f"    相位 半段1~全记录（双方 R²≥0.3, n={len(b2)}）: 中位 |Δ| = "
               f"{d2.abs().median():.2f} 月; ≤1 月 {(d2.abs() <= 1).mean() * 100:.1f}%")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b", "report", "all"],
                    default="all")
    ap.add_argument("--s0", type=int, default=0)
    ap.add_argument("--s1", type=int, default=0)
    args = ap.parse_args()
    if args.arm == "a":
        rep = ["D2 短记录保真度检验（2026-09-03）", ""] + arm_a()
        (OUT / "D2_snapshot_fidelity.txt").write_text("\n".join(rep),
                                               encoding="utf-8")
        print("\n" + "\n".join(rep))
    elif args.arm == "b":
        tag = f"_part{args.s0}_{args.s1}" if args.s1 else ""
        arm_b(args.s0, args.s1, tag)
    elif args.arm == "report":
        txt = "\n".join(report_b())
        (OUT / "D2_armB_summary.txt").write_text(txt, encoding="utf-8")
        print("\n" + txt)
    else:
        rep = ["D2 短记录保真度检验（2026-09-03）", ""]
        rep += arm_a()
        rep += report_b(arm_b())
        txt = "\n".join(rep)
        (OUT / "D2_snapshot_fidelity.txt").write_text(txt, encoding="utf-8")
        print("\n" + txt)


if __name__ == "__main__":
    main()
