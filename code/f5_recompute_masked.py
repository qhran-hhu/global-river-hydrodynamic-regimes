# -*- coding: utf-8 -*-
"""f5 门控 bug 量化：掩膜版谐波重算（只读诊断，不改任何管线文件）

背景：build_features_v1.harmonic_one 对含 NaN 的 WSE 序列直接 lstsq，
NaN 传播导致 38.5% reach 的 f5_* 全为 NaN（被当作"门控/事件主导"）。
本脚本用 NaN 掩膜重算全部 reach 的 (phase, amp, r2)，输出：
  output/regime_space/f5_masked_refit.parquet  （reach_id, phase, amp, r2, n_eff）
并打印：新门控率（R²<0.3）、纬度型态、与旧矩阵对比。
  python f5_recompute_masked.py [--limit N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SHARDS = BASE / "output" / "ts_shards_global"
OUT = BASE / "output" / "regime_space"
MIN_OBS = 15


def harmonic_masked(t, y):
    m = np.isfinite(y) & np.isfinite(t)
    if m.sum() < MIN_OBS:
        return np.nan, np.nan, np.nan, int(m.sum())
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
    return phase, amp, r2, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(SHARDS.glob("shard_*.parquet"))
    if args.limit:
        files = files[:args.limit]
    print(f"分片 {len(files)} 个", flush=True)

    recs = {}
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["reach_id", "time_str", "wse"])
        df = df[(df.time_str != "no_data") & df.time_str.notna()]
        df["date"] = pd.to_datetime(df.time_str, utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df["wse"] = pd.to_numeric(df.wse, errors="coerce")
        df.loc[df.wse <= -1e11, "wse"] = np.nan
        df["doy"] = df.date.dt.dayofyear.values / 365.25
        for rid, x in df.groupby("reach_id"):
            recs[rid] = harmonic_masked(x.doy.values, x.wse.values)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} 分片, reach 累计 {len(recs)}",
                  flush=True)

    t = pd.DataFrame([(r, *v) for r, v in recs.items()],
                     columns=["reach_id", "f5_phase_m", "f5_amp_m",
                              "f5_r2_m", "n_eff"]).set_index("reach_id")
    t.to_parquet(OUT / "f5_masked_refit.parquet")
    print(f"保存 {OUT / 'f5_masked_refit.parquet'}: {t.shape}")

    # ---- 与旧矩阵对比 ----
    qc = pd.read_parquet(BASE / "output" / "feature_matrix_v1_qc.parquet",
                         columns=["f5_r2", "f5_amp", "f5_phase", "y",
                                  "f2_rel_range", "n_obs"])
    m = qc.join(t, how="left")
    old_gate = m.f5_r2.isna()
    new_gate03 = m.f5_r2_m < 0.3
    print(f"\n矩阵内 reach: {len(m)}, 掩膜重算成功: {m.f5_r2_m.notna().sum()}")
    print(f"旧门控率（r2 NaN）: {old_gate.mean()*100:.1f}%")
    print(f"新口径 R²<0.3 占比: {new_gate03.mean()*100:.1f}%")
    print(f"新口径 R²≥0.3 占比: {(m.f5_r2_m >= 0.3).mean()*100:.1f}%")
    # 纬度型态（新口径）
    for lo, hi, lab in [(-90, 55 * 0 + 0, None)]:
        pass
    bins = [(-90, -66.5), (-66.5, -55), (-55, -23.5), (-23.5, 23.5),
            (23.5, 55), (55, 66.5), (66.5, 90)]
    print("\n纬度带 | 旧门控% | 新R²<0.3% | 新R²中位")
    for lo, hi in bins:
        s = m[(m.y >= lo) & (m.y < hi)]
        if len(s) == 0:
            continue
        print(f"  [{lo:+6.1f},{hi:+6.1f}): {s.f5_r2.isna().mean()*100:5.1f}% |"
              f" {(s.f5_r2_m < 0.3).mean()*100:5.1f}% |"
              f" {s.f5_r2_m.median():.3f}  (n={len(s)})")
    # 旧门控群体在新口径下的 r2 分布
    og = m[old_gate]
    print(f"\n旧门控群体 (n={len(og)}) 掩膜后 r2 分布: "
          f"中位 {og.f5_r2_m.median():.3f}, "
          f"R²≥0.3 占 {(og.f5_r2_m >= 0.3).mean()*100:.1f}%")
    vg = m[~old_gate]
    print(f"旧有效群体 (n={len(vg)}): r2 中位 {vg.f5_r2_m.median():.3f}, "
          f"R²≥0.3 占 {(vg.f5_r2_m >= 0.3).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
