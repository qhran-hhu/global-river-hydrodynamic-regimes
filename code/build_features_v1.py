# -*- coding: utf-8 -*-
"""全量特征矩阵 v1 构建器（向量化生产版，139k reach 规模）。

与 features.py（小样本参考实现）公式完全一致，但：
  - 全表一次清洗（向量化 to_datetime / 填充值剔除）
  - groupby 聚合算 f2/f3/f4/f7 原料（不做逐 reach DataFrame 扫描）
  - 谐波拟合逐 reach lstsq（~1 ms/reach，可承受）
  - 分片级 checkpoint：每个输入分片输出一个 part parquet，中断可续

用法：python build_features_v1.py [--budget 270]
输出：output/features_v1_parts/part_<shard名>（每片 ~200 reach 的特征行）
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SHARDS = BASE / "output" / "ts_shards_global"
META_CSV = BASE / "output" / "全球reach清单.csv"
PARTS = BASE / "output" / "features_v1_parts"
MIN_OBS = 15
RHO = 1000.0


def harmonic_one(t, y):
    """单 reach 年循环一次谐波 lstsq，返回 (phase, amp, r2)。"""
    if len(y) < MIN_OBS:
        return np.nan, np.nan, np.nan
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


def process_shard(f, slope_map):
    """单个时间序列分片 → 特征行 DataFrame。"""
    df = pd.read_parquet(f)
    df = df[(df.time_str != "no_data") & df.time_str.notna()]
    df["date"] = pd.to_datetime(df.time_str, utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ("wse", "width", "slope"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -1e11, c] = np.nan
    df["doy"] = df.date.dt.dayofyear / 365.25

    g = df.groupby("reach_id")
    n_obs = g.size()
    keep = n_obs[n_obs >= MIN_OBS].index
    df = df[df.reach_id.isin(keep)]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("reach_id")

    out = pd.DataFrame({"n_obs": g.size()})
    # f2: (P90-P10)/P50 of WSE
    wp = g.wse.quantile([0.1, 0.5, 0.9]).unstack()
    out["f2_rel_range"] = (wp[0.9] - wp[0.1]) / wp[0.5]
    # f3: (P95-P50)/P50 of width；f7 原料：宽度 IQR/P50
    wq = g.width.quantile([0.25, 0.5, 0.75, 0.95]).unstack()
    out["f3_event_resp"] = (wq[0.95] - wq[0.5]) / wq[0.5]
    out["width_iqr_ratio"] = (wq[0.75] - wq[0.25]) / wq[0.5]
    # 坡度：观测中位（正值）缺省用先验
    s_obs = g.slope.apply(lambda s: s[(s > 0) & (s < 1)].median()
                          if ((s > 0) & (s < 1)).sum() >= 5 else np.nan)
    out["f6_slope"] = s_obs.fillna(pd.Series(slope_map)).reindex(out.index)
    # H_proxy（路径甲）：h_rel = wse - P5(wse)；u² ∝ h_rel^(4/3)·S
    p5 = g.wse.quantile(0.05)
    df["h_rel"] = (df.wse - df.reach_id.map(p5)).clip(lower=1e-3)
    slope_row = df.reach_id.map(out.f6_slope)
    s_valid = (df.slope > 0) & (df.slope < 1)
    df["s_use"] = np.where(s_valid, df.slope, slope_row)
    df["h"] = RHO * df.h_rel ** (4 / 3) * df.s_use
    gh = df.groupby("reach_id").h
    out["med_hproxy"] = gh.median()
    logh = np.log10(df.h.where(df.h > 0))
    lq = logh.groupby(df.reach_id).quantile([0.25, 0.75]).unstack()
    out["f4_iqr_logh"] = lq[0.75] - lq[0.25]
    # f5：谐波拟合（逐 reach lstsq）
    harm = df.groupby("reach_id").apply(
        lambda x: harmonic_one(x.doy.values, x.wse.values),
        include_groups=False)
    out["f5_phase"] = harm.apply(lambda t: t[0])
    out["f5_amp"] = harm.apply(lambda t: t[1])
    out["f5_r2"] = harm.apply(lambda t: t[2])
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=270)
    args = ap.parse_args()
    PARTS.mkdir(exist_ok=True)
    meta = pd.read_csv(META_CSV, usecols=["reach_id", "momma_slope"])
    slope_map = {k: v for k, v in
                 meta.set_index("reach_id").momma_slope.to_dict().items()
                 if np.isfinite(v)}  # NaN 键值必须丢弃，否则会覆盖填补值
    # 第二回退：邻域填补坡度（make_slope_fill.py 生成）
    fill_p = BASE / "output" / "slope_fill.parquet"
    if fill_p.exists():
        fill = pd.read_parquet(fill_p).slope.to_dict()
        slope_map = {**fill, **slope_map}  # momma 优先，邻域填补兜底
        print(f"已载入邻域坡度填补 {len(fill)} 条", flush=True)

    shards = sorted(SHARDS.glob("shard_*.parquet"))
    done = {p.stem.replace("part_", "") for p in PARTS.glob("part_*.parquet")}
    todo = [f for f in shards if f.stem not in done]
    print(f"分片 {len(shards)}，已完成 {len(done)}，待处理 {len(todo)}",
          flush=True)
    t0, n = time.time(), 0
    for f in todo:
        if time.time() - t0 > args.budget:
            break
        try:
            part = process_shard(f, slope_map)
            part.to_parquet(PARTS / f"part_{f.stem}.parquet", index=False)
            n += 1
        except Exception as ex:
            print(f"  !! {f.stem}: {ex}", flush=True)
        if n % 50 == 0 and n:
            print(f"  进度 {n}/{len(todo)}（{time.time()-t0:.0f}s）",
                  flush=True)
    print(f"本轮处理 {n} 片，累计 part {len(list(PARTS.glob('part_*.parquet')))}",
          flush=True)


if __name__ == "__main__":
    main()
