# -*- coding: utf-8 -*-
"""应对④：坡度邻域填补，救回因 f6 缺失被 QC 剔除的 ~1.7 万 reach。

步骤：
  1. 从已有 features_v1 parts 取有效 f6_slope，kNN(k=20) 邻域均值填补缺失
  2. 写 output/slope_fill.parquet（reach_id → slope）
  3. 删除含待救 reach 的 part 文件 → 重跑 build_features_v1.py 自动补算
     （builder 已支持读 slope_fill.parquet 作为第二回退）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
PARTS = BASE / "output" / "features_v1_parts"
SHARDS = BASE / "output" / "ts_shards_global"
META = BASE / "output" / "global_reach_list.csv"

fm = pd.concat([pd.read_parquet(f) for f in sorted(PARTS.glob("part_*.parquet"))],
               ignore_index=True).set_index("reach_id")
meta = pd.read_csv(META, usecols=["reach_id", "x", "y"]).set_index("reach_id")
fm = fm.join(meta)

ok = fm[fm.f6_slope.notna() & (fm.f6_slope > 0)]
need = fm[fm.f6_slope.isna()]
print(f"有效坡度 {len(ok)}，待填补 {len(need)}")

pts_ok = ok[["x", "y"]].values.copy()
pts_ok[:, 0] *= np.cos(np.deg2rad(pts_ok[:, 1]))
tree = cKDTree(pts_ok)
pts_need = need[["x", "y"]].values.copy()
pts_need[:, 0] *= np.cos(np.deg2rad(pts_need[:, 1]))
dist, idx = tree.query(pts_need, k=20)
slope_vals = ok.f6_slope.values
fill = np.median(slope_vals[idx], axis=1)  # 邻域中位数（稳健）
fill_s = pd.Series(fill, index=need.index, name="slope")
# 邻域距离过大的（>1°）不可信，不填
fill_s[dist.max(axis=1) > 1.0] = np.nan
fill_s = fill_s.dropna()
fill_s.to_frame().to_parquet(BASE / "output" / "slope_fill.parquet")
print(f"填补成功 {len(fill_s)}（邻域距离≤1°），中位坡度 {fill_s.median():.2e}")

# 找含待救 reach 的分片并删除对应 part
rescue = set(fill_s.index)
todo_shards = []
for f in sorted(SHARDS.glob("shard_*.parquet")):
    ids = pd.read_parquet(f, columns=["reach_id"]).reach_id
    if ids.isin(rescue).any():
        todo_shards.append(f.stem)
n_del = 0
for stem in todo_shards:
    p = PARTS / f"part_{stem}.parquet"
    if p.exists():
        p.unlink()
        n_del += 1
print(f"删除 {n_del} 个 part 文件（含待救 reach 的分片），"
      f"重跑 build_features_v1.py 即可补算")
