# -*- coding: utf-8 -*-
"""f2 高程混淆修正：深度相对变幅重算 + 矩阵补丁

旧定义 f2=(P90-P10)/P50(WSE)，分母为大地高程 → 与海拔 ρ=-0.79（§8.30）。
新定义 f2'=(P90-P10)/mean(h_rel)，h_rel = WSE - P5(WSE)（clip 1e-3）。
旧列备份为 f2_rel_range_geo。

用法：python f2_recompute_depth.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SHARDS = BASE / "output" / "ts_shards_global"
OUT = BASE / "output" / "regime_space"
MIN_OBS = 15


def main():
    files = sorted(SHARDS.glob("shard_*.parquet"))
    recs = {}
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["reach_id", "time_str", "wse"])
        df = df[(df.time_str != "no_data") & df.time_str.notna()]
        df["wse"] = pd.to_numeric(df.wse, errors="coerce")
        df.loc[df.wse <= -1e11, "wse"] = np.nan
        df = df.dropna(subset=["wse"])
        g = df.groupby("reach_id").wse
        q = g.quantile([0.05, 0.1, 0.9]).unstack()
        n = g.size()
        p5 = q[0.05]
        mean_hrel = (g.mean() - p5).clip(lower=1e-3)
        f2d = (q[0.9] - q[0.1]) / mean_hrel
        f2d[n < MIN_OBS] = np.nan
        for rid, v in f2d.items():
            recs[rid] = (float(v) if np.isfinite(v) else np.nan,
                         float(mean_hrel.get(rid, np.nan)))
        if (i + 1) % 100 == 0:
            print(f"  shard {i + 1}/{len(files)}", flush=True)
    R = pd.DataFrame.from_dict(recs, orient="index",
                               columns=["f2_depth", "mean_h_rel"])
    R.index.name = "reach_id"
    R.to_parquet(OUT / "f2_depth_relative_recompute.parquet")
    v = R.f2_depth.dropna()
    print(f"重算 {len(R)} reach，有效 {v.notna().sum()}")
    print(f"f2' 分布: 中位 {v.median():.3f}, IQR [{v.quantile(.25):.3f}, "
          f"{v.quantile(.75):.3f}], P95 {v.quantile(.95):.3f}")

    # 补丁两个矩阵
    for name in ["feature_matrix_v1.parquet", "feature_matrix_v1_qc.parquet"]:
        mp = BASE / "output" / name
        m = pd.read_parquet(mp)
        if "reach_id" in m.columns:
            m = m.set_index("reach_id")
        if "f2_rel_range_geo" not in m.columns:
            m["f2_rel_range_geo"] = m["f2_rel_range"]  # 备份旧定义
        m = m.drop(columns=["f2_rel_range"]).join(R[["f2_depth"]])
        m = m.rename(columns={"f2_depth": "f2_rel_range"})
        m.to_parquet(mp)
        print(f"patched {name}: f2' 非NaN {m.f2_rel_range.notna().sum()}"
              f"/{len(m)}")


if __name__ == "__main__":
    main()
