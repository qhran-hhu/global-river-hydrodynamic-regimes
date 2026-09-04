# -*- coding: utf-8 -*-
"""f5 修复：把掩膜重算的谐波列写回特征矩阵（备份旧列）。

特征矩阵_v1.parquet / 特征矩阵_v1_qc.parquet 的
f5_phase / f5_amp / f5_r2 ← output/regime_space/f5_掩膜重算.parquet
旧列保留为 f5_*_buggy（审计用）。
  python patch_f5_matrix.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
FIX = OUT / "regime_space" / "f5_掩膜重算.parquet"

fix = pd.read_parquet(FIX)
print(f"修复表: {fix.shape}, r2 非NaN {fix.f5_r2_m.notna().sum()}")

for name in ["特征矩阵_v1.parquet", "特征矩阵_v1_qc.parquet"]:
    d = pd.read_parquet(OUT / name)
    for c in ["f5_phase", "f5_amp", "f5_r2"]:
        d[c + "_buggy"] = d[c]
    d["f5_phase"] = fix.f5_phase_m.reindex(d.index)
    d["f5_amp"] = fix.f5_amp_m.reindex(d.index)
    d["f5_r2"] = fix.f5_r2_m.reindex(d.index)
    n_fix = d.f5_r2.notna().sum()
    old_gate = d.f5_r2_buggy.isna().mean() * 100
    new_weak = ((d.f5_r2 < 0.3) | d.f5_r2.isna()).mean() * 100
    d.to_parquet(OUT / name)
    print(f"{name}: n={len(d)}, 修复后 r2 非NaN {n_fix} "
          f"({n_fix/len(d)*100:.1f}%), 旧门控 {old_gate:.1f}%, "
          f"新弱年循环(R²<0.3) {new_weak:.1f}%", flush=True)
