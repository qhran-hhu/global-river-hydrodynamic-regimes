# -*- coding: utf-8 -*-
"""汇总 features_v1_parts → 特征矩阵 v1（含 f1 秩次、f7 耦合、元数据、QC 标记）。

输出：output/feature_matrix_v1.parquet + QC 后 _qc.parquet
QC 规则（方案 §5）：ice_clim_f > 0.2 或 dark_frac 高（>0.5）剔除；
-999 填充视为未知，不剔除；f1–f7 关键特征缺失剔除（在 cluster 端处理）。
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
PARTS = BASE / "output" / "features_v1_parts"
META_CSV = BASE / "output" / "global_reach_list.csv"
OUT = BASE / "output"

fm = pd.concat([pd.read_parquet(f) for f in sorted(PARTS.glob("part_*.parquet"))],
               ignore_index=True).set_index("reach_id")
print(f"原始特征行：{len(fm)}")

# f1：全球分位数秩
fm["f1_level_rank"] = np.log10(fm.med_hproxy.where(fm.med_hproxy > 0)).rank(pct=True)
# f7：宽度 IQR 比 × 坡度
fm["f7_width_slope"] = fm.width_iqr_ratio * fm.f6_slope

meta = pd.read_csv(META_CSV, usecols=["reach_id", "continent", "x", "y",
                                      "river_name", "momma_width",
                                      "momma_depth", "consensus_q",
                                      "momma_q", "metroman_q",
                                      "ice_clim_f", "dark_frac",
                                      "has_validation"]).set_index("reach_id")
fm = fm.join(meta)
fm.to_parquet(OUT / "feature_matrix_v1.parquet")
print(f"合并元数据后：{fm.shape}")

# QC
def bad(flag, thr):
    return flag.where(flag > -900).fillna(0) > thr  # -999 → 未知 → 保留

qc = fm[~bad(fm.ice_clim_f, 0.2) & ~bad(fm.dark_frac, 0.5)]
qc = qc[qc.f1_level_rank.notna() & qc.f2_rel_range.notna()]
qc.to_parquet(OUT / "feature_matrix_v1_qc.parquet")
print(f"QC 后：{len(qc)}（剔除 {len(fm)-len(qc)}：冰雪/暗像素/特征缺失）")
print("\n分洲分布：")
print(qc.continent.value_counts().to_string())
print("\nf5 R²≥0.3 占比: %.0f%%" % (100 * (qc.f5_r2 >= 0.3).mean()))
print("路径乙原料（consensus_q>0）覆盖: %.0f%%"
      % (100 * (qc.consensus_q > 0).mean()))
