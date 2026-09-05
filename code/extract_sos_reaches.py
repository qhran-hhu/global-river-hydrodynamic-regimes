# -*- coding: utf-8 -*-
"""从六洲 SoS 文件提取全球 reach 清单 + 流量均值参考表（W1 交付物）。

输出（code/output/）：
  global_reach_list.csv  —— reach_id, continent, x, y, river_name, observations,
                       momma_width/slope/Y/v/Qmean, metroman_allq, consensus_q,
                       hivdi_q, sad_q, sic4dvar_q_mm,
                       QC flags（ice_clim_f, dark_frac, low_slope_flag,
                                wse/width/slope_outliers, has_validation）
用法：python extract_sos_reaches.py
"""
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path

SOS_DIR = Path("data/sos")
CONTINENTS = ["na", "sa", "eu", "af", "as", "oc"]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

FILL = -1e11


def read_var(group, name):
    """防御性读取：缺变量返回 None；vlen/ragged 取首元素；填充值→NaN。"""
    if name not in group.variables:
        return None
    raw = group.variables[name][:]
    if getattr(raw, "dtype", None) == object:
        def _to_float(x):
            a = np.atleast_1d(x)
            if not a.size:
                return np.nan
            try:
                return float(a[0])
            except (TypeError, ValueError):
                return np.nan
        arr = np.array([_to_float(x) for x in raw], dtype=float)
    else:
        arr = np.asarray(raw, dtype=float).ravel()
    return np.where(np.isfinite(arr) & (np.abs(arr) < abs(FILL)), arr, np.nan)


def read_names(var):
    """river_name 为 vlen 字符串数组。"""
    raw = var[:]
    out = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", "ignore"))
        else:
            out.append(str(x))
    return out


def extract_continent(cc):
    f = SOS_DIR / f"{cc}_sos_v3.nc"
    with Dataset(f) as ds:
        reaches = ds.groups["reaches"]
        rid = np.asarray(reaches.variables["reach_id"][:], dtype="int64")
        n = len(rid)
        d = dict(reach_id=rid,
                 x=read_var(reaches, "x"), y=read_var(reaches, "y"),
                 observations=read_var(reaches, "observations"))
        d["river_name"] = read_names(reaches.variables["river_name"])

        momma = ds.groups["momma"]
        for src, dst in [("width", "momma_width"), ("slope", "momma_slope"),
                         ("Y", "momma_depth"), ("v", "momma_v"),
                         ("Qmean_momma", "momma_q")]:
            d[dst] = read_var(momma, src)
        d["metroman_q"] = read_var(ds.groups["metroman"], "allq")
        d["consensus_q"] = read_var(ds.groups["consensus"], "consensus_q")
        d["hivdi_q"] = read_var(ds.groups["hivdi"], "Q")
        d["sad_q"] = read_var(ds.groups["sad"], "Qa")
        d["sic4dvar_q"] = read_var(ds.groups["sic4dvar"], "Q_mm")

        pre = ds.groups["prediagnostics"].groups["reach"]
        for v in ["ice_clim_f", "dark_frac", "low_slope_flag",
                  "wse_outliers", "width_outliers", "slope_outliers"]:
            d[v] = read_var(pre, v)
        flpe = ds.groups["validation"].groups["flpe"]
        hv = read_var(flpe, "has_validation")
        d["has_validation"] = (hv == 1).astype(float) if hv is not None else None

    df = pd.DataFrame(d)
    df.insert(1, "continent", cc)
    return df


def main():
    frames = []
    for cc in CONTINENTS:
        df = extract_continent(cc)
        frames.append(df)
        gauged = int(np.nansum(df.has_validation))
        print(f"{cc}: {len(df)} reaches, 有官方验证 {gauged}, "
              f"consensus_q 有效 {df.consensus_q.notna().sum()}")
    allr = pd.concat(frames, ignore_index=True)
    p = OUT / "global_reach_list.csv"
    allr.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n合计 {len(allr)} reaches -> {p}")
    print(f"有官方验证 reach 总数: {int(np.nansum(allr.has_validation))}")
    print(f"consensus_q 有效: {allr.consensus_q.notna().sum()}")


if __name__ == "__main__":
    main()
