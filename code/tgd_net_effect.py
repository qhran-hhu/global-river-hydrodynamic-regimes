# -*- coding: utf-8 -*-
"""A1 修补：tgd_fingerprint净效应重算（意见书问题 1）。

1. 逐年特征（稳健口径）+ 全期合并特征（复现旧口径 -21/-33/-76）
2. 净坝效应 = 各站变化 − 朱沱（上游对照）变化
3. 汉口站极端年排查：pre-dam 1976-1985 逐年 IQR(logH)，剔除最大年后重估
输出：output/human_activity/tgd_net_effect_recompute.csv + 逐年诊断表
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
DATA = BASE.parent / "yangtze_hydrodynamics_dataset" / "yangtze_daily_hydrodynamics.csv"
if not DATA.is_file():
    raise SystemExit(
        f"Yangtze gauge-derived file not found: {DATA}\n"
        "These records are restricted (Changjiang Water Resources Commission) "
        "and are not redistributable; see data/README.md."
    )
OUT = BASE / "output" / "human_activity"

DIST = {"宜昌站": 40, "监利站": 330, "汉口站": 500, "九江站": 750,
        "大通站": 1000, "朱沱站": -600}


def feats(g):
    z, q, h = g.Z_m.values, g.Q_m3s.values, g.H_Pa.values
    if len(z) < 60:
        return None
    pz = np.percentile(z, [10, 50, 90])
    pq = np.percentile(q, [10, 50, 90, 95])
    logh = np.log10(h[h > 0])
    return dict(var_Z=(pz[2] - pz[0]) / pz[1],
                var_Q=(pq[2] - pq[0]) / pq[1],
                ev_Q=(pq[3] - pq[1]) / pq[1],
                iqr_H=np.percentile(logh, 75) - np.percentile(logh, 25),
                Z_median=pz[1], Q_median=pq[1])


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df.drop(columns=["station"]).rename(columns={"station_zh": "station"})
    df["year"] = df.date.dt.year

    # ---- 1. 逐年特征 ----
    yearly = []
    for (st, per, yr), g in df.groupby(["station", "period", "year"]):
        f = feats(g)
        if f:
            f.update(station=st, period=per, year=yr, n=len(g))
            yearly.append(f)
    y = pd.DataFrame(yearly)
    y.to_csv(OUT / "tgd_net_effect_yearly.csv", index=False,
             encoding="utf-8-sig")

    # ---- 2. 两种口径的期别特征：全期合并 vs 逐年中位 ----
    rows = []
    for (st, per), g in df.groupby(["station", "period"]):
        pooled = feats(g)
        ym = y[(y.station == st) & (y.period == per)]
        med = ym[["var_Z", "var_Q", "ev_Q", "iqr_H"]].median()
        rows.append(dict(station=st, period=per,
                         **{f"{k}_pooled": pooled[k] for k in med.index},
                         **{f"{k}_ymed": med[k] for k in med.index}))
    d = pd.DataFrame(rows)

    # ---- 3. 变化与净效应 ----
    out = []
    for st in DIST:
        pre = d[(d.station == st) & (d.period == "pre-dam")]
        post = d[(d.station == st) & (d.period == "post-dam")]
        if pre.empty or post.empty:
            continue
        r = {"station": st, "dist_km": DIST[st]}
        for c in ["var_Z", "var_Q", "ev_Q", "iqr_H"]:
            for mode in ["pooled", "ymed"]:
                a = pre[f"{c}_{mode}"].iloc[0]
                b = post[f"{c}_{mode}"].iloc[0]
                r[f"{c}_{mode}_pct"] = 100 * (b - a) / abs(a)
        out.append(r)
    s = pd.DataFrame(out)
    # 净效应 = 减朱沱（上游对照）
    ctrl = s[s.station == "朱沱站"].iloc[0]
    for c in ["var_Z", "var_Q", "ev_Q", "iqr_H"]:
        for mode in ["pooled", "ymed"]:
            s[f"{c}_{mode}_net"] = s[f"{c}_{mode}_pct"] - ctrl[f"{c}_{mode}_pct"]
    s.to_csv(OUT / "tgd_net_effect_recompute.csv", index=False, encoding="utf-8-sig")

    # ---- 4. 汉口极端年排查 ----
    hk = y[(y.station == "汉口站") & (y.period == "pre-dam")].sort_values(
        "iqr_H", ascending=False)
    print("== 汉口 pre-dam 逐年 IQR(logH)（降序）==")
    print(hk[["year", "n", "iqr_H", "var_Z", "Z_median"]].to_string(index=False))
    top = hk.iloc[0]
    hk_ex = hk[hk.year != top.year]
    print(f"\n剔除最大年 {int(top.year)} 后: 逐年中位 iqr_H "
          f"{hk_ex.iqr_H.median():.3f} (原 {hk.iqr_H.median():.3f})")

    print("\n== 净效应汇总（ymed 口径，减朱沱对照）==")
    cols = ["station", "dist_km", "var_Z_ymed_pct", "var_Z_ymed_net",
            "iqr_H_ymed_pct", "iqr_H_ymed_net"]
    print(s[cols].round(1).to_string(index=False))
    print("\n== 对照：pooled 口径（旧结果复现）==")
    cols2 = ["station", "dist_km", "var_Z_pooled_pct", "iqr_H_pooled_pct",
             "iqr_H_pooled_net"]
    print(s[cols2].round(1).to_string(index=False))

    # 朱沱对照自身（数据稀少，需谨慎）
    zt = y[y.station == "朱沱站"]
    print("\n== 朱沱对照逐年（检验对照可靠性）==")
    print(zt[["period", "year", "n", "iqr_H", "var_Z"]].to_string(index=False))


if __name__ == "__main__":
    main()
