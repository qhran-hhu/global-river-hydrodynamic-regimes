# -*- coding: utf-8 -*-
"""三峡指纹实验：TGD 在抗噪特征空间造成的位移（应对①）。

数据：长江六站前后两时段逐日 Q/Z/A/H（yangtze_daily_hydrodynamics.csv）。
特征（与全球 f2–f7 同构，用实测 Z 代替 SWOT WSE）：
  var_Z   = (P90-P10)/P50 of 水位 Z      —— 对应 f2
  iqr_H   = IQR(log10 H)                 —— 对应 f4
  ev_Q    = (P95-P50)/P50 of Q           —— 事件响应（对应 f3 的流量版）
  var_Q   = (P90-P10)/P50 of Q           —— 流量变幅（机制解释用）
  phase   = Z 年循环谐波相位（峰值月份）   —— 对应 f5
站点按距坝里程排序：宜昌(约40km) < 监利 < 汉口 < 九江 < 大通；朱沱在上游库尾（对照）。

判据：宜昌（坝下最近站）出现方向一致、量级显著的位移，
且位移随距坝里程衰减 → 特征能"看见"大坝 → 应对①通过。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from features import harmonic_fit

try:
    from plotstyle import setup_plot
    setup_plot()
except Exception:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

DATA = BASE.parent / "yangtze_hydrodynamics_dataset" / "yangtze_daily_hydrodynamics.csv"
if not DATA.is_file():
    raise SystemExit(
        f"Yangtze gauge-derived file not found: {DATA}\n"
        "These records are restricted (Changjiang Water Resources Commission) "
        "and are not redistributable; see data/README.md."
    )
OUT = BASE / "output" / "human_activity"
OUT.mkdir(exist_ok=True)

# 距三峡大坝里程（km，约值；朱沱为上游库尾对照）
DIST = {"宜昌站": 40, "监利站": 330, "汉口站": 500, "九江站": 750,
        "大通站": 1000, "朱沱站": -600}


def feats(g):
    z, q, h = g.Z_m.values, g.Q_m3s.values, g.H_Pa.values
    pz = np.percentile(z, [10, 50, 90])
    pq = np.percentile(q, [10, 50, 90, 95])
    logh = np.log10(h[h > 0])
    ph, _, r2 = harmonic_fit(pd.Series(g.date), z.astype(float))
    return dict(var_Z=(pz[2] - pz[0]) / pz[1],
                var_Q=(pq[2] - pq[0]) / pq[1],
                ev_Q=(pq[3] - pq[1]) / pq[1],
                iqr_H=np.percentile(logh, 75) - np.percentile(logh, 25),
                phase_month=ph / (2 * np.pi) * 12 + 1 if np.isfinite(ph)
                else np.nan,
                phase_r2=r2,
                Z_median=pz[1], Q_median=pq[1])


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df.drop(columns=["station"]).rename(columns={"station_zh": "station"})
    rows = []
    for (st, per), g in df.groupby(["station", "period"]):
        f = feats(g)
        f.update(station=st, period=per, dist=DIST[st])
        rows.append(f)
    d = pd.DataFrame(rows)
    piv = {}
    feat_cols = ["var_Z", "var_Q", "ev_Q", "iqr_H", "phase_month",
                 "Z_median", "Q_median"]
    for c in feat_cols:
        t = d.pivot(index="station", columns="period", values=c)
        t["delta"] = t["post-dam"] - t["pre-dam"]
        t["delta_pct"] = 100 * t["delta"] / t["pre-dam"].abs()
        piv[c] = t
    # 相位是有单位的月份，不用相对百分比（会产生歧义），其余特征用 %
    summary = pd.DataFrame({c: (piv[c]["delta"] if c == "phase_month"
                                else piv[c]["delta_pct"]) for c in feat_cols})
    summary = summary.rename(columns={"phase_month": "phase_shift_months"})
    r2p = d.pivot(index="station", columns="period", values="phase_r2")
    summary["phase_r2_pre"] = r2p["pre-dam"]
    summary["phase_r2_post"] = r2p["post-dam"]
    summary["dist_km"] = [DIST[s] for s in summary.index]
    summary = summary.sort_values("dist_km")
    print("=== 建坝前后特征变化（%，post-pre 相对 pre）===")
    print(summary.round(1).to_string())
    summary.to_csv(OUT / "三峡指纹_特征位移.csv", encoding="utf-8-sig")

    # 相位变化（绝对月份）
    ph = d.pivot(index="station", columns="period", values="phase_month")
    ph["dist_km"] = [DIST[s] for s in ph.index]
    print("\n=== 水位峰值月份（建坝前 → 后）===")
    print(ph.sort_values("dist_km").round(2).to_string())

    # 图：位移 vs 距坝里程
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for c, lab, col in [("var_Z", "水位变幅 (P90-P10)/P50", "#c0504d"),
                        ("var_Q", "流量变幅 (P90-P10)/P50", "#4bacc6"),
                        ("ev_Q", "事件响应 (P95-P50)/P50", "#9bbb59"),
                        ("iqr_H", "IQR(log10H)", "#8064a2")]:
        s = summary.sort_values("dist_km")
        ax.plot(s.dist_km, s[c], "o-", label=lab, color=col, lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.annotate("三峡大坝", (0, ax.get_ylim()[1]), fontsize=8,
                ha="center", va="bottom")
    ax.set_xlabel("距三峡大坝里程（km；负=上游库尾对照）")
    ax.set_ylabel("建坝后变化（%）")
    ax.set_title("三峡指纹：特征位移随距坝里程的衰减")
    ax.legend(fontsize=8)

    ax = axes[1]
    s = ph.sort_values("dist_km").dropna(subset=["pre-dam", "post-dam"])
    x = s.dist_km.values
    ax.plot(x, s["pre-dam"], "o-", color="#4bacc6", label="建坝前")
    ax.plot(x, s["post-dam"], "s-", color="#c0504d", label="建坝后")
    for xi, a, b in zip(x, s["pre-dam"], s["post-dam"]):
        if np.isfinite(a) and np.isfinite(b) and abs(b - a) > 1e-6:
            ax.annotate("", xy=(xi, b), xytext=(xi, a),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.set_xlabel("距三峡大坝里程（km）")
    ax.set_ylabel("水位峰值月份")
    ax.set_title("季节相位移动")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "三峡指纹.png", bbox_inches="tight", dpi=150)
    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
