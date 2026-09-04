# -*- coding: utf-8 -*-
"""WSE 相位 vs 流量相位对照（回应 v0 遗留问题：北半球 f5 峰值偏早）。

对 10 个北美测站，用同一谐波拟合（features.harmonic_fit，年积日基准）
分别计算 SWOT WSE 相位与 USGS 逐日流量相位，比较峰值月份差。
若 WSE 系统性领先 Q，则 v0 的"北半球偏早"是物理真实（水位先涨）而非缺陷。

输出：code/output/feature_matrix_v0/相位对照_WSE_vs_Q.csv/png
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

GS = BASE.parent / "阶段0_可行性验证" / "output" / "global_stations"
OUT = BASE / "output" / "feature_matrix_v0"


def phase_of(dates, values):
    ph, amp, r2 = harmonic_fit(pd.Series(dates), np.asarray(values, float))
    return ph, r2


def main():
    match = pd.read_csv(GS / "站点_reach匹配.csv")
    rows = []
    for _, r in match.iterrows():
        site = str(r.usgs_site).zfill(8)
        f_q = GS / f"usgs_{site}_daily.csv"
        f_w = GS / f"swot_{site}_timeseries.csv"
        if not f_q.exists() or not f_w.exists():
            continue
        q = pd.read_csv(f_q, parse_dates=["date"])
        w = pd.read_csv(f_w)
        w["date"] = pd.to_datetime(w.time_str, utc=True, errors="coerce")
        w = w.dropna(subset=["date"])
        w.loc[w.wse <= -1e11, "wse"] = np.nan
        if w.wse.notna().sum() < 15:
            print(f"{r.station}: SWOT WSE 有效观测不足，跳过")
            continue
        ph_q, r2_q = phase_of(q.date, q.Q_m3s)
        ph_w, r2_w = phase_of(w.dt if hasattr(w, "dt") else w.date, w.wse)
        m_q = ph_q / (2 * np.pi) * 12 + 1
        m_w = ph_w / (2 * np.pi) * 12 + 1
        dm = (m_w - m_q + 6) % 12 - 6  # 环形差，正=WSE晚于Q
        rows.append(dict(station=r.station, reach_id=r.reach_id,
                         peak_Q_month=round(m_q, 1), r2_Q=round(r2_q, 2),
                         peak_WSE_month=round(m_w, 1), r2_WSE=round(r2_w, 2),
                         WSE_minus_Q_month=round(dm, 1)))
        print(f"{r.station:32s} Q峰值 {m_q:4.1f}月(R²={r2_q:.2f})  "
              f"WSE峰值 {m_w:4.1f}月(R²={r2_w:.2f})  差 {dm:+.1f}月")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "相位对照_WSE_vs_Q.csv", index=False,
              encoding="utf-8-sig")
    good = df[(df.r2_Q >= 0.3) & (df.r2_WSE >= 0.3)]
    print(f"\n双 R²≥0.3 的 {len(good)} 站：WSE−Q 相位差中位 "
          f"{good.WSE_minus_Q_month.median():+.1f} 月")
    print(f"（负值=WSE 领先 Q；全部 10 站中位 "
          f"{df.WSE_minus_Q_month.median():+.1f} 月）")

    # 图
    fig, ax = plt.subplots(figsize=(8, 5))
    c = np.where((df.r2_Q >= 0.3) & (df.r2_WSE >= 0.3), "#c0504d", "gray")
    ax.scatter(df.peak_Q_month, df.peak_WSE_month, s=60, c=c, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r.station.split(" R ")[0],
                    (r.peak_Q_month, r.peak_WSE_month),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    lim = (0.5, 12.5)
    ax.plot(lim, lim, "k--", lw=0.8, label="WSE 与 Q 同步")
    ax.fill_between(lim, [lim[0] - 1, lim[1] - 1], lim, color="g",
                    alpha=0.08, label="WSE 领先（偏早 1 月内）")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("USGS 流量峰值月份")
    ax.set_ylabel("SWOT WSE 峰值月份")
    ax.set_title("WSE 相位 vs 流量相位（10 北美站；红=双 R²≥0.3）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "相位对照_WSE_vs_Q.png", bbox_inches="tight", dpi=150)
    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
