# -*- coding: utf-8 -*-
"""阶段 1 抗噪特征套件 f1–f7（定稿自阶段 0 验证③，见工作纪要 §6）。

设计原则（阶段 0 硬约束）：
  - 不写绝对 H 值：一切水平量用跨站秩次/分位数表达（验证①）
  - 禁用 CV 等绝对方差型特征（验证③：朴素特征 ARI 显著低于抗噪特征）
  - "丰水月"已否决（单河无相位方差），改用 WSE 年循环谐波相位（f5）

特征定义（方案 §4）：
  f1 水平秩次    log10(中位 H_proxy) 的全球分位数秩 —— 秩次对单调 bias 免疫（r=1.000）
  f2 相对变幅    (P90−P10)/P50 of WSE
  f3 事件响应    (P95−P50)/P50 of width
  f4 对数离散    IQR(log10 H_proxy)
  f5 谐波相位    WSE 年循环一次谐波相位（对 21 天稀疏采样稳健）
  f6 坡度先验    SWORD/momma 坡度（静态先验，无观测误差）
  f7 宽-坡耦合   宽度 IQR 型变异 × f6

H_proxy 双路径（初期并行，终检为全球尺度双路径 ARI）：
  路径甲（纯观测）：Manning 型代理  u² ∝ h_rel^(4/3)·S，
                    h_rel = WSE − P5(WSE)（相对水深，单位一致即可，常数不影响秩次）
  路径乙（流量参考）：u = Q_sos /(W_momma·D_momma)，H = ρu²，
                    经河宽分层校正（窄河 +21% → 宽河 −3%，验证②）

QC（在特征计算前执行，方案 §5 的观测级部分）：
  - 剔除填充值（≤ −1e11）与 time_str == "no_data"
  - 有效观测 < MIN_OBS（默认 15）的 reach 不参与特征计算
"""
import numpy as np
import pandas as pd

RHO = 1000.0
MIN_OBS = 15
FILL = -1e11  # 填充值阈值


# ---------- 观测级清洗 ----------

def clean_series(df):
    """剔除填充值与 no_data 行，解析时间。输入须含 time_str,wse,width,slope。"""
    df = df.copy()
    df = df[df.time_str.notna() & (df.time_str != "no_data")]
    df["date"] = pd.to_datetime(df.time_str, utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ("wse", "width", "slope"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= FILL, c] = np.nan
    return df.sort_values("date")


# ---------- H_proxy 双路径 ----------

def hproxy_path_a(df, slope_prior=None):
    """路径甲（纯观测）：Manning 型 H 代理。
    u² ∝ R^(4/3)·S（Manning，n 为未知常数，不影响单 reach 内统计形态；
    跨 reach 秩次的有效性由双路径 ARI 终检验证）。
    返回与 df 对齐的 H_proxy 序列（无法计算处为 NaN）。"""
    wse = df.wse.values.astype(float)
    slope = np.where(np.isfinite(df.slope.values.astype(float))
                     & (df.slope.values.astype(float) > 0),
                     df.slope.values.astype(float), np.nan)
    if slope_prior is not None and np.isfinite(slope_prior) and slope_prior > 0:
        slope = np.where(np.isfinite(slope), slope, slope_prior)
    p5 = np.nanpercentile(wse, 5) if np.isfinite(wse).sum() >= 5 else np.nan
    h_rel = np.clip(wse - p5, 1e-3, None)
    return RHO * h_rel ** (4.0 / 3.0) * slope


def width_bias_correction(width_m):
    """河宽分层的 Q bias 校正因子（验证②）：
    窄河（≤100 m）官方验证 nbias ≈ +21% → Q 除以 1.21；
    宽河（≥1000 m）≈ −3% → 除以 0.97；
    中间按 log10(width) 线性插值，界外取端点值。
    返回乘在 Q 上的校正系数（即 1/(1+nbias)）。"""
    lw = np.log10(np.clip(width_m, 30, 3000))
    lo, hi = np.log10(100.0), np.log10(1000.0)
    nbias = 0.21 + (lw - lo) / (hi - lo) * (-0.03 - 0.21)
    nbias = np.clip(nbias, -0.03, 0.21)
    return 1.0 / (1.0 + nbias)


def hproxy_path_b(q_mean, width_m, depth_m):
    """路径乙（流量参考）：SoS 平均 Q 经河宽分层校正 → u → H。
    q_mean, width_m, depth_m 可为标量或数组。返回 H_proxy（标量/数组）。"""
    q = np.asarray(q_mean, dtype=float) * width_bias_correction(width_m)
    area = np.asarray(width_m, dtype=float) * np.asarray(depth_m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = q / area
    return RHO * u ** 2


# ---------- 单 reach 特征 f2–f5, f7 ----------

def annual_harmonic_phase(dates, values):
    """WSE 年循环一次谐波相位（弧度，[0, 2π)，0 = 1 月 1 日峰值）。
    对稀疏/不规则采样稳健：直接对观测点做最小二乘拟合 cos/sin。
    注意：t 必须用绝对年积日（day-of-year），不能用相对序列起点的时间，
    否则相位基准随各 reach 首观测日期漂移（v0 曾因此得到错误的南北半球结论）。"""
    phase, _, _ = harmonic_fit(dates, values)
    return phase


def harmonic_fit(dates, values):
    """年循环一次谐波最小二乘拟合，返回 (相位, 振幅, R²)。
    R² 低说明年循环信号弱，相位不可信（v0 发现北半球相位散布的主因），
    用于特征层的 f5 质量屏蔽。"""
    mask = np.isfinite(values)
    if mask.sum() < MIN_OBS:
        return np.nan, np.nan, np.nan
    t = dates[mask].dt.dayofyear.values / 365.25
    y = values[mask]
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


def reach_features(df, slope_prior=None, path="A"):
    """单 reach 的 f2–f5, f7（f1/f6 需跨 reach 或外部先验，在矩阵层合并）。
    返回 dict；观测不足时返回 None。"""
    df = clean_series(df)
    n = len(df)
    if n < MIN_OBS:
        return None
    wse, width = df.wse.values, df.width.values
    h = hproxy_path_a(df, slope_prior) if path == "A" else None

    def ratio(v, lo, hi, base=50):
        v = v[np.isfinite(v)]
        if len(v) < MIN_OBS:
            return np.nan
        p = np.percentile(v, [lo, base, hi])
        return (p[2] - p[0]) / p[1] if p[1] > 0 else np.nan

    f2 = ratio(wse, 10, 90)                    # 相对变幅 of WSE
    f3 = ratio(width, 50, 95)                  # 事件响应 of width
    logh = np.log10(h[np.isfinite(h)]) if h is not None else np.array([])
    f4 = (np.percentile(logh, 75) - np.percentile(logh, 25)
          if len(logh) >= MIN_OBS else np.nan)  # IQR(log H)
    f5, f5_amp, f5_r2 = harmonic_fit(df.date, wse)  # 谐波相位+质量
    med_h = float(np.nanmedian(h)) if h is not None and len(logh) >= MIN_OBS \
        else np.nan
    return dict(n_obs=n, med_hproxy=med_h, f2_rel_range=f2,
                f3_event_resp=f3, f4_iqr_logh=f4, f5_phase=f5,
                f5_amp=f5_amp, f5_r2=f5_r2)


# ---------- 特征矩阵（跨 reach 合并 f1/f6/f7） ----------

def build_feature_matrix(ts_df, reach_table, slope_col="slope", path="A"):
    """ts_df: Hydrocron 拉取的时间序列（reach_id,time_str,wse,width,slope）。
    reach_table: 每 reach 一行，含 reach_id 与坡度先验列（默认 slope，
                 未来接 SoS reaches/momma 组）。
    返回每 reach 一行的特征表（f1–f7），f1 为 log10(中位 H_proxy) 的分位数秩。"""
    slope_map = reach_table.set_index("reach_id")[slope_col].to_dict() \
        if slope_col in reach_table.columns else {}
    rows = []
    for rid, g in ts_df.groupby("reach_id"):
        ft = reach_features(g, slope_prior=slope_map.get(rid), path=path)
        if ft is None:
            continue
        ft["reach_id"] = rid
        ft["f6_slope"] = slope_map.get(rid, np.nan)
        rows.append(ft)
    out = pd.DataFrame(rows).set_index("reach_id")
    if len(out) == 0:
        return out
    # f1：水平秩次（分位数秩，0–1）
    out["f1_level_rank"] = np.log10(out.med_hproxy).rank(pct=True)
    # f7：宽度 IQR 型变异 × 坡度
    wiqr = []
    for rid in out.index:
        g = clean_series(ts_df[ts_df.reach_id == rid])
        w = g.width.values
        w = w[np.isfinite(w)]
        p50 = np.percentile(w, 50) if len(w) >= MIN_OBS else np.nan
        iqr = (np.percentile(w, 75) - np.percentile(w, 25)) \
            if len(w) >= MIN_OBS else np.nan
        wiqr.append(iqr / p50 if p50 and p50 > 0 else np.nan)
    out["f7_width_slope"] = np.asarray(wiqr) * out.f6_slope
    cols = ["n_obs", "med_hproxy", "f1_level_rank", "f2_rel_range",
            "f3_event_resp", "f4_iqr_logh", "f5_phase", "f5_amp", "f5_r2",
            "f6_slope", "f7_width_slope"]
    return out[cols]


if __name__ == "__main__":
    # 冒烟测试：合成数据验证代码路径（真实 smoke test 见 test_shards 运行）
    rng = np.random.default_rng(0)
    frames = []
    for i, rid in enumerate([111, 222, 333]):
        n = 80
        t = pd.date_range("2023-01-01", periods=n, freq="14D", tz="UTC")
        phase = i * 2.0
        wse = 100 + 10 * i + 5 * np.cos(2 * np.pi * np.arange(n) / 26 + phase) \
            + rng.normal(0, 0.3, n)
        width = 300 + 100 * i + 30 * np.cos(2 * np.pi * np.arange(n) / 26
                                            + phase) + rng.normal(0, 5, n)
        frames.append(pd.DataFrame(dict(
            reach_id=rid, time_str=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            wse=wse, width=width, slope=1e-4 * (i + 1))))
    ts = pd.concat(frames, ignore_index=True)
    rt = pd.DataFrame(dict(reach_id=[111, 222, 333], slope=[1e-4, 2e-4, 3e-4]))
    fm = build_feature_matrix(ts, rt, path="A")
    print(fm.round(4).to_string())
    assert fm.shape == (3, 11) and fm.f1_level_rank.is_monotonic_increasing
    print("\n冒烟测试通过：特征矩阵形状与秩次单调性符合预期")
