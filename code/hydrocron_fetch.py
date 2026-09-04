# -*- coding: utf-8 -*-
"""Hydrocron 全球 reach 时间序列并发拉取器（断点续传）。

端点：https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries
实测：单次约 3.2 s，多 pass 聚合后约 104 次观测/3 年（优于单 pass 提取的 ~50 次）。
注意：Hydrocron 只支持几何字段（reach_id,time_str,wse,width,slope），
      dschg_* 一律 400；时间格式必须 YYYY-MM-DDTHH:MM:SSZ。

用法：
    # W1 速率限制试拉（1,000 reach）
    python hydrocron_fetch.py --reaches 全球reach清单.csv --limit 1000
    # 全球全量（断点续传，中断后再次运行自动跳过已完成 reach）
    python hydrocron_fetch.py --reaches 全球reach清单.csv --workers 10

输入：reach 清单 CSV，至少一列 reach_id。
输出：output/ts_shards/shard_00000.parquet ...（每 shard 2,000 个 reach）
      checkpoint 即"已出现在 shard 中的 reach_id"，重启自动跳过。
"""
import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

API = ("https://soto.podaac.earthdatacloud.nasa.gov/"
       "hydrocron/v1/timeseries")
FIELDS = "reach_id,time_str,wse,width,slope"
START, END = "2023-01-01T00:00:00Z", "2026-09-01T00:00:00Z"
SHARD_SIZE = 2000
RETRY_MAX = 3


def fetch_one(reach_id, session=None):
    """拉取单个 reach 的时间序列，返回 DataFrame（空表表示无数据）。"""
    params = dict(feature="Reach", feature_id=str(reach_id),
                  start_time=START, end_time=END,
                  output="csv", fields=FIELDS)
    http = session or requests
    for attempt in range(RETRY_MAX):
        try:
            r = http.get(API, params=params, timeout=60)
            if r.status_code in (429, 502, 503, 504):  # 限流/网关抖动：退避重试
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                # 400 = "Feature ID not found"（永久性，Hydrocron 库无此 reach）
                return None, f"HTTP {r.status_code}", r.status_code == 400
            js = r.json()
            csv_text = js.get("results", {}).get("csv", "")
            if not csv_text.strip():
                return pd.DataFrame(), None, False  # 无数据
            df = pd.read_csv(io.StringIO(csv_text))
            keep = [c for c in ("reach_id", "time_str", "wse", "width",
                                "slope") if c in df.columns]
            return df[keep], None, False
        except requests.RequestException as ex:
            time.sleep(3 * (attempt + 1))
            err = str(ex)
    return None, err if "err" in dir() else "retry exhausted", False


def load_done_ids(shard_dir):
    done = set()
    for f in sorted(shard_dir.glob("shard_*.parquet")):
        try:
            done.update(pd.read_parquet(f, columns=["reach_id"])
                        .reach_id.unique().tolist())
        except Exception:
            pass
    return done


def load_failed_ids(shard_dir):
    f = shard_dir / "failed_reaches.csv"
    if f.exists():
        try:
            return set(pd.read_csv(f, dtype={"reach_id": "int64"}).reach_id)
        except Exception:
            pass
    return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaches", required=True, type=Path,
                    help="reach 清单 CSV（含 reach_id 列）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只拉前 N 个未完成 reach（0=全部；试拉用 1000）")
    ap.add_argument("--workers", type=int, default=8, help="并发数（默认 8）")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "output" / "ts_shards")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reaches = (pd.read_csv(args.reaches, dtype={"reach_id": "int64"})
               .reach_id.unique().tolist())
    done = load_done_ids(args.out)
    failed = load_failed_ids(args.out)
    todo = [r for r in reaches if r not in done and r not in failed]
    if args.limit:
        todo = todo[:args.limit]
    print(f"清单 {len(reaches)}，已完成 {len(done)}，已知无效 {len(failed)}，"
          f"本次待拉 {len(todo)}，并发 {args.workers}")
    if not todo:
        print("无待拉 reach，退出")
        return

    buf, t0, n_ok, n_err = [], time.time(), 0, 0
    n_since_flush = 0
    FLUSH_EVERY = 200  # 每完成 200 个 reach 落盘一次（防超时丢失）
    shard_idx = len(list(args.out.glob("shard_*.parquet")))

    def flush():
        nonlocal buf, shard_idx, n_since_flush
        if not buf:
            return
        df = pd.concat(buf, ignore_index=True)
        p = args.out / f"shard_{shard_idx:05d}.parquet"
        df.to_parquet(p, index=False)
        print(f"  >> 写入 {p.name}（{df.reach_id.nunique()} reach, "
              f"{len(df)} 行）")
        shard_idx += 1
        buf = []
        n_since_flush = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, rid): rid for rid in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rid = futs[fut]
            df, err, permanent = fut.result()
            if df is not None:
                if len(df):
                    buf.append(df)
                n_ok += 1
            else:
                n_err += 1
                if permanent:  # HTTP 400：记录为已知无效，重启不再重试
                    with open(args.out / "failed_reaches.csv", "a") as ff:
                        ff.write(f"{rid}\n")
                else:
                    print(f"  !! reach {rid} 失败: {err}")
            n_since_flush += 1
            if i % 200 == 0 or i == len(todo):
                dt = time.time() - t0
                rate = i / dt
                eta = (len(todo) - i) / rate / 3600 if rate else 0
                print(f"  进度 {i}/{len(todo)}（成功 {n_ok} 失败 {n_err}）"
                      f" {rate:.2f} reach/s，ETA {eta:.1f} h", flush=True)
            if n_since_flush >= FLUSH_EVERY:
                flush()
    flush()
    dt = time.time() - t0
    print(f"完成：成功 {n_ok}，失败 {n_err}，用时 {dt/60:.1f} min，"
          f"平均 {dt/max(len(todo),1):.2f} s/reach")


if __name__ == "__main__":
    main()
