# -*- coding: utf-8 -*-
"""SoS L4 分洲断点续传下载器（阶段 1 通用化版本，替代阶段 0 的 download_sos.py）。

用法（凭据只走环境变量，绝不写入文件）：
    set EARTHDATA_USER=...  &  set EARTHDATA_PASSWORD=...
    python sos_download.py --resolve        # 只解析各洲文件名与大小（无需凭据）
    python sos_download.py sa               # 下载南美（可反复调用，断点续传）
    python sos_download.py sa as --budget 280
    python sos_download.py all              # 依次下载所有未完成的洲

文件名通过 CMR 实时解析，避免硬编码 URL/时间戳漂移。
目标目录默认 E:\\_swot_tmp（netCDF4 打不开中文路径），可用 --dest 改。
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME = "SWOT_L4_HR_DAWG_SOS_DISCHARGE_V3"
CONTINENTS = ["na", "sa", "eu", "af", "as", "oc"]
DEFAULT_DEST = Path("data/sos")
MIB = 1.048576e6  # CMR granule_size 单位是 MiB


class Session(requests.Session):
    """保留跨域 redirect 的 Authorization 到 urs.earthdata.nasa.gov。"""

    def __init__(self, user, password):
        super().__init__()
        self.auth = HTTPBasicAuth(user, password)

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if "Authorization" in headers:
            orig = requests.utils.urlparse(response.request.url).hostname
            redir = requests.utils.urlparse(url).hostname
            if orig != redir and redir != "urs.earthdata.nasa.gov" \
                    and orig != "urs.earthdata.nasa.gov":
                del headers["Authorization"]


def resolve_granule(continent):
    """CMR 实时解析某洲 SoS 颗粒的 (URL, 大小字节, 文件名)。"""
    params = [("short_name", SHORT_NAME),
              ("granule_ur", f"{continent}_sword_v16_SOS_results_unconstrained_*"),
              ("options[granule_ur][pattern]", "true"),
              ("page_size", "5")]
    r = requests.get(CMR, params=params, timeout=60)
    r.raise_for_status()
    entries = r.json()["feed"]["entry"]
    if not entries:
        raise RuntimeError(f"CMR 未找到 {continent} 洲 SoS 颗粒")
    e = entries[0]
    url = next(l["href"] for l in e["links"]
               if l.get("rel", "").endswith("/data#"))
    size = int(float(e["granule_size"]) * MIB)
    return url, size, e["title"]


def cmd_resolve():
    print(f"{'洲':<4}{'大小(GB)':>10}  文件名")
    total = 0
    for cc in CONTINENTS:
        try:
            url, size, title = resolve_granule(cc)
            total += size
            print(f"{cc:<6}{size/1e9:>8.1f}  {title}")
        except Exception as ex:
            print(f"{cc:<6}   解析失败: {ex}")
    print(f"{'合计':<6}{total/1e9:>8.1f} GB")


def download(continent, dest_dir, budget):
    url, total, title = resolve_granule(continent)
    dest = dest_dir / f"{continent}_sos_v3.nc"
    if dest.exists() and dest.stat().st_size < 1000:
        dest.unlink()  # 清除错误内容
    have = dest.stat().st_size if dest.exists() else 0
    print(f"[{continent}] {title}")
    print(f"[{continent}] 已有 {have/1e9:.2f} / {total/1e9:.2f} GB")
    if have >= total:
        print(f"[{continent}] 已完成，跳过")
        return True
    user = os.environ.get("EARTHDATA_USER")
    pw = os.environ.get("EARTHDATA_PASSWORD")
    if not user or not pw:
        print("缺少 EARTHDATA_USER / EARTHDATA_PASSWORD 环境变量")
        return False
    s = Session(user, pw)
    t0 = time.time()
    headers = {"Range": f"bytes={have}-"} if have else {}
    with s.get(url, headers=headers, stream=True, timeout=120) as r:
        print(f"[{continent}] status: {r.status_code}")
        if r.status_code not in (200, 206):
            print(r.text[:200])
            return False
        mode = "ab" if have and r.status_code == 206 else "wb"
        n = 0
        with open(dest, mode) as f:
            for chunk in r.iter_content(1024 * 1024 * 8):
                f.write(chunk)
                n += len(chunk)
                if time.time() - t0 > budget:
                    break
    now = dest.stat().st_size
    print(f"[{continent}] 本次写入 {n/1e9:.2f} GB，累计 {now/1e9:.2f} GB "
          f"({now/total*100:.0f}%)")
    return now >= total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("continents", nargs="*",
                    help="洲代码（na/sa/eu/af/as/oc）或 all")
    ap.add_argument("--resolve", action="store_true",
                    help="只解析文件名与大小，不下载")
    ap.add_argument("--budget", type=int, default=260,
                    help="单洲单次运行秒数预算（默认 260）")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help="目标目录（默认 E:\\_swot_tmp）")
    args = ap.parse_args()
    if args.resolve:
        cmd_resolve()
        return
    if not args.continents:
        ap.error("请指定洲代码或 all，或用 --resolve")
    args.dest.mkdir(parents=True, exist_ok=True)
    targets = CONTINENTS if args.continents == ["all"] else args.continents
    for cc in targets:
        if cc not in CONTINENTS:
            print(f"未知洲代码 {cc}，可选 {CONTINENTS}")
            continue
        ok = download(cc, args.dest, args.budget)
        if not ok:
            print(f"[{cc}] 未完成（可再次运行续传）")


if __name__ == "__main__":
    main()
