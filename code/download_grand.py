# -*- coding: utf-8 -*-
"""从 NASA Earthdata GIS 服务下载 GRanD v1.1 全球大坝点位（分页 JSON）。
输出：code/output/human_activity/grand_dams.csv
"""
import time
from pathlib import Path

import pandas as pd
import requests

API = ("https://gis.earthdata.nasa.gov/maps/rest/services/"
       "grand-v1-rev01/GRanD_dams_v1_1/MapServer/0/query")
OUT = Path(__file__).resolve().parent / "output" / "human_activity"
OUT.mkdir(exist_ok=True)
FIELDS = "GRAND_ID,DAM_NAME,RES_NAME,RIVER,COUNTRY,LONG_DD,LAT_DD,CAP_MCM,AREA_SKM,DAM_HGT_M,YEAR,MAIN_USE"

rows, offset = [], 0
while True:
    params = dict(where="1=1", outFields=FIELDS, returnGeometry="false",
                  f="json", resultRecordCount=2000, resultOffset=offset)
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, timeout=120)
            js = r.json()
            break
        except Exception:
            time.sleep(5)
    feats = js.get("features", [])
    if not feats:
        break
    rows.extend(f["attributes"] for f in feats)
    print(f"已下载 {len(rows)} 条", flush=True)
    if len(feats) < 2000:
        break
    offset += 2000

df = pd.DataFrame(rows)
df.to_csv(OUT / "grand_dams.csv", index=False, encoding="utf-8-sig")
print(f"完成：{len(df)} 座大坝 -> {OUT / 'grand_dams.csv'}")
print(df.head(3).to_string())
