"""서울 생활인구(행정동) 공공데이터 → 데모용 작은 파일 2개.

원본(공공누리 1유형, 출처: 서울특별시 열린데이터광장)
  - 행정동 단위 서울 생활인구(내국인) OA-14991 · LOCAL_PEOPLE_DONG_YYYYMM.zip
  - 서울시 상권분석서비스(영역-행정동) shp  (seoul-living-population-gis 저장소 01.data/01.raw_data/)
결과(이 폴더 data/)
  - seoul_dong.geojson            행정동 425개 경계(위경도, 단순화)
  - seoul_dong_pop_YYYYMM.csv     행정동 × 주중/주말 × 시간대(0~23) 평균 생활인구 + 면적(㎢)

  uv run --with pandas --with pyshp --with pyproj --with shapely python prepare_seoul.py <zip 경로> <shp 경로(확장자 없이)>
"""
import io, json, os, sys, zipfile

import pandas as pd
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_population(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(io.BytesIO(z.read(name)), encoding="utf-8-sig", dtype=str, index_col=False,   # 줄 끝 쉼표로 칸이 하나 많음
                         usecols=["기준일ID", "시간대구분", "행정동코드", "총생활인구수"])
    month = name.split("_")[-1][:6]
    df["총생활인구수"] = pd.to_numeric(df["총생활인구수"], errors="coerce")
    d = pd.to_datetime(df["기준일ID"], format="%Y%m%d")
    df["day_type"] = d.dt.day_of_week.map(lambda w: "주말" if w >= 5 else "주중")
    df["hour"] = df["시간대구분"].astype(int)
    agg = (df.groupby(["행정동코드", "day_type", "hour"])["총생활인구수"].mean().round(1)
             .reset_index().rename(columns={"행정동코드": "dong_code", "총생활인구수": "pop"}))
    return agg, month


def load_boundaries(shp_base):
    r = shapefile.Reader(shp_base, encoding="utf-8")
    prj = open(shp_base + ".prj", encoding="utf-8").read()
    to_wgs = Transformer.from_crs(CRS.from_wkt(prj), "EPSG:4326", always_xy=True).transform
    feats, meta = [], []
    for sr in r.shapeRecords():
        rec = sr.record.as_dict()
        g = shape(sr.shape.__geo_interface__).simplify(15, preserve_topology=True)     # 15m 단순화
        area_km2 = shape(sr.shape.__geo_interface__).area / 1e6
        g = transform(to_wgs, g)
        geo = json.loads(json.dumps(mapping(g)), parse_float=lambda x: round(float(x), 5))
        code = str(rec["ADSTRD_CD"]).strip()
        feats.append({"type": "Feature", "id": code, "properties": {"code": code, "name": rec["ADSTRD_NM"].strip()},
                      "geometry": geo})
        c = g.representative_point()
        meta.append({"dong_code": code, "dong_name": rec["ADSTRD_NM"].strip(), "area_km2": round(area_km2, 4),
                     "lon": round(c.x, 5), "lat": round(c.y, 5)})
    return {"type": "FeatureCollection", "features": feats}, pd.DataFrame(meta)


def main(zip_path, shp_base):
    os.makedirs(OUT, exist_ok=True)
    pop, month = load_population(zip_path)
    gj, meta = load_boundaries(shp_base)
    pop = pop.merge(meta, on="dong_code", how="inner")
    missing = set(meta["dong_code"]) - set(pop["dong_code"])
    with open(os.path.join(OUT, "seoul_dong.geojson"), "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, separators=(",", ":"))
    pop.to_csv(os.path.join(OUT, f"seoul_dong_pop_{month}.csv"), index=False, encoding="utf-8")
    print(f"행정동 {pop['dong_code'].nunique()}개 · 행 {len(pop)} · 경계에만 있는 동 {len(missing)}개 · 월 {month}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
