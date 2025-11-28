"""
3.3.2 공간 분석 파이프라인.

- 입력 데이터: uda_final_topics_clean.csv
- 대상 소스: 새올 전자민원창구_2
- region_code: MongoDB에서 동일 content를 가진 문서의 region_code를 사용

기본 Mongo 접속 정보(3.1 정제 파이프라인과 동일)를 내장했으며,
필요 시 환경변수(MONGO_URI / MONGO_DB / MONGO_COLLECTION)로 덮어쓸 수 있다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import geopandas as gpd
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "uda_final_topics_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "spatial"
POP_PATH = BASE_DIR / "reference" / "인구밀도_20251124230406.csv"
GEO_PATH = BASE_DIR / "reference" / "3_서울시_자치구.geojson"
TARGET_SOURCE = "새올 전자민원창구_2"

DEFAULT_MONGO_URI = "mongodb+srv://team5:1886@unstructureddataanalysi.h5nf52z.mongodb.net/"
DEFAULT_MONGO_DB = "UDA"
DEFAULT_MONGO_COLLECTION = "새올 전자민원창구_2"


def load_seoul_geo() -> gpd.GeoDataFrame:
    """서울 25개 자치구 GeoJSON 로드 및 EPSG:4326 보정."""
    gdf = gpd.read_file(GEO_PATH)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    seoul = gdf.copy()
    seoul["region_label"] = "서울" + seoul["SIG_KOR_NM"].str.replace(" ", "", regex=False)
    return seoul


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_filtered_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    cols = {"content", "main_topic_manual", "source", "date"}
    missing = cols.difference(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    df = df[df["source"] == TARGET_SOURCE].copy()
    if df.empty:
        raise ValueError(f"{TARGET_SOURCE} 데이터가 없습니다.")
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    return df


def fetch_region_codes(contents: pd.Series, batch_size: int = 500) -> Dict[str, str]:
    mongo_uri = os.getenv("MONGO_URI", DEFAULT_MONGO_URI)
    mongo_db = os.getenv("MONGO_DB", DEFAULT_MONGO_DB)
    mongo_col = os.getenv("MONGO_COLLECTION", DEFAULT_MONGO_COLLECTION)

    client = MongoClient(mongo_uri)
    collection = client[mongo_db][mongo_col]
    mapping: Dict[str, str] = {}
    unique_contents = list(contents.dropna().unique())
    for start in range(0, len(unique_contents), batch_size):
        chunk = unique_contents[start : start + batch_size]
        cursor = collection.find(
            {"content": {"$in": chunk}},
            {"content": 1, "region_code": 1, "_id": 0},
        )
        for doc in cursor:
            content = doc.get("content")
            region_code = doc.get("region_code")
            if content and region_code and content not in mapping:
                mapping[content] = region_code
    return mapping


def attach_region_code(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    df["region_code"] = df["content"].map(mapping)
    missing = df["region_code"].isna().sum()
    if missing:
        sample = df[df["region_code"].isna()].head(5)["content"].tolist()
        raise ValueError(
            f"region_code를 찾지 못한 행이 {missing}건 있습니다. 예시: {sample}"
        )
    df["normalized_region"] = (
        df["region_code"]
        .str.replace("서울시", "서울", regex=False)
        .str.replace("서울특별시", "서울", regex=False)
    )
    return df


def load_population() -> pd.DataFrame | None:
    if not POP_PATH.exists():
        return None
    pop_df = pd.read_csv(POP_PATH, header=1)
    rename_map = {
        pop_df.columns[0]: "level1",
        pop_df.columns[1]: "district",
        pop_df.columns[2]: "subdistrict",
        pop_df.columns[3]: "population",
    }
    pop_df = pop_df.rename(columns=rename_map)
    pop_df = pop_df[pop_df["subdistrict"] == "소계"]
    pop_df = pop_df[pop_df["district"].notna() & (pop_df["district"] != "소계")]
    pop_df["population"] = pd.to_numeric(pop_df["population"], errors="coerce")
    pop_df = pop_df.dropna(subset=["population"])
    pop_df["region_code"] = "서울" + pop_df["district"].str.replace(" ", "", regex=False)
    pop_df = (
        pop_df.groupby("region_code", as_index=False)["population"]
        .sum()
    )
    return pop_df


def compute_spatial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["main_topic_manual", "normalized_region"])
        .size()
        .reset_index(name="count")
    )
    pop_df = load_population()
    if pop_df is not None:
        grouped = grouped.merge(pop_df, left_on="normalized_region", right_on="region_code", how="left")
        grouped["per_100k"] = (
            grouped["count"] / grouped["population"] * 100_000
        ).round(2)
    grouped = grouped.rename(columns={"normalized_region": "region_label"})
    seoul_geo = load_seoul_geo()
    full_regions = seoul_geo[["region_label"]].drop_duplicates()
    grouped = full_regions.merge(grouped, on="region_label", how="left").fillna({"count": 0})
    return grouped


def export_tables(spatial_df: pd.DataFrame) -> None:
    spatial_df.to_csv(OUTPUT_DIR / "topic_region_counts.csv", index=False)


def plot_choropleth(spatial_df: pd.DataFrame) -> None:
    if not GEO_PATH.exists():
        print("서울 자치구 GeoJSON 파일이 없어 Choropleth를 건너뜁니다.")
        return

    seoul_geo = load_seoul_geo()

    for topic in spatial_df["main_topic_manual"].unique():
        sub = spatial_df[spatial_df["main_topic_manual"] == topic]
        merged = seoul_geo.merge(sub, on="region_label", how="left")
        fig = px.choropleth(
            merged,
            geojson=json.loads(merged.to_json()),
            locations=merged.index,
            color="count",
            hover_name="SIG_KOR_NM",
            hover_data=["per_100k"] if "per_100k" in merged.columns else None,
            title=f"{topic} | 지역별 민원 건수",
            color_continuous_scale="Reds",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.write_html(
            OUTPUT_DIR / f"choropleth_{topic}.html",
            include_plotlyjs="cdn",
        )


def main() -> None:
    _ensure_output_dir()
    df = load_filtered_data()
    mapping = fetch_region_codes(df["content"])
    df = attach_region_code(df, mapping)
    spatial_df = compute_spatial_metrics(df)
    export_tables(spatial_df)
    plot_choropleth(spatial_df)


if __name__ == "__main__":
    main()

