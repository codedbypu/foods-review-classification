"""
GeoJSON export for scored reviews (map viz).

COMMON ERRORS:
  - ValueError: Missing required column lat/lon/review_id — add coordinates or omit --geojson_out in score_and_flag.
  - NaN lat/lon: cast/filter rows before export (float() will fail on NaN in strict pipelines).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def to_geojson_points(
    df: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    id_col: str = "review_id",
) -> dict:
    """
    Convert a scored dataframe to a GeoJSON FeatureCollection of Points.
    Requires lat/lon columns. All other columns are copied into properties.
    """
    for c in (lat_col, lon_col, id_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    features = []
    prop_cols = [c for c in df.columns if c not in {lat_col, lon_col}]
    for _, row in df.iterrows():
        lon = float(row[lon_col])
        lat = float(row[lat_col])
        props = {c: row[c] for c in prop_cols}
        features.append(
            {
                "type": "Feature",
                "id": str(row[id_col]),
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_geojson(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

