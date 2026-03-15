from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


SUPPORTED_ECMWF_INIT_WEEKDAYS = (0, 3)  # Monday, Thursday


@dataclass(frozen=True)
class ECMWFGuidanceSpec:
    key: str
    short_name: str
    out_var: str
    clip_and_normalize: tuple[float, float, float, float]


ECMWF_GUIDANCE_VARIABLES: dict[str, ECMWFGuidanceSpec] = {
    "ecmwf_sm100": ECMWFGuidanceSpec(
        key="ecmwf_sm100",
        short_name="sm100",
        out_var="ecmwf_sm100",
        clip_and_normalize=(0.0, 0.8, -1.0, 1.0),
    ),
    "ecmwf_2t": ECMWFGuidanceSpec(
        key="ecmwf_2t",
        short_name="2t",
        out_var="ecmwf_2t",
        clip_and_normalize=(240.0, 330.0, -1.0, 1.0),
    ),
}


def parse_metadata_dates(metadata_path_or_df: str | pd.DataFrame) -> list[datetime]:
    if isinstance(metadata_path_or_df, pd.DataFrame):
        df = metadata_path_or_df
    else:
        df = pd.read_csv(metadata_path_or_df, index_col="id", converters={"files": ast.literal_eval})

    timestamps: list[datetime] = []
    for files in df["files"]:
        for item in files:
            timestamps.append(datetime.strptime(Path(item).name, "SM_%Y%m%d%H%M"))
    return timestamps


def derive_candidate_init_dates(
    metadata_sources: Sequence[str | pd.DataFrame],
    *,
    weekdays: Sequence[int] = SUPPORTED_ECMWF_INIT_WEEKDAYS,
    min_year: int = 1995,
) -> list[datetime]:
    allowed_weekdays = {int(x) for x in weekdays}
    all_dates: set[datetime] = set()
    for source in metadata_sources:
        for dt in parse_metadata_dates(source):
            if dt.year < int(min_year):
                continue
            if dt.weekday() not in allowed_weekdays:
                continue
            all_dates.add(datetime(dt.year, dt.month, dt.day))
    return sorted(all_dates)


def default_guidance_root(base_dir: str | Path) -> Path:
    return Path(base_dir) / "guidance" / "ecmwf"
