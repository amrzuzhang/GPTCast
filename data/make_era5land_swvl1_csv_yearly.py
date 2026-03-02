#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate ERA5-Land swvl1 metadata CSVs (yearly, MIARAD-style).

We keep the same CSV schema as MIARAD:
  id,start_datetime,end_datetime,seq_length,pixel_average,files

Each row contains a Python-list-as-string in `files`, where each element is a key:
  era5land/SM_YYYYMMDDHHMM

Why "yearly":
- Easier to inspect and edit than huge multi-year segments.
- Still safe for seq_len>1: the dataset implementation filters out windows that
  cross any time-axis discontinuity (e.g., 2018-12 time shift).

This script does NOT download data; it assumes yearly NetCDF files exist at:
  <base_dir>/<year>/volumetric_soil_water_layer_1.nc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr


EXPECTED_GAP = pd.Timedelta(days=1)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    years: list[int]


def _iter_years(start: int, end: int) -> list[int]:
    assert start <= end
    return list(range(int(start), int(end) + 1))


def _load_times(nc_path: Path) -> pd.DatetimeIndex:
    ds = xr.open_dataset(nc_path)
    try:
        t = pd.to_datetime(ds["time"].values)
    finally:
        try:
            ds.close()
        except Exception:
            pass
    # Ensure monotonic order.
    return pd.DatetimeIndex(t).sort_values()


def _split_by_discontinuity(times: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """Split a year's times into 1-day-continuous segments."""
    if len(times) == 0:
        return []
    segments: list[pd.DatetimeIndex] = []
    start = 0
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]) != EXPECTED_GAP:
            segments.append(times[start:i])
            start = i
    segments.append(times[start:])
    return segments


def _make_rows(
    *,
    base_dir: Path,
    years: Iterable[int],
    yearly_filename: str,
) -> list[dict]:
    rows: list[dict] = []
    next_id = 0

    for year in years:
        nc_path = base_dir / str(year) / yearly_filename
        if not nc_path.exists():
            raise FileNotFoundError(f"Missing yearly NetCDF: {nc_path}")

        times = _load_times(nc_path)
        segments = _split_by_discontinuity(times)

        for seg in segments:
            if len(seg) == 0:
                continue

            keys = [f"era5land/SM_{ts.strftime('%Y%m%d%H%M')}" for ts in seg]

            rows.append(
                {
                    "id": next_id,
                    "start_datetime": seg[0].isoformat(),
                    "end_datetime": seg[-1].isoformat(),
                    "seq_length": int(len(keys)),
                    "pixel_average": 0.0,
                    "files": keys,
                }
            )
            next_id += 1

    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Keep the schema identical to MIARAD files.
    df = df[["id", "start_datetime", "end_datetime", "seq_length", "pixel_average", "files"]]
    df.to_csv(out_path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "0.1"
    base_dir = data_dir / "1" / "land_surface"

    yearly_filename = "volumetric_soil_water_layer_1.nc"

    splits = [
        SplitSpec("train", _iter_years(1979, 2014)),
        SplitSpec("val", _iter_years(2015, 2018)),
        SplitSpec("test", _iter_years(2019, 2020)),
    ]

    for spec in splits:
        rows = _make_rows(base_dir=base_dir, years=spec.years, yearly_filename=yearly_filename)
        out_path = data_dir / f"era5land_swvl1_{spec.name}.csv"
        _write_csv(rows, out_path)
        total = sum(int(r["seq_length"]) for r in rows)
        print(f"[{spec.name}] rows={len(rows)} total_frames={total} -> {out_path}")


if __name__ == "__main__":
    main()

