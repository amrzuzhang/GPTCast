#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


PASS_THROUGH_STATIC_KEYS = (
    "sand_fraction",
    "clay_fraction",
    "silt_fraction",
    "porosity",
    "field_capacity",
    "wilting_point",
    "depth_to_bedrock_m",
    "depth_to_water_table_m",
    "topographic_wetness_index",
)


def _coord_name(ds: xr.Dataset, *candidates: str) -> str:
    for key in candidates:
        if key in ds.coords:
            return key
    raise KeyError(f"None of the coordinate names {candidates!r} were found in dataset coords={list(ds.coords)}")


def _single_data_var(ds: xr.Dataset) -> str:
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"Expected exactly one data variable, found {data_vars}")
    return data_vars[0]


def _open_reference_grid(base_dir: Path, year: int, source_filename: str, source_var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = base_dir / "land_surface" / f"{year:04d}" / source_filename
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    ds = xr.open_dataset(src)
    try:
        lat_name = _coord_name(ds, "latitude", "lat")
        lon_name = _coord_name(ds, "longitude", "lon")
        lat = ds[lat_name].values.astype(np.float32, copy=False)
        lon = ds[lon_name].values.astype(np.float32, copy=False)
        first = ds[source_var].isel(time=0).values.astype(np.float32, copy=False)
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return lat, lon, first


def _align_2d_field(path: Path, expected_var: str, ref_lat: np.ndarray, ref_lon: np.ndarray) -> np.ndarray:
    ds = xr.open_dataset(path)
    try:
        lat_name = _coord_name(ds, "latitude", "lat")
        lon_name = _coord_name(ds, "longitude", "lon")
        var_name = expected_var if expected_var in ds.data_vars else _single_data_var(ds)
        da = ds[var_name]
        if "time" in da.dims:
            da = da.isel(time=0)
        da = da.rename({lat_name: "latitude", lon_name: "longitude"})
        da = da.interp(latitude=ref_lat, longitude=ref_lon, method="linear")
        arr = da.values.astype(np.float32, copy=False)
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return arr


def _write_field(out_dir: Path, key: str, arr: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Path:
    coords = {"latitude": lat, "longitude": lon}
    dims = ("latitude", "longitude")
    ds = xr.Dataset({key: xr.DataArray(arr.astype(np.float32, copy=False), coords=coords, dims=dims)})
    path = out_dir / f"{key}.nc"
    try:
        ds.to_netcdf(path)
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return path


def _build_minimal_static(lat: np.ndarray, lon: np.ndarray, first: np.ndarray) -> dict[str, np.ndarray]:
    land_mask = (~np.isnan(first)).astype(np.float32)
    lat2d = np.broadcast_to(lat[:, None], first.shape).astype(np.float32, copy=False)
    lon2d = np.broadcast_to(lon[None, :], first.shape).astype(np.float32, copy=False)
    lat_norm = 2.0 * (lat2d - float(lat.min())) / (float(lat.max()) - float(lat.min()) + 1e-12) - 1.0
    lon_norm = 2.0 * (lon2d - float(lon.min())) / (float(lon.max()) - float(lon.min()) + 1e-12) - 1.0
    return {
        "land_mask": land_mask,
        "latitude_norm": lat_norm,
        "longitude_norm": lon_norm,
    }


def _derive_topography_features(elevation: np.ndarray, lat: np.ndarray, lon: np.ndarray, land_mask: np.ndarray) -> dict[str, np.ndarray]:
    elevation = np.where(land_mask > 0.5, elevation, np.nan).astype(np.float32, copy=False)
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("Need at least two latitude/longitude points to derive slope/aspect")

    lat_spacing_deg = float(np.abs(np.mean(np.diff(lat))))
    lon_spacing_deg = float(np.abs(np.mean(np.diff(lon))))
    mean_lat_rad = np.deg2rad(float(np.mean(lat)))
    dy_m = lat_spacing_deg * 111_320.0
    dx_m = lon_spacing_deg * 111_320.0 * np.cos(mean_lat_rad)
    elev_filled = np.nan_to_num(elevation, nan=float(np.nanmean(elevation)))
    dz_dlat, dz_dlon = np.gradient(elev_filled, dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(dz_dlat**2 + dz_dlon**2))
    slope_deg = np.rad2deg(slope_rad).astype(np.float32, copy=False)
    aspect_rad = np.arctan2(-dz_dlon, dz_dlat)
    aspect_sin = np.sin(aspect_rad).astype(np.float32, copy=False)
    aspect_cos = np.cos(aspect_rad).astype(np.float32, copy=False)

    slope_deg = np.where(land_mask > 0.5, slope_deg, np.nan)
    aspect_sin = np.where(land_mask > 0.5, aspect_sin, np.nan)
    aspect_cos = np.where(land_mask > 0.5, aspect_cos, np.nan)

    return {
        "elevation_m": elevation,
        "slope_degrees": slope_deg,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
    }


def _ancillary_file(ancillary_dir: Path, key: str) -> Path:
    return ancillary_dir / f"{key}.nc"


def build_static_features(
    *,
    base_dir: Path,
    year: int,
    source_filename: str,
    source_var: str,
    out_dir: Path,
    ancillary_dir: Path | None = None,
) -> list[Path]:
    lat, lon, first = _open_reference_grid(base_dir, year, source_filename, source_var)
    minimal = _build_minimal_static(lat, lon, first)
    land_mask = minimal["land_mask"]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key, arr in minimal.items():
        written.append(_write_field(out_dir, key, arr, lat, lon))

    if ancillary_dir is None:
        return written

    ancillary_dir = Path(ancillary_dir)
    elevation_path = _ancillary_file(ancillary_dir, "elevation_m")
    if elevation_path.exists():
        elevation = _align_2d_field(elevation_path, "elevation_m", lat, lon)
        topo = _derive_topography_features(elevation, lat, lon, land_mask)
        for key, arr in topo.items():
            written.append(_write_field(out_dir, key, arr, lat, lon))

    for key in PASS_THROUGH_STATIC_KEYS:
        src = _ancillary_file(ancillary_dir, key)
        if not src.exists():
            continue
        arr = _align_2d_field(src, key, lat, lon)
        arr = np.where(land_mask > 0.5, arr, np.nan).astype(np.float32, copy=False)
        written.append(_write_field(out_dir, key, arr, lat, lon))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build ERA5-Land static features for static-aware second-stage experiments. "
            "Always writes minimal geographic features (land_mask, latitude_norm, longitude_norm). "
            "If --ancillary-dir is provided, the script also looks for terrain/soil NetCDFs named after known keys "
            "(elevation_m, sand_fraction, clay_fraction, silt_fraction, porosity, field_capacity, "
            "wilting_point, depth_to_bedrock_m, depth_to_water_table_m, topographic_wetness_index). "
            "If elevation_m.nc is present, slope_degrees / aspect_sin / aspect_cos are derived automatically."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "0.1" / "1")
    parser.add_argument("--year", type=int, default=1979)
    parser.add_argument("--source-filename", default="volumetric_soil_water_layer_1.nc")
    parser.add_argument("--source-var", default="swvl1")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--ancillary-dir",
        type=Path,
        default=None,
        help="Optional directory of ancillary static NetCDFs. Each file should be named <key>.nc and contain a matching variable or a single data var.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.base_dir / "static")
    written = build_static_features(
        base_dir=args.base_dir,
        year=int(args.year),
        source_filename=str(args.source_filename),
        source_var=str(args.source_var),
        out_dir=Path(out_dir),
        ancillary_dir=None if args.ancillary_dir is None else Path(args.ancillary_dir),
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
