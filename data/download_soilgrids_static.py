#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time as time_module

import numpy as np
import xarray as xr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_era5land_static_features import build_static_features


SOILGRIDS_REQUIRED_KEYS = ("sand", "clay", "silt", "bdod")
DEPTHS_0_100CM = (
    ("0-5cm", 5.0),
    ("5-15cm", 10.0),
    ("15-30cm", 15.0),
    ("30-60cm", 30.0),
    ("60-100cm", 40.0),
)


@dataclass(frozen=True)
class ReferenceGrid:
    lat: np.ndarray
    lon: np.ndarray
    height: int
    width: int


def _require_soilgrids():
    try:
        from soilgrids import SoilGrids  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: soilgrids\n"
            "Install with one of:\n"
            "  pip install soilgrids\n"
            "  conda install -c conda-forge soilgrids\n"
            f"Original error: {type(exc).__name__}: {exc}"
        )
    return SoilGrids


def _reference_grid(base_dir: Path, year: int, source_filename: str, source_var: str) -> ReferenceGrid:
    src = base_dir / "land_surface" / f"{year:04d}" / source_filename
    if not src.exists():
        raise FileNotFoundError(f"Missing reference source file: {src}")
    ds = xr.open_dataset(src)
    try:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat = ds[lat_name].values.astype(np.float32, copy=False)
        lon = ds[lon_name].values.astype(np.float32, copy=False)
        _ = ds[source_var].isel(time=0)
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return ReferenceGrid(lat=lat, lon=lon, height=int(lat.shape[0]), width=int(lon.shape[0]))


def _download_coverage(
    soil_grids,
    *,
    service_id: str,
    coverage_id: str,
    grid: ReferenceGrid,
    output_path: Path,
    max_retries: int = 6,
    retry_sleep_seconds: float = 5.0,
    retry_backoff: float = 2.0,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tif_path = output_path.with_suffix(".tif")

    def _read_tif(path: Path) -> np.ndarray:
        try:
            import rasterio  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "Missing dependency: rasterio\n"
                "Install with one of:\n"
                "  python -m pip install rasterio\n"
                "  conda install -c conda-forge rasterio\n"
                f"Original error: {type(exc).__name__}: {exc}"
            )

        with rasterio.open(path) as src:
            arr_local = src.read(1).astype(np.float32, copy=False)
        return arr_local

    if tif_path.exists():
        print(f"[soilgrids] reusing cached {tif_path.name}")
        arr = _read_tif(tif_path)
    else:
        delay = float(retry_sleep_seconds)
        last_error: Exception | None = None
        arr = None
        for attempt in range(1, int(max_retries) + 1):
            try:
                data = soil_grids.get_coverage_data(
                    service_id=service_id,
                    coverage_id=coverage_id,
                    west=float(grid.lon.min()),
                    south=float(grid.lat.min()),
                    east=float(grid.lon.max()),
                    north=float(grid.lat.max()),
                    crs="urn:ogc:def:crs:EPSG::4326",
                    width=int(grid.width),
                    height=int(grid.height),
                    response_crs="urn:ogc:def:crs:EPSG::4326",
                    output=str(tif_path),
                )
                if hasattr(data, "values"):
                    arr = np.asarray(data.values, dtype=np.float32)
                else:
                    arr = np.asarray(data, dtype=np.float32)
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"[soilgrids] attempt {attempt}/{max_retries} failed for {coverage_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt >= int(max_retries):
                    raise RuntimeError(
                        f"Failed to download {coverage_id} after {max_retries} attempts. "
                        f"You can rerun the same command later; any cached *.tif files will be reused."
                    ) from exc
                time_module.sleep(delay)
                delay *= float(retry_backoff)

        if arr is None:
            raise RuntimeError(f"Download unexpectedly returned no data for {coverage_id}") from last_error

    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.shape != (grid.height, grid.width):
        raise ValueError(f"Unexpected downloaded shape for {coverage_id}: got {arr.shape}, expected {(grid.height, grid.width)}")
    return arr


def _weighted_average_depths(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    stacked = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    weights_np = np.asarray(weights, dtype=np.float32)
    weights_np = weights_np / weights_np.sum()
    return np.tensordot(weights_np, stacked, axes=(0, 0)).astype(np.float32, copy=False)


def _write_static(out_path: Path, key: str, arr: np.ndarray, grid: ReferenceGrid) -> Path:
    ds = xr.Dataset(
        {key: xr.DataArray(arr.astype(np.float32, copy=False), coords={"latitude": grid.lat, "longitude": grid.lon}, dims=("latitude", "longitude"))}
    )
    try:
        ds.to_netcdf(out_path)
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return out_path


def _convert_texture_fraction(raw: np.ndarray) -> np.ndarray:
    # SoilGrids FAQ: sand/clay/silt in g/kg; divide by 1000 for mass fraction [0,1].
    return np.clip(raw / 1000.0, 0.0, 1.0).astype(np.float32, copy=False)


def _convert_bulk_density_kg_dm3(raw: np.ndarray) -> np.ndarray:
    # SoilGrids FAQ: bdod in cg/cm3; divide by 100 for kg/dm3 (equivalent to g/cm3).
    return np.clip(raw / 100.0, 0.0, None).astype(np.float32, copy=False)


def _convert_volumetric_fraction(raw: np.ndarray) -> np.ndarray:
    # SoilGrids FAQ: wv003 / wv1500 in 10^-3 cm3/cm3; divide by 1000 for fraction [0,1].
    return np.clip(raw / 1000.0, 0.0, 1.0).astype(np.float32, copy=False)


def download_soilgrids_static(
    *,
    base_dir: Path,
    year: int = 1979,
    source_filename: str = "volumetric_soil_water_layer_1.nc",
    source_var: str = "swvl1",
    raw_out_dir: Path | None = None,
    final_static_dir: Path | None = None,
    build_minimal_static: bool = True,
    max_retries: int = 6,
    retry_sleep_seconds: float = 5.0,
    retry_backoff: float = 2.0,
) -> list[Path]:
    """Download SoilGrids-derived static properties aligned to the ERA5-Land grid.

    This downloader complements the CDS ERA5-Land script: ERA5-Land itself does not provide terrain/soil static
    properties such as texture fractions or soil water retention. The current implementation downloads SoilGrids
    layers for 0-100 cm and derives:
    - sand_fraction
    - clay_fraction
    - silt_fraction
    - porosity          (from bdod, particle density fixed at 2.65 g/cm3)

    Notes:
    - Depth aggregation is a thickness-weighted mean over 0-5, 5-15, 15-30, 30-60, and 60-100 cm.
    - The current SoilGrids Python wrapper/service map does not expose wv003/wv1500 in service_ids,
      so field_capacity / wilting_point are NOT downloaded automatically here.
      If you have those layers from another source, the static builder can still ingest them as
      local NetCDFs named field_capacity.nc / wilting_point.nc.
    - The downloaded outputs are saved as NetCDFs that can be consumed directly by build_era5land_static_features.py.
    """

    SoilGrids = _require_soilgrids()
    soil_grids = SoilGrids()
    grid = _reference_grid(base_dir, year, source_filename, source_var)

    raw_dir = base_dir / "static_raw" if raw_out_dir is None else Path(raw_out_dir)
    final_dir = base_dir / "static" if final_static_dir is None else Path(final_static_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, np.ndarray] = {}
    for key in SOILGRIDS_REQUIRED_KEYS:
        depth_arrays: list[np.ndarray] = []
        depth_weights: list[float] = []
        for depth, thickness_cm in DEPTHS_0_100CM:
            coverage_id = f"{key}_{depth}_mean"
            out_path = raw_dir / f"{coverage_id}.nc"
            print(f"[soilgrids] downloading {coverage_id}")
            raw = _download_coverage(
                soil_grids,
                service_id=key,
                coverage_id=coverage_id,
                grid=grid,
                output_path=out_path,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
                retry_backoff=retry_backoff,
            )
            depth_arrays.append(raw)
            depth_weights.append(thickness_cm)
        downloaded[key] = _weighted_average_depths(depth_arrays, depth_weights)

    outputs: list[Path] = []
    sand_fraction = _convert_texture_fraction(downloaded["sand"])
    clay_fraction = _convert_texture_fraction(downloaded["clay"])
    silt_fraction = _convert_texture_fraction(downloaded["silt"])
    bulk_density = _convert_bulk_density_kg_dm3(downloaded["bdod"])
    porosity = np.clip(1.0 - bulk_density / 2.65, 0.0, 1.0).astype(np.float32, copy=False)

    for key, arr in {
        "sand_fraction": sand_fraction,
        "clay_fraction": clay_fraction,
        "silt_fraction": silt_fraction,
        "porosity": porosity,
    }.items():
        outputs.append(_write_static(raw_dir / f"{key}.nc", key, arr, grid))

    if build_minimal_static:
        outputs.extend(
            build_static_features(
                base_dir=base_dir,
                year=year,
                source_filename=source_filename,
                source_var=source_var,
                out_dir=final_dir,
                ancillary_dir=raw_dir,
            )
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download SoilGrids-based soil static features aligned to the ERA5-Land grid. "
            "This is separate from the CDS ERA5-Land downloader because SoilGrids is an external data source."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "0.1" / "1")
    parser.add_argument("--year", type=int, default=1979)
    parser.add_argument("--source-filename", default="volumetric_soil_water_layer_1.nc")
    parser.add_argument("--source-var", default="swvl1")
    parser.add_argument("--raw-out-dir", type=Path, default=None, help="Directory to store aligned SoilGrids raw static NetCDFs.")
    parser.add_argument("--final-static-dir", type=Path, default=None, help="Directory to store final static NetCDFs consumed by the hydro datamodule.")
    parser.add_argument(
        "--no-build-minimal-static",
        action="store_true",
        help="Only download/write SoilGrids-derived soil files; do not also regenerate land_mask/latitude_norm/longitude_norm.",
    )
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = download_soilgrids_static(
        base_dir=Path(args.base_dir),
        year=int(args.year),
        source_filename=str(args.source_filename),
        source_var=str(args.source_var),
        raw_out_dir=None if args.raw_out_dir is None else Path(args.raw_out_dir),
        final_static_dir=None if args.final_static_dir is None else Path(args.final_static_dir),
        build_minimal_static=not bool(args.no_build_minimal_static),
        max_retries=int(args.max_retries),
        retry_sleep_seconds=float(args.retry_sleep_seconds),
        retry_backoff=float(args.retry_backoff),
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
