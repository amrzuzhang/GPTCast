#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Download ERA5-Land swvl1 (soil moisture layer 1) from Copernicus CDS.

This script is meant to populate the on-disk layout expected by:
  `gptcast.data.era5land_swvl1.Era5LandSwvl1`

Default output layout:
  <repo>/data/0.1/1/land_surface/<year>/volumetric_soil_water_layer_1.nc

Important: credentials are NOT stored in this repository.
Configure cdsapi via either:
1) ~/.cdsapirc (recommended)
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-TOKEN>
2) environment variables:
   export CDSAPI_URL="https://cds.climate.copernicus.eu/api"
   export CDSAPI_KEY="<YOUR-TOKEN>"

Notes:
- We download one fixed time per day to create a *daily* time axis, which matches the
  dataset code's `seq_len>1` assumption (exact 1-day gaps).
- Downloads are done month-by-month to reduce the chance of CDS timeouts.
- We then concatenate the 12 monthly NetCDFs into a single yearly NetCDF.
"""

from __future__ import annotations

import calendar
import concurrent.futures as cf
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence, Tuple


def _require(module_name: str) -> None:
    try:
        __import__(module_name)
    except Exception as e:
        raise SystemExit(
            f"Missing dependency: {module_name}\n"
            f"Install one of:\n"
            f"  pip install {module_name}\n"
            f"  conda install -c conda-forge {module_name}\n"
            f"Original error: {type(e).__name__}: {e}"
        )


def _month_str(m: int) -> str:
    return f"{int(m):02d}"


def _day_list(year: int, month: int) -> list[str]:
    last = calendar.monthrange(int(year), int(month))[1]
    return [f"{d:02d}" for d in range(1, last + 1)]


def _safe_print_cds_config_hint() -> None:
    # Never print secrets; just hint.
    cfg = Path.home() / ".cdsapirc"
    has_cfg = cfg.exists()
    has_env = bool(os.environ.get("CDSAPI_URL")) or bool(os.environ.get("CDSAPI_KEY"))
    if has_cfg or has_env:
        return
    print(
        "\n[cdsapi] Credentials not detected.\n"
        "Create ~/.cdsapirc (recommended):\n"
        "  url: https://cds.climate.copernicus.eu/api\n"
        "  key: <YOUR-TOKEN>\n"
        "Or set env vars:\n"
        "  export CDSAPI_URL=\"https://cds.climate.copernicus.eu/api\"\n"
        "  export CDSAPI_KEY=\"<YOUR-TOKEN>\"\n",
        file=sys.stderr,
    )


def _read_magic(path: Path, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _is_zip(path: Path) -> bool:
    # ZIP local file header / empty archive / spanned archive
    magic = _read_magic(path, n=4)
    return magic in {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}


def _is_netcdf(path: Path) -> bool:
    # NetCDF classic: b"CDF\x01"/b"CDF\x02"
    # NetCDF4 is HDF5: b"\x89HDF\r\n\x1a\n"
    magic8 = _read_magic(path, n=8)
    if magic8.startswith(b"CDF"):
        return True
    if magic8 == b"\x89HDF\r\n\x1a\n":
        return True
    return False


def _ensure_netcdf_inplace(path: Path) -> None:
    """Ensure `path` is a readable NetCDF file.

    CDS often returns NetCDF packaged inside a ZIP, even if you name the output *.nc.
    If `path` is a ZIP, we extract the single contained *.nc and replace `path` in-place.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    if _is_netcdf(path):
        return

    if _is_zip(path):
        tmp_out = path.parent / (path.name + ".tmp")
        try:
            try:
                zf_ctx = zipfile.ZipFile(path)
            except zipfile.BadZipFile as e:
                raise ValueError(f"Corrupted ZIP (likely partial download): {path}") from e

            with zf_ctx as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                if not members:
                    raise ValueError(f"ZIP archive is empty: {path}")

                # Prefer *.nc, otherwise fall back to single-file archives.
                nc_members = [m for m in members if m.lower().endswith(".nc")]
                if len(nc_members) == 1:
                    member = nc_members[0]
                elif len(members) == 1:
                    member = members[0]
                else:
                    raise ValueError(
                        f"ZIP archive contains multiple files; cannot choose automatically: {path} -> {members}"
                    )

                print(f"[unzip] {path.name} -> {member}")
                with zf.open(member) as src, open(tmp_out, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            tmp_out.replace(path)
        finally:
            try:
                tmp_out.unlink()
            except FileNotFoundError:
                pass

        if not _is_netcdf(path):
            raise ValueError(f"After unzip, file is still not NetCDF: {path}")
        return

    # Not NetCDF and not ZIP: keep the failure mode clear.
    raise ValueError(
        f"File is neither NetCDF nor ZIP (CDS sometimes returns ZIP-wrapped NetCDF). "
        f"Bad file: {path}"
    )


def _download_month(
    *,
    client,
    dataset: str,
    var_request: str,
    year: int,
    month: int,
    time: str,
    area_list: Sequence[float],
    out_path: Path,
) -> None:
    m = _month_str(month)
    req = {
        "product_type": "reanalysis",
        "variable": var_request,
        "year": str(year),
        "month": m,
        "day": _day_list(year, month),
        "time": time,
        "area": list(area_list),
        "format": "netcdf",
    }

    print(f"[download] year={year} month={m} -> {out_path.name}")
    client.retrieve(dataset, req, str(out_path))


def _ensure_month_ok_or_redownload(
    *,
    client,
    dataset: str,
    var_request: str,
    year: int,
    month: int,
    time: str,
    area_list: Sequence[float],
    month_path: Path,
    overwrite: bool,
    max_attempts: int = 3,
) -> None:
    """Validate or (re)download a monthly file.

    This handles the common failure mode where CDS leaves a partial/corrupt file
    on disk (e.g. truncated ZIP). In that case we delete and retry the download.
    """

    last_err: Exception | None = None
    for attempt in range(1, int(max_attempts) + 1):
        if month_path.exists() and not overwrite:
            try:
                _ensure_netcdf_inplace(month_path)
                return
            except Exception as e:
                last_err = e
                print(
                    f"[warn] year={year} month={month:02d} existing file invalid "
                    f"(attempt {attempt}/{max_attempts}): {type(e).__name__}: {e}"
                )
                try:
                    month_path.unlink()
                except FileNotFoundError:
                    pass

        try:
            _download_month(
                client=client,
                dataset=dataset,
                var_request=var_request,
                year=year,
                month=month,
                time=time,
                area_list=area_list,
                out_path=month_path,
            )
            _ensure_netcdf_inplace(month_path)
            return
        except Exception as e:
            last_err = e
            print(
                f"[warn] year={year} month={month:02d} download/convert failed "
                f"(attempt {attempt}/{max_attempts}): {type(e).__name__}: {e}"
            )
            try:
                month_path.unlink()
            except FileNotFoundError:
                pass

    raise RuntimeError(
        f"Failed to obtain a valid monthly file after {max_attempts} attempts: {month_path}\n"
        f"Last error: {type(last_err).__name__}: {last_err}"
    )


def _concat_months_to_year(
    *,
    month_paths: Sequence[Path],
    out_path: Path,
    var_name: str = "swvl1",
    compression_level: int = 4,
) -> None:
    _require("xarray")

    import xarray as xr

    if not month_paths:
        raise ValueError("No monthly files to concatenate.")

    dsets = []
    for p in month_paths:
        _ensure_netcdf_inplace(p)
        ds = xr.open_dataset(p)
        # CDS sometimes uses `valid_time` instead of `time` even for reanalysis products.
        # Normalize here so the yearly file matches the expectations of our dataset code.
        if "time" not in ds.dims and "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        # Keep only the essential coordinates to make concat robust across months.
        keep_coords = {"time", "latitude", "longitude"}
        drop_coords = [c for c in ds.coords if c not in keep_coords]
        if drop_coords:
            ds = ds.drop_vars(drop_coords, errors="ignore")
        # Normalize variable name to `swvl1` for downstream code.
        if var_name not in ds.data_vars:
            if "volumetric_soil_water_layer_1" in ds.data_vars:
                ds = ds.rename({"volumetric_soil_water_layer_1": var_name})
            elif len(ds.data_vars) == 1:
                only = next(iter(ds.data_vars))
                ds = ds.rename({only: var_name})
            else:
                raise KeyError(
                    f"Expected variable '{var_name}' in {p.name}; found: {list(ds.data_vars)}"
                )
        # Keep only what we need.
        ds = ds[[var_name]]
        dsets.append(ds)

    try:
        year_ds = xr.concat(
            dsets,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        ).sortby("time")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            var_name: {
                "zlib": True,
                "complevel": int(compression_level),
                "shuffle": True,
                "dtype": "float32",
            }
        }
        year_ds[var_name] = year_ds[var_name].astype("float32")
        year_ds.to_netcdf(out_path, encoding=encoding)
    finally:
        for ds in dsets:
            try:
                ds.close()
            except Exception:
                pass


def _download_one_year(
    *,
    year: int,
    out_root: Path,
    client,
    dataset: str,
    var_request: str,
    time: str,
    area_list: Sequence[float],
    overwrite: bool,
    keep_monthly: bool,
    compression_level: int,
    yearly_filename: str,
) -> str:
    year_dir = out_root / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    yearly_path = year_dir / yearly_filename

    if yearly_path.exists() and not overwrite:
        return f"[skip] {yearly_path} exists"

    tmp_dir = year_dir / "_tmp_monthly"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    month_paths: list[Path] = []
    for month in range(1, 13):
        m = _month_str(month)
        month_path = tmp_dir / f"{var_request}_{year}{m}.nc"
        month_paths.append(month_path)
        _ensure_month_ok_or_redownload(
            client=client,
            dataset=dataset,
            var_request=var_request,
            year=year,
            month=month,
            time=time,
            area_list=area_list,
            month_path=month_path,
            overwrite=overwrite,
            max_attempts=3,
        )

    print(f"[concat] year={year} -> {yearly_path.name}")
    _concat_months_to_year(
        month_paths=month_paths,
        out_path=yearly_path,
        var_name="swvl1",
        compression_level=compression_level,
    )

    if not keep_monthly:
        for p in month_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            # Directory not empty (partial downloads, etc.)
            pass

    return f"[done] {yearly_path}"


def _download_one_year_worker(
    year: int,
    out_root: str,
    dataset: str,
    var_request: str,
    time: str,
    area_list: Tuple[float, float, float, float],
    overwrite: bool,
    keep_monthly: bool,
    compression_level: int,
    yearly_filename: str,
) -> str:
    # Each process creates its own cdsapi.Client; do NOT share clients across processes.
    _require("cdsapi")
    import cdsapi  # type: ignore

    client = cdsapi.Client()
    return _download_one_year(
        year=int(year),
        out_root=Path(out_root),
        client=client,
        dataset=str(dataset),
        var_request=str(var_request),
        time=str(time),
        area_list=area_list,
        overwrite=bool(overwrite),
        keep_monthly=bool(keep_monthly),
        compression_level=int(compression_level),
        yearly_filename=str(yearly_filename),
    )


def download_era5land_swvl1(
    *,
    out_dir: str | None = None,
    # Default is a tiny smoke test range; override this explicitly for full runs.
    year_start: int = 2020,
    year_end: int = 2020,
    years: Iterable[int] | None = None,
    time: str = "12:00",
    # CDS bbox: [north, west, south, east] in degrees.
    area: Sequence[float] = (55.0, 70.0, 15.0, 140.0),
    num_workers: int = 1,
    overwrite: bool = False,
    keep_monthly: bool = False,
    compression_level: int = 4,
    yearly_filename: str = "volumetric_soil_water_layer_1.nc",
) -> None:
    """Download ERA5-Land swvl1 for a region and write one NetCDF per year.

    Args:
        out_dir: Root directory (expects per-year subfolders). If None, uses
            `<repo>/data/0.1/1/land_surface`.
        years: Iterable of years to download.
        time: One time per day, e.g. "12:00".
        area: Bounding box (north, west, south, east).
        overwrite: Re-download/overwrite existing monthly and yearly files.
        keep_monthly: If True, keep monthly files after writing the yearly file.
        compression_level: NetCDF zlib compression level (0-9).
        yearly_filename: Final yearly filename inside each `<year>/` directory.
    """

    _require("cdsapi")

    import cdsapi  # type: ignore

    project_root = Path(__file__).resolve().parents[1]
    if out_dir is None:
        out_root = project_root / "data" / "0.1" / "1" / "land_surface"
    else:
        out_root = Path(out_dir).expanduser().resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    area_list = [float(x) for x in area]
    if len(area_list) != 4:
        raise ValueError(
            "Invalid `area`. Expected 4 numbers [north, west, south, east], "
            f"got: {area!r}"
        )

    dataset = "reanalysis-era5-land"
    var_request = "volumetric_soil_water_layer_1"

    if years is None:
        year_start_i = int(year_start)
        year_end_i = int(year_end)
        if year_end_i < year_start_i:
            raise ValueError(f"year_end ({year_end_i}) must be >= year_start ({year_start_i})")
        years_list = list(range(year_start_i, year_end_i + 1))
    else:
        years_list = [int(y) for y in years]
    if not years_list:
        raise ValueError("Empty `years`.")

    print(
        "[era5land] dataset=reanalysis-era5-land\n"
        f"[era5land] variable={var_request}\n"
        f"[era5land] time={time}\n"
        f"[era5land] area(N,W,S,E)={area_list}\n"
        f"[era5land] years={years_list[0]}..{years_list[-1]} (n={len(years_list)})\n"
        f"[era5land] out_root={out_root}\n"
        f"[era5land] num_workers={int(num_workers)}"
    )

    _safe_print_cds_config_hint()

    workers = int(num_workers)
    if workers < 1:
        raise ValueError("num_workers must be >= 1")

    if workers == 1:
        client = cdsapi.Client()
        for year in years_list:
            msg = _download_one_year(
                year=int(year),
                out_root=out_root,
                client=client,
                dataset=dataset,
                var_request=var_request,
                time=time,
                area_list=area_list,
                overwrite=overwrite,
                keep_monthly=keep_monthly,
                compression_level=compression_level,
                yearly_filename=yearly_filename,
            )
            print(msg)
        return

    # Parallel across years. Keep this low (e.g. 2-3) to avoid CDS rate limits.
    area_tuple = (area_list[0], area_list[1], area_list[2], area_list[3])
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_year = {
            ex.submit(
                _download_one_year_worker,
                int(year),
                str(out_root),
                dataset,
                var_request,
                time,
                area_tuple,
                overwrite,
                keep_monthly,
                compression_level,
                yearly_filename,
            ): int(year)
            for year in years_list
        }
        for fut in cf.as_completed(fut_to_year):
            year = fut_to_year[fut]
            try:
                print(fut.result())
            except Exception as e:
                # Fail fast: resume is supported on re-run.
                print(f"[error] year={year} failed: {type(e).__name__}: {e}", file=sys.stderr)
                raise


def main() -> None:
    _require("fire")

    from fire import Fire  # type: ignore

    Fire(download_era5land_swvl1)


if __name__ == "__main__":
    main()
