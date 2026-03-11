#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Download ERA5-Land variables into a LandBench-like on-disk layout (cdsapi).

Goal
----
Keep the *project* structure stable while making the *data* layout clean and inspectable.

This script writes one NetCDF file per variable per year, using the folder names you asked for.
It supports multiple download profiles:

- ``baseline``: original forcing-only dataset used in this repo
- ``hydrology``: baseline + multi-layer soil moisture + core hydrologic fluxes
- ``full``: hydrology + ET component breakdown

Typical ``hydrology`` output:

  data/0.1/1/atmosphere/<year>/2m_temperature.nc
  data/0.1/1/atmosphere/<year>/surface_pressure.nc
  data/0.1/1/atmosphere/<year>/10m_u_component_of_wind.nc
  data/0.1/1/atmosphere/<year>/10m_v_component_of_wind.nc
  data/0.1/1/atmosphere/<year>/precipitation.nc                  (m/hr, from total_precipitation)
  data/0.1/1/atmosphere/<year>/specific_humidity.nc              (kg/kg, derived from d2m + sp)

  data/0.1/1/land_surface/<year>/surface_solar_radiation_downwards_w_m2.nc
  data/0.1/1/land_surface/<year>/surface_thermal_radiation_downwards_w_m2.nc
  data/0.1/1/land_surface/<year>/volumetric_soil_water_layer_1.nc
  data/0.1/1/land_surface/<year>/volumetric_soil_water_layer_2.nc
  data/0.1/1/land_surface/<year>/volumetric_soil_water_layer_3.nc
  data/0.1/1/land_surface/<year>/volumetric_soil_water_layer_4.nc
  data/0.1/1/land_surface/<year>/evapotranspiration.nc
  data/0.1/1/land_surface/<year>/surface_runoff.nc
  data/0.1/1/land_surface/<year>/sub_surface_runoff.nc
  data/0.1/1/land_surface/<year>/runoff.nc
  data/0.1/1/land_surface/<year>/root_zone_soil_moisture_0_100cm.nc
  data/0.1/1/land_surface/<year>/soil_water_storage_0_100cm_mm.nc

Notes
-----
- This is *not* changing model/training code. It's a standalone downloader/preprocessor.
- To keep dataset size manageable, we download ONE fixed time per day (default: 12:00).
  That yields ~365 frames/year, compatible with the existing SWVL1 daily layout.
- ERA5-Land `total_precipitation` (tp) is meters accumulated over the hour in the NetCDF output.
  We store it as a rate in m/hr (numerically identical for a 1-hour accumulation).
- ERA5-Land `ssrd/strd` are J/m^2 accumulated over the hour; we convert to W/m^2 by /3600.
- ERA5-Land `total_evaporation` is negative for evaporation by ECMWF convention.
  For hydrologic interpretability we export `evapotranspiration` with sign flipped,
  so positive values mean upward water loss.

Credentials
-----------
Configure cdsapi via either:
1) ~/.cdsapirc (recommended)
   url: https://cds.climate.copernicus.eu/api
   key: <YOUR-TOKEN>
2) environment variables:
   export CDSAPI_URL="https://cds.climate.copernicus.eu/api"
   export CDSAPI_KEY="<YOUR-TOKEN>"
"""

from __future__ import annotations

import calendar
import concurrent.futures as cf
import argparse
import os
import random
import shutil
import sys
import time as time_module
import zipfile
from dataclasses import dataclass
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
    """Ensure `path` is a readable NetCDF file (handles ZIP-wrapped NetCDF)."""

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

    raise ValueError(f"File is neither NetCDF nor ZIP: {path}")


def _download_month(
    *,
    client,
    dataset: str,
    var_requests: Sequence[str],
    year: int,
    month: int,
    time: str,
    area_list: Sequence[float],
    out_path: Path,
) -> None:
    m = _month_str(month)
    req = {
        "product_type": "reanalysis",
        "variable": list(var_requests),
        "year": str(year),
        "month": m,
        "day": _day_list(year, month),
        "time": time,
        "area": list(area_list),
        "format": "netcdf",
    }

    print(f"[download] year={year} month={m} vars={len(var_requests)} -> {out_path.name}")
    client.retrieve(dataset, req, str(out_path))


def _is_likely_transient_download_error(exc: Exception) -> bool:
    current = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        text = str(current).lower()
        if any(
            token in name or token in text
            for token in (
                "ssl",
                "timeout",
                "connection",
                "chunked",
                "incomplete",
                "temporarily unavailable",
                "max retries exceeded",
                "unexpected eof",
                "badzipfile",
                "corrupted zip",
                "partial download",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _ensure_month_ok_or_redownload(
    *,
    client,
    client_factory=None,
    dataset: str,
    var_requests: Sequence[str],
    year: int,
    month: int,
    time: str,
    area_list: Sequence[float],
    month_path: Path,
    overwrite: bool,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 5.0,
    retry_backoff: float = 2.0,
    retry_jitter_seconds: float = 1.0,
) -> None:
    """Validate or (re)download a monthly file (handles partial/corrupt leftovers)."""

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
            active_client = client_factory() if client_factory is not None else client
            _download_month(
                client=active_client,
                dataset=dataset,
                var_requests=var_requests,
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
            if attempt < int(max_attempts) and _is_likely_transient_download_error(e):
                sleep_s = float(retry_sleep_seconds) * (float(retry_backoff) ** (attempt - 1))
                sleep_s += random.uniform(0.0, float(retry_jitter_seconds))
                print(
                    f"[retry] year={year} month={month:02d} sleeping {sleep_s:.1f}s before retry"
                )
                time_module.sleep(max(0.0, sleep_s))

    raise RuntimeError(
        f"Failed to obtain a valid monthly file after {max_attempts} attempts: {month_path}\n"
        f"Last error: {type(last_err).__name__}: {last_err}"
    )


@dataclass(frozen=True)
class _OutSpec:
    category: str  # "atmosphere" or "land_surface"
    filename: str
    out_var: str


_LAYER_THICKNESS_M = {
    "swvl1": 0.07,
    "swvl2": 0.21,
    "swvl3": 0.72,
    "swvl4": 1.89,
}


# Raw variables requested from CDS for the original forcing-only setup.
_BASELINE_RAW_VARS: Tuple[str, ...] = (
    "2m_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "2m_dewpoint_temperature",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
)

_HYDROLOGY_RAW_VARS: Tuple[str, ...] = (
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "total_evaporation",
    "surface_runoff",
    "sub_surface_runoff",
)

_FULL_EVAP_RAW_VARS: Tuple[str, ...] = (
    "evaporation_from_bare_soil",
    "evaporation_from_open_water_surfaces_excluding_oceans",
    "evaporation_from_the_top_of_canopy",
    "evaporation_from_vegetation_transpiration",
)

# Mapping from internal short names to possible CDS NetCDF var names.
_ALL_RAW_TO_INTERNAL = {
    "t2m": ("t2m", "2m_temperature"),
    "sp": ("sp", "surface_pressure"),
    "u10": ("u10", "10m_u_component_of_wind"),
    "v10": ("v10", "10m_v_component_of_wind"),
    "tp": ("tp", "total_precipitation"),
    "d2m": ("d2m", "2m_dewpoint_temperature"),
    "ssrd": ("ssrd", "surface_solar_radiation_downwards"),
    "strd": ("strd", "surface_thermal_radiation_downwards"),
    "swvl1": ("swvl1", "volumetric_soil_water_layer_1"),
    "swvl2": ("swvl2", "volumetric_soil_water_layer_2"),
    "swvl3": ("swvl3", "volumetric_soil_water_layer_3"),
    "swvl4": ("swvl4", "volumetric_soil_water_layer_4"),
    "e": ("e", "total_evaporation"),
    "sro": ("sro", "surface_runoff"),
    "ssro": ("ssro", "sub_surface_runoff"),
    "evabs": ("evabs", "evaporation_from_bare_soil"),
    "evaow": ("evaow", "evaporation_from_open_water_surfaces_excluding_oceans"),
    "evatc": ("evatc", "evaporation_from_the_top_of_canopy"),
    "evavt": ("evavt", "evaporation_from_vegetation_transpiration"),
}

_BASELINE_OUT_SPECS: Tuple[_OutSpec, ...] = (
    _OutSpec("atmosphere", "2m_temperature.nc", "t2m"),
    _OutSpec("atmosphere", "surface_pressure.nc", "sp"),
    _OutSpec("atmosphere", "10m_u_component_of_wind.nc", "u10"),
    _OutSpec("atmosphere", "10m_v_component_of_wind.nc", "v10"),
    _OutSpec("atmosphere", "precipitation.nc", "precipitation"),
    _OutSpec("atmosphere", "specific_humidity.nc", "specific_humidity"),
    _OutSpec("land_surface", "surface_solar_radiation_downwards_w_m2.nc", "ssrd_w_m2"),
    _OutSpec("land_surface", "surface_thermal_radiation_downwards_w_m2.nc", "strd_w_m2"),
)

_HYDROLOGY_OUT_SPECS: Tuple[_OutSpec, ...] = (
    _OutSpec("land_surface", "volumetric_soil_water_layer_1.nc", "swvl1"),
    _OutSpec("land_surface", "volumetric_soil_water_layer_2.nc", "swvl2"),
    _OutSpec("land_surface", "volumetric_soil_water_layer_3.nc", "swvl3"),
    _OutSpec("land_surface", "volumetric_soil_water_layer_4.nc", "swvl4"),
    _OutSpec("land_surface", "evapotranspiration.nc", "evapotranspiration"),
    _OutSpec("land_surface", "surface_runoff.nc", "surface_runoff"),
    _OutSpec("land_surface", "sub_surface_runoff.nc", "sub_surface_runoff"),
    _OutSpec("land_surface", "runoff.nc", "runoff"),
    _OutSpec("land_surface", "root_zone_soil_moisture_0_100cm.nc", "rzsm_0_100cm"),
    _OutSpec("land_surface", "soil_water_storage_0_100cm_mm.nc", "soil_water_storage_0_100cm_mm"),
)

_FULL_EVAP_OUT_SPECS: Tuple[_OutSpec, ...] = (
    _OutSpec("land_surface", "evaporation_from_bare_soil.nc", "evaporation_from_bare_soil"),
    _OutSpec(
        "land_surface",
        "evaporation_from_open_water_surfaces_excluding_oceans.nc",
        "evaporation_from_open_water_surfaces_excluding_oceans",
    ),
    _OutSpec("land_surface", "evaporation_from_the_top_of_canopy.nc", "evaporation_from_the_top_of_canopy"),
    _OutSpec("land_surface", "evaporation_from_vegetation_transpiration.nc", "evaporation_from_vegetation_transpiration"),
)


def _dedupe_keep_order(items: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _build_download_profile(
    *,
    download_profile: str,
    include_root_zone_derivatives: bool,
) -> tuple[Tuple[str, ...], dict, Tuple[_OutSpec, ...]]:
    profile = str(download_profile).strip().lower()
    if profile not in {"baseline", "hydrology", "full"}:
        raise ValueError(
            f"Invalid download_profile={download_profile!r}. "
            "Choose one of: 'baseline', 'hydrology', 'full'."
        )

    raw_vars = list(_BASELINE_RAW_VARS)
    out_specs = list(_BASELINE_OUT_SPECS)

    if profile in {"hydrology", "full"}:
        raw_vars.extend(_HYDROLOGY_RAW_VARS)
        out_specs.extend(_HYDROLOGY_OUT_SPECS)
        if not include_root_zone_derivatives:
            out_specs = [
                s
                for s in out_specs
                if s.out_var not in {"rzsm_0_100cm", "soil_water_storage_0_100cm_mm"}
            ]

    if profile == "full":
        raw_vars.extend(_FULL_EVAP_RAW_VARS)
        out_specs.extend(_FULL_EVAP_OUT_SPECS)

    raw_vars = _dedupe_keep_order(raw_vars)

    raw_to_internal = {}
    for internal, candidates in _ALL_RAW_TO_INTERNAL.items():
        if candidates[-1] in raw_vars:
            raw_to_internal[internal] = candidates

    return tuple(raw_vars), raw_to_internal, tuple(out_specs)


def _normalize_monthly_ds(ds, raw_to_internal: dict):
    """Normalize coordinate + variable naming for a CDS monthly dataset."""
    _require("xarray")

    # CDS sometimes uses `valid_time` instead of `time`.
    if "time" not in ds.dims and "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # Keep only robust coords (concat across months).
    keep_coords = {"time", "latitude", "longitude"}
    drop_coords = [c for c in ds.coords if c not in keep_coords]
    if drop_coords:
        ds = ds.drop_vars(drop_coords, errors="ignore")

    # Rename variables to internal names.
    rename_map = {}
    for internal, candidates in raw_to_internal.items():
        if internal in ds.data_vars:
            continue
        for cand in candidates:
            if cand in ds.data_vars:
                rename_map[cand] = internal
                break
    if rename_map:
        ds = ds.rename(rename_map)

    missing = [k for k in raw_to_internal.keys() if k not in ds.data_vars]
    if missing:
        raise KeyError(
            f"Monthly dataset missing vars after rename: {missing}. Found: {list(ds.data_vars)}"
        )

    return ds[list(raw_to_internal.keys())]


def _specific_humidity_from_dewpoint_and_pressure(d2m_k, sp_pa):
    """Compute near-surface specific humidity (kg/kg) from dewpoint (K) and pressure (Pa)."""
    import numpy as np

    # Magnus-Tetens approximation using dewpoint in Celsius.
    td_c = d2m_k - 273.15
    e_pa = 611.2 * np.exp((17.67 * td_c) / (td_c + 243.5))

    # q = eps * e / (p - (1-eps) e)
    eps = 0.622
    denom = sp_pa - (1.0 - eps) * e_pa
    return eps * e_pa / denom


def _concat_months_to_year(*, month_paths: Sequence[Path], raw_to_internal: dict):
    _require("xarray")
    import xarray as xr

    opened = []
    dsets = []
    for p in month_paths:
        _ensure_netcdf_inplace(p)
        ds = xr.open_dataset(p)
        opened.append(ds)
        dsets.append(_normalize_monthly_ds(ds, raw_to_internal=raw_to_internal))

    year_ds = xr.concat(
        dsets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    ).sortby("time")

    year_ds.attrs["source"] = "Copernicus Climate Data Store (CDS) - ERA5-Land"
    return year_ds, opened


def _write_single_var_yearly(
    *,
    year_ds,
    out_path: Path,
    out_var: str,
    compression_level: int,
) -> None:
    _require("xarray")
    import xarray as xr

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_var in {"t2m", "sp", "u10", "v10", "swvl1", "swvl2", "swvl3", "swvl4"}:
        da = year_ds[out_var].astype("float32", copy=False)
        out_ds = da.to_dataset(name=out_var)
    elif out_var == "precipitation":
        tp = year_ds["tp"].astype("float32", copy=False)
        out_da = tp.rename("precipitation")
        out_da.attrs = dict(tp.attrs)
        out_da.attrs["units"] = "m hr**-1"
        out_da.attrs["long_name"] = "precipitation_rate"
        out_ds = out_da.to_dataset()
    elif out_var == "evapotranspiration":
        evap = (-year_ds["e"]).astype("float32", copy=False)
        out_da = evap.rename("evapotranspiration")
        out_da.attrs = dict(year_ds["e"].attrs)
        out_da.attrs["units"] = "m hr**-1"
        out_da.attrs["long_name"] = "actual_evapotranspiration"
        out_da.attrs["hydrology_sign_convention"] = "positive_upward_water_loss"
        out_ds = out_da.to_dataset()
    elif out_var == "surface_runoff":
        sro = year_ds["sro"].astype("float32", copy=False)
        out_da = sro.rename("surface_runoff")
        out_da.attrs = dict(year_ds["sro"].attrs)
        out_da.attrs["units"] = "m hr**-1"
        out_da.attrs["long_name"] = "surface_runoff_rate"
        out_ds = out_da.to_dataset()
    elif out_var == "sub_surface_runoff":
        ssro = year_ds["ssro"].astype("float32", copy=False)
        out_da = ssro.rename("sub_surface_runoff")
        out_da.attrs = dict(year_ds["ssro"].attrs)
        out_da.attrs["units"] = "m hr**-1"
        out_da.attrs["long_name"] = "sub_surface_runoff_rate"
        out_ds = out_da.to_dataset()
    elif out_var == "runoff":
        ro = (year_ds["sro"] + year_ds["ssro"]).astype("float32", copy=False)
        out_da = ro.rename("runoff")
        out_da.attrs["units"] = "m hr**-1"
        out_da.attrs["long_name"] = "runoff_rate"
        out_da.attrs["derivation"] = "surface_runoff + sub_surface_runoff"
        out_ds = out_da.to_dataset()
    elif out_var == "specific_humidity":
        q = _specific_humidity_from_dewpoint_and_pressure(year_ds["d2m"], year_ds["sp"]).rename(
            "specific_humidity"
        )
        q = q.astype("float32", copy=False)
        q.attrs["units"] = "kg kg**-1"
        q.attrs["long_name"] = "near_surface_specific_humidity"
        out_ds = q.to_dataset()
    elif out_var == "ssrd_w_m2":
        ssrd = (year_ds["ssrd"] / 3600.0).astype("float32", copy=False)
        out_da = ssrd.rename("ssrd_w_m2")
        out_da.attrs = dict(year_ds["ssrd"].attrs)
        out_da.attrs["units"] = "W m**-2"
        out_da.attrs["long_name"] = "surface_solar_radiation_downwards"
        out_ds = out_da.to_dataset()
    elif out_var == "strd_w_m2":
        strd = (year_ds["strd"] / 3600.0).astype("float32", copy=False)
        out_da = strd.rename("strd_w_m2")
        out_da.attrs = dict(year_ds["strd"].attrs)
        out_da.attrs["units"] = "W m**-2"
        out_da.attrs["long_name"] = "surface_thermal_radiation_downwards"
        out_ds = out_da.to_dataset()
    elif out_var == "rzsm_0_100cm":
        total_depth = _LAYER_THICKNESS_M["swvl1"] + _LAYER_THICKNESS_M["swvl2"] + _LAYER_THICKNESS_M["swvl3"]
        rzsm = (
            year_ds["swvl1"] * _LAYER_THICKNESS_M["swvl1"]
            + year_ds["swvl2"] * _LAYER_THICKNESS_M["swvl2"]
            + year_ds["swvl3"] * _LAYER_THICKNESS_M["swvl3"]
        ) / total_depth
        out_da = rzsm.astype("float32", copy=False).rename("rzsm_0_100cm")
        out_da.attrs["units"] = "m3 m**-3"
        out_da.attrs["long_name"] = "root_zone_soil_moisture_0_100cm"
        out_da.attrs["derivation"] = "thickness-weighted mean of swvl1-3 over 0-100 cm"
        out_ds = out_da.to_dataset()
    elif out_var == "soil_water_storage_0_100cm_mm":
        storage_mm = 1000.0 * (
            year_ds["swvl1"] * _LAYER_THICKNESS_M["swvl1"]
            + year_ds["swvl2"] * _LAYER_THICKNESS_M["swvl2"]
            + year_ds["swvl3"] * _LAYER_THICKNESS_M["swvl3"]
        )
        out_da = storage_mm.astype("float32", copy=False).rename("soil_water_storage_0_100cm_mm")
        out_da.attrs["units"] = "mm"
        out_da.attrs["long_name"] = "soil_water_storage_0_100cm"
        out_da.attrs["derivation"] = "1000 * sum(swvl_i * layer_thickness_m) for layers 1-3"
        out_ds = out_da.to_dataset()
    elif out_var in {
        "evaporation_from_bare_soil",
        "evaporation_from_open_water_surfaces_excluding_oceans",
        "evaporation_from_the_top_of_canopy",
        "evaporation_from_vegetation_transpiration",
    }:
        source_map = {
            "evaporation_from_bare_soil": "evabs",
            "evaporation_from_open_water_surfaces_excluding_oceans": "evaow",
            "evaporation_from_the_top_of_canopy": "evatc",
            "evaporation_from_vegetation_transpiration": "evavt",
        }
        src = source_map[out_var]
        out_da = (-year_ds[src]).astype("float32", copy=False).rename(out_var)
        out_da.attrs = dict(year_ds[src].attrs)
        out_da.attrs["units"] = "m hr**-1"
        out_da.attrs["hydrology_sign_convention"] = "positive_upward_water_loss"
        out_ds = out_da.to_dataset()
    else:
        raise ValueError(f"Unknown out_var: {out_var}")

    encoding = {
        v: {
            "zlib": True,
            "complevel": int(compression_level),
            "shuffle": True,
            "dtype": "float32",
        }
        for v in out_ds.data_vars
    }
    out_ds.to_netcdf(out_path, encoding=encoding)
    try:
        out_ds.close()
    except Exception:
        pass


def _download_one_year(
    *,
    year: int,
    out_root: Path,
    client,
    client_factory=None,
    dataset: str,
    raw_vars: Sequence[str],
    raw_to_internal: dict,
    out_specs: Sequence[_OutSpec],
    time: str,
    area_list: Sequence[float],
    overwrite: bool,
    keep_monthly: bool,
    compression_level: int,
    max_download_attempts: int,
    retry_sleep_seconds: float,
    retry_backoff: float,
    retry_jitter_seconds: float,
) -> str:
    required_paths = [out_root / s.category / str(year) / s.filename for s in out_specs]
    if not overwrite and all(p.exists() for p in required_paths):
        return f"[skip] year={year} all files exist"

    tmp_dir = out_root / "_tmp_landbench_style" / str(year)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    month_paths: list[Path] = []
    for month in range(1, 13):
        m = _month_str(month)
        month_path = tmp_dir / f"forcing_{year}{m}.nc"
        month_paths.append(month_path)
        _ensure_month_ok_or_redownload(
            client=client,
            client_factory=client_factory,
            dataset=dataset,
            var_requests=raw_vars,
            year=year,
            month=month,
            time=time,
            area_list=area_list,
            month_path=month_path,
            overwrite=overwrite,
            max_attempts=max_download_attempts,
            retry_sleep_seconds=retry_sleep_seconds,
            retry_backoff=retry_backoff,
            retry_jitter_seconds=retry_jitter_seconds,
        )

    year_ds, opened = _concat_months_to_year(month_paths=month_paths, raw_to_internal=raw_to_internal)
    try:
        for spec in out_specs:
            out_path = out_root / spec.category / str(year) / spec.filename
            if out_path.exists() and not overwrite:
                print(f"[skip] {out_path} exists")
                continue
            print(f"[write] {out_path}")
            _write_single_var_yearly(
                year_ds=year_ds,
                out_path=out_path,
                out_var=spec.out_var,
                compression_level=compression_level,
            )
    finally:
        try:
            year_ds.close()
        except Exception:
            pass
        for ds in opened:
            try:
                ds.close()
            except Exception:
                pass

    if not keep_monthly:
        for p in month_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return f"[done] year={year}"


def _download_one_year_worker(
    year: int,
    out_root: str,
    dataset: str,
    download_profile: str,
    include_root_zone_derivatives: bool,
    time: str,
    area_list: Tuple[float, float, float, float],
    overwrite: bool,
    keep_monthly: bool,
    compression_level: int,
    max_download_attempts: int,
    retry_sleep_seconds: float,
    retry_backoff: float,
    retry_jitter_seconds: float,
) -> str:
    _require("cdsapi")
    import cdsapi  # type: ignore

    client = cdsapi.Client()
    raw_vars, raw_to_internal, out_specs = _build_download_profile(
        download_profile=download_profile,
        include_root_zone_derivatives=include_root_zone_derivatives,
    )
    return _download_one_year(
        year=int(year),
        out_root=Path(out_root),
        client=client,
        client_factory=cdsapi.Client,
        dataset=str(dataset),
        raw_vars=raw_vars,
        raw_to_internal=raw_to_internal,
        out_specs=out_specs,
        time=str(time),
        area_list=area_list,
        overwrite=bool(overwrite),
        keep_monthly=bool(keep_monthly),
        compression_level=int(compression_level),
        max_download_attempts=int(max_download_attempts),
        retry_sleep_seconds=float(retry_sleep_seconds),
        retry_backoff=float(retry_backoff),
        retry_jitter_seconds=float(retry_jitter_seconds),
    )


def download_era5land_landbench_style(
    *,
    out_dir: str | None = None,
    year_start: int = 2020,
    year_end: int = 2020,
    years: Iterable[int] | None = None,
    download_profile: str = "hydrology",
    include_root_zone_derivatives: bool = True,
    time: str = "12:00",
    # Default bbox: East China (tunable). Format: [north, west, south, east]
    area: Sequence[float] = (42.0, 105.0, 20.0, 125.0),
    num_workers: int = 1,
    overwrite: bool = False,
    keep_monthly: bool = False,
    compression_level: int = 4,
    max_download_attempts: int = 8,
    retry_sleep_seconds: float = 5.0,
    retry_backoff: float = 2.0,
    retry_jitter_seconds: float = 1.0,
) -> None:
    """Download ERA5-Land vars and write LandBench-like yearly files.

    Args:
        download_profile:
            ``baseline`` keeps the original forcing-only dataset.
            ``hydrology`` adds swvl1-4, evapotranspiration, runoff and root-zone derivatives.
            ``full`` further adds ET component breakdown fields.
        include_root_zone_derivatives:
            If ``True``, also export derived 0-100 cm root-zone moisture/storage files.
        max_download_attempts:
            Number of monthly download retries before the script gives up.
        retry_sleep_seconds:
            Initial sleep before retrying a transient monthly download failure.
        retry_backoff:
            Exponential backoff multiplier applied to each retry sleep.
        retry_jitter_seconds:
            Extra random jitter added to retry sleep to avoid retry storms.
    """

    _require("cdsapi")
    _require("xarray")

    import cdsapi  # type: ignore

    project_root = Path(__file__).resolve().parents[1]
    if out_dir is None:
        out_root = project_root / "data" / "0.1" / "1"
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
    raw_vars, raw_to_internal, out_specs = _build_download_profile(
        download_profile=download_profile,
        include_root_zone_derivatives=include_root_zone_derivatives,
    )

    if years is None:
        ys = list(range(int(year_start), int(year_end) + 1))
    else:
        ys = [int(y) for y in years]
    if not ys:
        raise ValueError("Empty `years`.")

    print(
        "[era5land] dataset=reanalysis-era5-land\n"
        f"[era5land] profile={download_profile}\n"
        f"[era5land] variables={list(raw_vars)}\n"
        f"[era5land] outputs={[s.filename for s in out_specs]}\n"
        f"[era5land] time={time}\n"
        f"[era5land] area(N,W,S,E)={area_list}\n"
        f"[era5land] years={ys[0]}..{ys[-1]} (n={len(ys)})\n"
        f"[era5land] out_root={out_root}\n"
        f"[era5land] num_workers={int(num_workers)}\n"
        f"[era5land] max_download_attempts={int(max_download_attempts)}"
    )

    _safe_print_cds_config_hint()

    workers = int(num_workers)
    if workers < 1:
        raise ValueError("num_workers must be >= 1")

    if workers == 1:
        client = cdsapi.Client()
        for y in ys:
            print(
                _download_one_year(
                    year=int(y),
                    out_root=out_root,
                    client=client,
                    client_factory=cdsapi.Client,
                    dataset=dataset,
                    raw_vars=raw_vars,
                    raw_to_internal=raw_to_internal,
                    out_specs=out_specs,
                    time=time,
                    area_list=area_list,
                    overwrite=overwrite,
                    keep_monthly=keep_monthly,
                    compression_level=compression_level,
                    max_download_attempts=max_download_attempts,
                    retry_sleep_seconds=retry_sleep_seconds,
                    retry_backoff=retry_backoff,
                    retry_jitter_seconds=retry_jitter_seconds,
                )
            )
        return

    # Parallel across years. Keep small (e.g. 2-3) to avoid CDS rate limits.
    area_tuple = (area_list[0], area_list[1], area_list[2], area_list[3])
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_year = {
            ex.submit(
                _download_one_year_worker,
                int(y),
                str(out_root),
                dataset,
                str(download_profile),
                bool(include_root_zone_derivatives),
                time,
                area_tuple,
                overwrite,
                keep_monthly,
                compression_level,
                max_download_attempts,
                retry_sleep_seconds,
                retry_backoff,
                retry_jitter_seconds,
            ): int(y)
            for y in ys
        }
        for fut in cf.as_completed(fut_to_year):
            y = fut_to_year[fut]
            try:
                print(fut.result())
            except Exception as e:
                print(f"[error] year={y} failed: {type(e).__name__}: {e}", file=sys.stderr)
                raise


def _parse_bool_flag(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {text!r}")


def _parse_int_list(text: str | None) -> list[int] | None:
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    return [float(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land yearly files in a LandBench-like layout.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--year-start", type=int, default=2020)
    parser.add_argument("--year-end", type=int, default=2020)
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated explicit years, e.g. 1979,1980,1981. Overrides year-start/year-end.",
    )
    parser.add_argument(
        "--download-profile",
        default="hydrology",
        choices=["baseline", "hydrology", "full"],
    )
    parser.add_argument(
        "--include-root-zone-derivatives",
        default="true",
        help="Boolean flag: true/false",
    )
    parser.add_argument("--time", default="12:00")
    parser.add_argument(
        "--area",
        default="42.0,105.0,20.0,125.0",
        help="Comma-separated bbox as north,west,south,east",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", default="false", help="Boolean flag: true/false")
    parser.add_argument("--keep-monthly", default="false", help="Boolean flag: true/false")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--max-download-attempts", type=int, default=8)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--retry-jitter-seconds", type=float, default=1.0)
    args = parser.parse_args()

    years = _parse_int_list(args.years)
    area = _parse_float_list(args.area)

    download_era5land_landbench_style(
        out_dir=args.out_dir,
        year_start=args.year_start,
        year_end=args.year_end,
        years=years,
        download_profile=args.download_profile,
        include_root_zone_derivatives=_parse_bool_flag(args.include_root_zone_derivatives),
        time=args.time,
        area=area,
        num_workers=args.num_workers,
        overwrite=_parse_bool_flag(args.overwrite),
        keep_monthly=_parse_bool_flag(args.keep_monthly),
        compression_level=args.compression_level,
        max_download_attempts=args.max_download_attempts,
        retry_sleep_seconds=args.retry_sleep_seconds,
        retry_backoff=args.retry_backoff,
        retry_jitter_seconds=args.retry_jitter_seconds,
    )


if __name__ == "__main__":
    main()
