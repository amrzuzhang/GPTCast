from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import xarray as xr

from gptcast.data.guidance_ecmwf import ECMWF_GUIDANCE_VARIABLES


def _require(module_name: str) -> None:
    try:
        __import__(module_name)
    except Exception as e:  # pragma: no cover - defensive
        if module_name == "ecmwfapi":
            install_hint = (
                "Missing dependency: ecmwfapi\n"
                "Install the official ECMWF Web API client package:\n"
                "  pip install ecmwf-api-client\n"
                "or\n"
                "  conda install -c conda-forge ecmwf-api-client\n"
                "Note: this environment currently appears unable to resolve external package indexes.\n"
                "If internet/DNS is blocked, install from a pre-downloaded wheel or use preprocessed guidance files.\n"
            )
        else:
            install_hint = (
                f"Missing dependency: {module_name}\n"
                f"Install one of:\n"
                f"  pip install {module_name}\n"
                f"  conda install -c conda-forge {module_name}\n"
            )
        raise SystemExit(
            install_hint
            + "\n"
            f"Original error: {type(e).__name__}: {e}"
        )


def _format_area(area: Sequence[float]) -> str:
    return "/".join(str(float(x)) for x in area)


def _step_list(lead_days: Sequence[int]) -> str:
    return "/".join(str(int(day) * 24) for day in lead_days)


def _guidance_file_path(out_root: Path, key: str, init_dt: datetime) -> Path:
    return out_root / key / f"{init_dt.year:04d}" / f"{init_dt:%Y%m%d}.nc"


def _normalize_downloaded_netcdf(
    raw_path: Path,
    out_path: Path,
    *,
    out_var: str,
    lead_days: Sequence[int],
) -> None:
    ds = xr.open_dataset(raw_path)
    try:
        coord_rename = {}
        if "longitude" not in ds.coords and "lon" in ds.coords:
            coord_rename["lon"] = "longitude"
        if "latitude" not in ds.coords and "lat" in ds.coords:
            coord_rename["lat"] = "latitude"
        if coord_rename:
            ds = ds.rename(coord_rename)

        var_name = next(iter(ds.data_vars), None)
        if var_name is None:
            raise RuntimeError(f"No data variable found in downloaded ECMWF file: {raw_path}")

        da = ds[var_name]
        if "time" in da.dims and da.sizes.get("time", 1) == 1:
            da = da.isel(time=0, drop=True)
        if "step" in da.coords:
            step_vals = da["step"].values
            if len(step_vals) > 0 and hasattr(step_vals[0], "astype"):
                lead_coord = [int(val.astype("timedelta64[h]").astype(int) // 24) for val in step_vals]
            else:
                lead_coord = list(lead_days)
            da = da.assign_coords(step=lead_coord).rename(step="lead_day")
        elif "lead_day" not in da.dims:
            da = da.expand_dims(lead_day=list(lead_days))

        out_ds = da.astype("float32", copy=False).to_dataset(name=out_var)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_netcdf(out_path)
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _retrieve_ecmwf(
    *,
    init_dt: datetime,
    variable_short_name: str,
    lead_days: Sequence[int],
    area: Sequence[float],
    grid: str,
    target_path: Path,
) -> None:
    _require("ecmwfapi")
    from ecmwfapi import ECMWFDataServer  # type: ignore

    request = {
        "class": "s2",
        "dataset": "s2s",
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": "ecmf",
        "param": variable_short_name,
        "step": _step_list(lead_days),
        "stream": "enfo",
        "time": "00:00:00",
        "type": "cf",
        "date": init_dt.strftime("%Y-%m-%d"),
        "area": _format_area(area),
        "grid": grid,
        "format": "netcdf",
        "target": str(target_path),
    }

    server = ECMWFDataServer()
    server.retrieve(request)


def ensure_ecmwf_guidance(
    *,
    variables: Sequence[str],
    init_dates: Sequence[datetime],
    out_root: Path,
    lead_days: Sequence[int],
    area: Sequence[float],
    grid: str,
    overwrite: bool,
) -> list[Path]:
    created: list[Path] = []
    for init_dt in init_dates:
        for key in variables:
            if key not in ECMWF_GUIDANCE_VARIABLES:
                raise KeyError(f"Unknown ECMWF guidance key: {key!r}")
            spec = ECMWF_GUIDANCE_VARIABLES[key]
            out_path = _guidance_file_path(out_root, key, init_dt)
            if out_path.exists() and not overwrite:
                continue

            with tempfile.TemporaryDirectory(prefix="ecmwf_guidance_") as tmpdir:
                tmp_path = Path(tmpdir) / f"{key}_{init_dt:%Y%m%d}.nc"
                print(f"[download] {key} init={init_dt:%Y-%m-%d} -> {out_path}")
                _retrieve_ecmwf(
                    init_dt=init_dt,
                    variable_short_name=spec.short_name,
                    lead_days=lead_days,
                    area=area,
                    grid=grid,
                    target_path=tmp_path,
                )
                _normalize_downloaded_netcdf(
                    tmp_path,
                    out_path,
                    out_var=spec.out_var,
                    lead_days=lead_days,
                )
            created.append(out_path)
    return created
