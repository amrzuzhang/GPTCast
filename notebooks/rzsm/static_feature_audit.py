from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_STATIC_KEYS = [
    "land_mask",
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "sand_fraction",
    "clay_fraction",
    "silt_fraction",
    "porosity",
    "field_capacity",
    "wilting_point",
    "depth_to_bedrock_m",
    "depth_to_water_table_m",
    "topographic_wetness_index",
]


def _open_single_var(path: Path, expected_var: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    try:
        if expected_var in ds.data_vars:
            da = ds[expected_var].load()
        else:
            data_vars = list(ds.data_vars)
            if len(data_vars) != 1:
                raise ValueError(f"Expected {expected_var!r} or a single data var in {path}, found {data_vars}")
            da = ds[data_vars[0]].load()
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return da


def load_static_maps(static_dir: Path, keys: Sequence[str] | None = None) -> dict[str, xr.DataArray]:
    selected = DEFAULT_STATIC_KEYS if keys is None else list(keys)
    loaded: dict[str, xr.DataArray] = {}
    for key in selected:
        path = Path(static_dir) / f"{key}.nc"
        if not path.exists():
            continue
        loaded[key] = _open_single_var(path, key)
    return loaded


def plot_static_maps(
    static_dir: Path,
    keys: Sequence[str] | None = None,
    *,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    dpi: int = 220,
    savepath: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    maps = load_static_maps(static_dir, keys=keys)
    if not maps:
        raise FileNotFoundError(f"No static maps found in {static_dir}")

    names = list(maps.keys())
    nrows = int(np.ceil(len(names) / ncols))
    if figsize is None:
        figsize = (4.2 * ncols, 3.4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, layout="constrained")
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, key in enumerate(names):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        da = maps[key]
        lat_name = "latitude" if "latitude" in da.coords else "lat"
        lon_name = "longitude" if "longitude" in da.coords else "lon"
        lat = da[lat_name].values
        lon = da[lon_name].values
        arr = np.ma.masked_invalid(da.values.astype(np.float32, copy=False))
        extent = [float(lon[0]), float(lon[-1]), float(lat[-1]), float(lat[0])]
        im = ax.imshow(arr, extent=extent, origin="upper", interpolation="nearest", aspect="auto")
        ax.set_title(key, fontsize=9, fontweight="semibold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig, axes
