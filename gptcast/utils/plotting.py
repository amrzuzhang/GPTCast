import math
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    # Optional dependency used by the original MIARAD/rainfall notebooks.
    from pysteps.visualization import plot_precip_field as plot_field  # type: ignore
except Exception:  # pragma: no cover
    plot_field = None


ERA5LAND_CMAP = "YlGnBu"
ERA5LAND_BAD_COLOR = "#d9d9d9"
ERA5LAND_CBAR_LABEL = "SWVL1 (m³/m³)"


def _get_era5land_cmap(cmap: str = ERA5LAND_CMAP):
    cm = plt.get_cmap(cmap)
    if hasattr(cm, "copy"):
        cm = cm.copy()
    cm.set_bad(color=ERA5LAND_BAD_COLOR)
    return cm


def _finalize_figure(fig, savepath: Optional[Union[str, Path]] = None):
    fig.patch.set_facecolor("white")
    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    backend = str(plt.get_backend()).lower()
    if "agg" in backend:
        plt.close(fig)
    else:
        plt.show()
    return fig

def plot_miarad(arr: Union[np.ndarray, np.ma.MaskedArray], colorbar=False, colorscale='STEPS-BE', use_utm_projection: bool = True, dpi: int = 200, title: str = '', figsize=None):
    if plot_field is None:
        raise ImportError("pysteps is required for plot_miarad(). Install pysteps or use plot_era5land().")
    assert arr.ndim in [2, 3], "Array must be 2D or 3D (time, height, width)"
    plt.rcParams.update({'font.size': 6, 'font.weight': 'normal'})

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
        steps = 1
    else:
        steps = arr.shape[0]

    # add 20% width if colorbar is True
    x_size = 1.2 if colorbar else 1
    if use_utm_projection:
        #utm 32632
        geodata = {
            "projection": "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs +type=crs",
            "x1": 458995.938,
            "y1": 4804873.924,
            "x2": 841529.075,
            "y2": 5103289.827,
            "yorigin":"lower"
        }
        # bbox in (lower left x, lower left y ,upper right x, upper right y)
        bbox = (geodata['x1']+32000, geodata['y1']+15000, geodata['x2'], geodata['y2']-10000)
        figsize = ((bbox[2]-bbox[0])/100000*x_size, (bbox[3]-bbox[1])/100000) if figsize is None else figsize
    else:
        geodata = {
            "projection": "+proj=longlat +datum=WGS84 +no_defs +type=crs",
            "x1": 8.4936750,
            "y1": 43.3955015,
            "x2": 13.2121251,
            "y2": 46.0054994,
            "yorigin":"lower"
        }
        bbox = (geodata['x1']+0.40, geodata['y1']+0.15, geodata['x2'], geodata['y2']-0.1)
        figsize=((bbox[2]-bbox[0]), bbox[3]-bbox[1]) if figsize is None else figsize

    for i in range(steps):
        titl = f"{title} +{i+1} steps" if steps > 1 else title
        figure, axis = plt.subplots(1, 1, layout="constrained", figsize=figsize, dpi=dpi)
        axis.axis('off')
        axis = plot_field(arr[i], title=titl, bbox=bbox, ax=axis, colorbar=colorbar, colorscale=colorscale, geodata=geodata,  axis='off', map_kwargs=dict(scale="10m", lw=1, drawlonlatlines=True))
        figure.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
        plt.show()
        plt.tight_layout()


def plot_mutiple(arr: Union[np.ndarray, np.ma.MaskedArray], colorbar=False, colorscale='STEPS-BE', dpi: int = 200, title: str = '', figsize=None):
    if plot_field is None:
        raise ImportError("pysteps is required for plot_mutiple(). Install pysteps or use plot_mutiple_era5land().")
    assert arr.ndim == 3, "Array must be 3D (time, height, width)"
    figure, axis = plt.subplots(1, 1, layout="constrained", figsize=figsize, dpi=dpi)
    axis.axis('off')
    axis = plot_field(np.flipud(np.ma.concatenate(arr, axis=1)), title=title, ax=axis, colorbar=colorbar, colorscale=colorscale, axis='off')
    figure.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
    plt.show()
    plt.tight_layout()


def plot_era5land(
    arr: Union[np.ndarray, np.ma.MaskedArray],
    colorbar: bool = False,
    cmap: str = ERA5LAND_CMAP,
    vmin: float = 0.0,
    vmax: float = 0.8,
    dpi: int = 200,
    title: str = "",
    figsize=None,
    frame_titles: Optional[Sequence[str]] = None,
    cbar_label: Optional[str] = ERA5LAND_CBAR_LABEL,
    ncols: Optional[int] = None,
    savepath: Optional[Union[str, Path]] = None,
):
    """Notebook-friendly plotting for ERA5-Land patches.

    For 2D inputs, render a single frame. For 3D inputs, render a compact grid
    with a shared colorbar, which is a better fit for notebook comparisons.
    """
    assert arr.ndim in [2, 3], "Array must be 2D or 3D (time, height, width)"
    if arr.ndim == 3:
        return plot_era5land_grid(
            arr,
            title=title,
            frame_titles=frame_titles,
            ncols=ncols,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=colorbar,
            cbar_label=cbar_label,
            dpi=dpi,
            figsize=figsize,
            savepath=savepath,
        )

    frame = _ensure_era5land_frames(arr)[0]
    if figsize is None:
        figsize = (4.2, 4.2)

    plt.rcParams.update({"font.size": 8, "font.weight": "normal"})
    cm = _get_era5land_cmap(cmap)

    fig, axis = plt.subplots(1, 1, figsize=figsize, dpi=dpi, layout="constrained")
    axis.axis("off")
    im = axis.imshow(frame, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    if title:
        axis.set_title(title)
    if colorbar:
        cbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.03)
        if cbar_label:
            cbar.set_label(cbar_label)

    _finalize_figure(fig, savepath=savepath)
    return fig, axis


def plot_mutiple_era5land(
    arr: Union[np.ndarray, np.ma.MaskedArray],
    colorbar: bool = False,
    cmap: str = ERA5LAND_CMAP,
    vmin: float = 0.0,
    vmax: float = 0.8,
    dpi: int = 200,
    title: str = "",
    figsize=None,
    frame_titles: Optional[Sequence[str]] = None,
    cbar_label: Optional[str] = ERA5LAND_CBAR_LABEL,
    savepath: Optional[Union[str, Path]] = None,
):
    """Plot a time sequence as a single horizontal strip with frame separators."""
    frames = _ensure_era5land_frames(arr)
    steps, _, width = frames.shape

    if frame_titles is not None and len(frame_titles) != steps:
        raise ValueError(f"frame_titles must have length {steps}, got {len(frame_titles)}")

    if figsize is None:
        base_w = 1.8 * steps + (0.8 if colorbar else 0.0)
        figsize = (base_w, 2.5)

    plt.rcParams.update({"font.size": 8, "font.weight": "normal"})
    fig, axis = plt.subplots(1, 1, figsize=figsize, dpi=dpi, layout="constrained")
    axis.set_yticks([])

    cm = _get_era5land_cmap(cmap)
    concat = np.ma.concatenate(frames, axis=1)
    im = axis.imshow(concat, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")

    boundaries = [idx * width - 0.5 for idx in range(1, steps)]
    for xpos in boundaries:
        axis.axvline(xpos, color="white", linewidth=0.8, alpha=0.8)

    if frame_titles is not None:
        centers = [width * idx + (width - 1) / 2 for idx in range(steps)]
        axis.set_xticks(centers)
        axis.set_xticklabels(frame_titles)
        axis.tick_params(axis="x", length=0, pad=2)
    else:
        axis.set_xticks([])

    if title:
        axis.set_title(title)
    if colorbar:
        cbar = fig.colorbar(im, ax=axis, fraction=0.04, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label)

    _finalize_figure(fig, savepath=savepath)
    return fig, axis


def _ensure_era5land_frames(arr: Union[np.ndarray, np.ma.MaskedArray]) -> np.ma.MaskedArray:
    """Normalize an input array into a masked (T,H,W) array for plotting."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError("Array must be 2D or 3D (time, height, width)")
    if isinstance(arr, np.ma.MaskedArray):
        return arr
    return np.ma.masked_invalid(np.asarray(arr))


def plot_era5land_grid(
    arr: Union[np.ndarray, np.ma.MaskedArray],
    *,
    title: str = "",
    frame_titles: Optional[Sequence[str]] = None,
    ncols: Optional[int] = None,
    cmap: str = ERA5LAND_CMAP,
    vmin: float = 0.0,
    vmax: float = 0.8,
    colorbar: bool = True,
    cbar_label: Optional[str] = ERA5LAND_CBAR_LABEL,
    dpi: int = 200,
    figsize=None,
    savepath: Optional[Union[str, Path]] = None,
):
    """Paper-style grid plot for ERA5-Land sequences.

    Compared to `plot_era5land(...)`, this renders all frames in a single figure with a shared
    colorbar and optional per-frame titles.
    """
    frames = _ensure_era5land_frames(arr)
    steps = int(frames.shape[0])

    if frame_titles is not None and len(frame_titles) != steps:
        raise ValueError(f"frame_titles must have length {steps}, got {len(frame_titles)}")

    if ncols is None:
        ncols = min(steps, 7)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(steps / ncols))

    if figsize is None:
        figsize = (2.15 * ncols + (0.6 if colorbar else 0.0), 2.35 * nrows + (0.35 if title else 0.0))

    plt.rcParams.update({"font.size": 8, "font.weight": "normal"})
    cm = _get_era5land_cmap(cmap)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, layout="constrained")
    axes_arr = np.atleast_1d(axes).reshape(-1)

    last_im = None
    for i, ax in enumerate(axes_arr):
        if i >= steps:
            ax.set_visible(False)
            continue
        ax.axis("off")
        last_im = ax.imshow(frames[i], cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
        if frame_titles is not None:
            ax.set_title(frame_titles[i])
        elif steps > 1:
            ax.set_title(f"+{i + 1}")

    if title:
        fig.suptitle(title)

    if colorbar and last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes_arr.tolist(), fraction=0.03, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label)

    _finalize_figure(fig, savepath=savepath)
    return fig, axes


def plot_era5land_panel(
    sequences: Sequence[Union[np.ndarray, np.ma.MaskedArray]],
    *,
    row_titles: Optional[Sequence[str]] = None,
    frame_titles: Optional[Sequence[str]] = None,
    title: str = "",
    cmap: str = ERA5LAND_CMAP,
    vmin: float = 0.0,
    vmax: float = 0.8,
    colorbar: bool = True,
    cbar_label: Optional[str] = ERA5LAND_CBAR_LABEL,
    dpi: int = 200,
    figsize=None,
    savepath: Optional[Union[str, Path]] = None,
):
    """Render multiple SWVL1 sequences as a compact comparison panel.

    Each sequence becomes one row, and each timestep becomes one column.
    This is useful for notebook comparisons such as input vs reconstruction
    or target vs persistence vs forecast.
    """
    if len(sequences) == 0:
        raise ValueError("sequences must not be empty")

    frames_list = [_ensure_era5land_frames(seq) for seq in sequences]
    steps = int(frames_list[0].shape[0])
    if any(int(frames.shape[0]) != steps for frames in frames_list):
        raise ValueError("All sequences must have the same number of frames")

    if row_titles is not None and len(row_titles) != len(frames_list):
        raise ValueError(f"row_titles must have length {len(frames_list)}, got {len(row_titles)}")
    if frame_titles is not None and len(frame_titles) != steps:
        raise ValueError(f"frame_titles must have length {steps}, got {len(frame_titles)}")

    nrows = len(frames_list)
    ncols = steps

    if figsize is None:
        figsize = (2.1 * ncols + (0.7 if colorbar else 0.0), 2.15 * nrows + (0.4 if title else 0.0))

    plt.rcParams.update({"font.size": 8, "font.weight": "normal"})
    cm = _get_era5land_cmap(cmap)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, layout="constrained", squeeze=False)
    last_im = None

    for row_idx, frames in enumerate(frames_list):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            last_im = ax.imshow(frames[col_idx], cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
            if row_idx == 0:
                if frame_titles is not None:
                    ax.set_title(frame_titles[col_idx])
                elif ncols > 1:
                    ax.set_title(f"+{col_idx + 1}")
            if row_titles is not None and col_idx == 0:
                ax.text(
                    -0.04,
                    0.5,
                    row_titles[row_idx],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="semibold",
                )

    if title:
        fig.suptitle(title)

    if colorbar and last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label)

    _finalize_figure(fig, savepath=savepath)
    return fig, axes
