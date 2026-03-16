from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import random
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gptcast.models import VAEGANVQ


@dataclass(frozen=True)
class CaseSpec:
    name: str
    t0: datetime
    lat: float
    lon: float


@dataclass(frozen=True)
class SearchConfig:
    clip: tuple[float, float] = (0.0, 0.8)
    norm: tuple[float, float] = (-1.0, 1.0)
    resize: int = 720
    crop: int = 256
    max_mask_frac: float = 0.40
    search_attempts: int = 128
    jitter_px: int = 110
    selection_mode: str = "dynamic_near_roi"
    top_candidates_to_show: int = 5


def resolve_device() -> str:
    if torch.cuda.is_available():
        try:
            _ = torch.cuda.current_device()
            return "cuda"
        except Exception as exc:
            print(f"CUDA probe failed ({exc}); falling back to CPU")
    return "cpu"


def norm_to_phys(
    x_norm: np.ndarray | np.ma.MaskedArray,
    clip: tuple[float, float] = (0.0, 0.8),
    norm: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray | np.ma.MaskedArray:
    cmin, cmax = float(clip[0]), float(clip[1])
    nmin, nmax = float(norm[0]), float(norm[1])
    if isinstance(x_norm, np.ma.MaskedArray):
        data = x_norm.data.astype(np.float32, copy=False)
        mask = x_norm.mask
    else:
        data = np.asarray(x_norm, dtype=np.float32)
        mask = None
    x01 = (data - nmin) / (nmax - nmin + 1e-12)
    x_phys = x01 * (cmax - cmin) + cmin
    x_phys = np.clip(x_phys, cmin, cmax)
    if mask is not None:
        return np.ma.masked_array(x_phys, mask=mask)
    return x_phys


def compressed_percentile(arr: np.ndarray | np.ma.MaskedArray, q: float) -> float:
    values = np.ma.array(arr).compressed()
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def geo_extent_from_axes(lat_axis: np.ndarray, lon_axis: np.ndarray) -> list[float]:
    return [float(lon_axis[0]), float(lon_axis[-1]), float(lat_axis[-1]), float(lat_axis[0])]


def open_year_ds(base_dir: Path, year: int) -> xr.Dataset:
    path = base_dir / str(year) / "root_zone_soil_moisture_0_100cm.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return xr.open_dataset(path)


def sel_time_rzsm(ds: xr.Dataset, dt: datetime):
    t = np.datetime64(dt)
    try:
        return ds["rzsm_0_100cm"].sel(time=t)
    except Exception:
        return ds["rzsm_0_100cm"].sel(time=t, method="nearest")


def clamp(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def score_crop(frames_phys_resized: np.ndarray, mask_resized: np.ndarray, y0: int, x0: int, crop: int) -> dict | None:
    patch = frames_phys_resized[:, y0 : y0 + crop, x0 : x0 + crop]
    patch_mask = mask_resized[y0 : y0 + crop, x0 : x0 + crop]
    if patch.shape[-2:] != (crop, crop):
        return None

    mask_frac = float(patch_mask.mean())
    patch_ma = np.ma.array(patch, mask=np.broadcast_to(patch_mask[None, ...], patch.shape))
    temporal_delta = np.ma.abs(np.diff(patch_ma, axis=0))
    grad_y = np.ma.abs(np.diff(patch_ma, axis=1))
    grad_x = np.ma.abs(np.diff(patch_ma, axis=2))
    spatial_values = np.concatenate([grad_y.compressed(), grad_x.compressed()])

    temporal_p95 = compressed_percentile(temporal_delta, 95)
    spatial_p95 = float(np.percentile(spatial_values, 95)) if spatial_values.size else 0.0
    local_range = compressed_percentile(patch_ma, 95) - compressed_percentile(patch_ma, 5)

    score = 2.0 * temporal_p95 + 1.0 * spatial_p95 + 0.5 * local_range - 3.0 * mask_frac
    return {
        "y0": y0,
        "x0": x0,
        "mask_frac": mask_frac,
        "temporal_p95": temporal_p95,
        "spatial_p95": spatial_p95,
        "local_range": local_range,
        "score": score,
    }


def _prepare_case(case: CaseSpec, base_dir: Path, search: SearchConfig) -> dict:
    seq_len = 7
    ds_y = open_year_ds(base_dir, case.t0.year)

    frames = []
    for step in range(seq_len):
        dt_i = case.t0 + timedelta(days=step)
        da = sel_time_rzsm(ds_y, dt_i)
        frames.append(da.values.astype(np.float32, copy=False))
    frames = np.stack(frames, axis=0)

    lat_vals = ds_y["latitude"].values
    lon_vals = ds_y["longitude"].values
    mask_native = np.isnan(frames[0])

    cmin, cmax = search.clip
    frames = np.nan_to_num(frames, nan=float(cmin))
    frames = np.clip(frames, float(cmin), float(cmax))

    x = torch.from_numpy(frames).unsqueeze(1)
    x = F.interpolate(x, size=(search.resize, search.resize), mode="bilinear", align_corners=False)
    frames_r = x.squeeze(1).numpy()

    m = torch.from_numpy(mask_native.astype(np.float32))[None, None, ...]
    m = F.interpolate(m, size=(search.resize, search.resize), mode="nearest")
    mask_r = m[0, 0].numpy().astype(bool)

    nmin, nmax = search.norm
    frames_n = (frames_r - float(cmin)) / (float(cmax) - float(cmin) + 1e-12)
    frames_n = frames_n * (float(nmax) - float(nmin)) + float(nmin)

    resized_lat_vals = np.interp(np.arange(search.resize), np.linspace(0, search.resize - 1, len(lat_vals)), lat_vals)
    resized_lon_vals = np.interp(np.arange(search.resize), np.linspace(0, search.resize - 1, len(lon_vals)), lon_vals)

    lat_idx = int(np.argmin(np.abs(lat_vals - float(case.lat))))
    lon_idx = int(np.argmin(np.abs(lon_vals - float(case.lon))))
    y_center = int(round(lat_idx / (len(lat_vals) - 1) * (search.resize - 1)))
    x_center = int(round(lon_idx / (len(lon_vals) - 1) * (search.resize - 1)))

    candidate_stats = []
    for k in range(search.search_attempts):
        dy = random.randint(-search.jitter_px, search.jitter_px) if k > 0 else 0
        dx = random.randint(-search.jitter_px, search.jitter_px) if k > 0 else 0
        y0 = clamp(y_center - search.crop // 2 + dy, 0, search.resize - search.crop)
        x0 = clamp(x_center - search.crop // 2 + dx, 0, search.resize - search.crop)
        stat = score_crop(frames_r, mask_r, y0, x0, search.crop)
        if stat is not None:
            candidate_stats.append(stat)

    feasible_candidates = [item for item in candidate_stats if item["mask_frac"] <= search.max_mask_frac]
    sorted_candidates = sorted(feasible_candidates or candidate_stats, key=lambda item: item["score"], reverse=True)
    best = min(candidate_stats, key=lambda item: item["mask_frac"]) if search.selection_mode == "lowest_mask_near_roi" else sorted_candidates[0]

    y0, x0 = best["y0"], best["x0"]
    crop_lat_vals = resized_lat_vals[y0 : y0 + search.crop]
    crop_lon_vals = resized_lon_vals[x0 : x0 + search.crop]
    crop_extent = geo_extent_from_axes(crop_lat_vals, crop_lon_vals)

    patch_n = frames_n[:, y0 : y0 + search.crop, x0 : x0 + search.crop]
    patch_mask = mask_r[y0 : y0 + search.crop, x0 : x0 + search.crop]
    input_patch = np.ma.array(patch_n, mask=np.broadcast_to(patch_mask[None, ...], patch_n.shape))

    return {
        "case": case,
        "input_patch_norm": input_patch,
        "crop_extent": crop_extent,
        "candidate_stats": sorted_candidates,
        "selected_candidate": best,
        "frames_n": frames_n,
        "mask_r": mask_r,
    }


def load_models(ckpt_old: Path, ckpt_new: Path, device: str | None = None) -> tuple[VAEGANVQ, VAEGANVQ, str]:
    resolved_device = resolve_device() if device is None else device
    ae_old = VAEGANVQ.load_from_pretrained(str(ckpt_old), device=resolved_device).to(resolved_device).eval()
    ae_new = VAEGANVQ.load_from_pretrained(str(ckpt_new), device=resolved_device).to(resolved_device).eval()
    return ae_old, ae_new, resolved_device


def reconstruct_pair(
    ae_old: VAEGANVQ,
    ae_new: VAEGANVQ,
    input_patch_norm: np.ma.MaskedArray,
    search: SearchConfig,
) -> dict:
    rec_old = ae_old.reconstruct_swvl1(input_patch_norm, normalized=True)
    rec_new = ae_new.reconstruct_swvl1(input_patch_norm, normalized=True)

    inp_phys = norm_to_phys(input_patch_norm, clip=search.clip, norm=search.norm)
    old_phys = norm_to_phys(rec_old, clip=search.clip, norm=search.norm)
    new_phys = norm_to_phys(rec_new, clip=search.clip, norm=search.norm)

    return {
        "input_phys": inp_phys,
        "old_phys": old_phys,
        "new_phys": new_phys,
        "old_abs_err": np.ma.abs(old_phys - inp_phys),
        "new_abs_err": np.ma.abs(new_phys - inp_phys),
    }


def per_lead_mae_rmse(pred: np.ma.MaskedArray, tgt: np.ma.MaskedArray) -> tuple[np.ndarray, np.ndarray]:
    diff = pred - tgt
    mae = np.ma.mean(np.abs(diff), axis=(1, 2))
    rmse = np.sqrt(np.ma.mean(diff ** 2, axis=(1, 2)))
    return np.asarray(mae, dtype=np.float32), np.asarray(rmse, dtype=np.float32)


def per_lead_abs_p95(pred: np.ma.MaskedArray, tgt: np.ma.MaskedArray) -> np.ndarray:
    diff = np.ma.abs(pred - tgt)
    return np.asarray([compressed_percentile(frame, 95) for frame in diff], dtype=np.float32)


def per_lead_bias(pred: np.ma.MaskedArray, tgt: np.ma.MaskedArray) -> np.ndarray:
    diff = pred - tgt
    return np.asarray([float(np.ma.mean(frame)) for frame in diff], dtype=np.float32)


def per_lead_grad_mae(pred: np.ma.MaskedArray, tgt: np.ma.MaskedArray) -> np.ndarray:
    dx_pred = np.ma.diff(pred, axis=2)
    dy_pred = np.ma.diff(pred, axis=1)
    dx_tgt = np.ma.diff(tgt, axis=2)
    dy_tgt = np.ma.diff(tgt, axis=1)
    grad_err = 0.5 * (
        np.ma.mean(np.abs(dx_pred - dx_tgt), axis=(1, 2))
        + np.ma.mean(np.abs(dy_pred - dy_tgt), axis=(1, 2))
    )
    return np.asarray([float(v) for v in grad_err], dtype=np.float32)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boot.append(float(sample.mean()))
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return lo, hi


def default_cases() -> list[CaseSpec]:
    return [
        CaseSpec(name="North China Plain", t0=datetime(2020, 4, 6, 12, 0), lat=36.5, lon=116.5),
        CaseSpec(name="Middle-Lower Yangtze", t0=datetime(2019, 8, 18, 12, 0), lat=30.5, lon=114.5),
        CaseSpec(name="Southeast Hills", t0=datetime(2020, 10, 8, 12, 0), lat=27.5, lon=118.0),
    ]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def read_scalar_series(run_dir: Path, tag: str) -> pd.DataFrame:
    event_path = next((run_dir / "tensorboard" / "version_0").glob("events.out.tfevents.*"))
    ea = EventAccumulator(str(event_path))
    ea.Reload()
    values = ea.Scalars(tag)
    return pd.DataFrame(
        {
            "step": [event.step for event in values],
            "value": [event.value for event in values],
        }
    )


def plot_stage1_training_audit(
    run_map: dict[str, Path],
    output_dir: Path,
    save_stem: str = "fig2_stage1_training_audit",
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = ["val/rec_loss", "val/p_loss", "val/quant_loss"]
    fig, axes = plt.subplots(1, len(tags), figsize=(14.0, 4.0), dpi=220, layout="constrained")
    summary_rows = []

    for ax, tag in zip(np.atleast_1d(axes), tags):
        for label, run_dir in run_map.items():
            series = read_scalar_series(run_dir, tag)
            ax.plot(series["step"], series["value"], label=label, linewidth=1.6)
            best_idx = int(series["value"].idxmin())
            ax.scatter(
                [float(series.loc[best_idx, "step"])],
                [float(series.loc[best_idx, "value"])],
                s=18,
                zorder=3,
            )
            summary_rows.append(
                {
                    "run_label": label,
                    "run_dir": str(run_dir),
                    "tag": tag,
                    "best_step": float(series.loc[best_idx, "step"]),
                    "best_value": float(series.loc[best_idx, "value"]),
                    "last_step": float(series["step"].iloc[-1]),
                    "last_value": float(series["value"].iloc[-1]),
                }
            )
        ax.set_title(tag.replace("val/", "").replace("_", " "))
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False)

    fig.suptitle("Stage-1 convergence audit (cross-run total loss is intentionally omitted)", fontsize=12)

    png_path = output_dir / f"{save_stem}.png"
    pdf_path = output_dir / f"{save_stem}.pdf"
    csv_path = output_dir / f"{save_stem}.csv"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)
    summary = {
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "runs": list(run_map.keys()),
    }
    with (output_dir / f"{save_stem}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary_df, summary


def make_case_study_figure(
    ckpt_old: Path,
    ckpt_new: Path,
    cases: Iterable[CaseSpec],
    base_dir: Path,
    output_dir: Path,
    search: SearchConfig | None = None,
    device: str | None = None,
    save_stem: str = "fig3_stage1_case_studies",
) -> tuple[pd.DataFrame, dict]:
    search_cfg = SearchConfig() if search is None else search
    output_dir.mkdir(parents=True, exist_ok=True)

    ae_old, ae_new, used_device = load_models(ckpt_old, ckpt_new, device=device)
    rows = []
    prepared_cases = []

    for case in cases:
        prepared = _prepare_case(case, base_dir=base_dir, search=search_cfg)
        rec = reconstruct_pair(ae_old, ae_new, prepared["input_patch_norm"], search=search_cfg)
        mae_old, rmse_old = per_lead_mae_rmse(rec["old_phys"], rec["input_phys"])
        mae_new, rmse_new = per_lead_mae_rmse(rec["new_phys"], rec["input_phys"])
        p95_old = per_lead_abs_p95(rec["old_phys"], rec["input_phys"])
        p95_new = per_lead_abs_p95(rec["new_phys"], rec["input_phys"])
        grad_old = per_lead_grad_mae(rec["old_phys"], rec["input_phys"])
        grad_new = per_lead_grad_mae(rec["new_phys"], rec["input_phys"])
        bias_old = per_lead_bias(rec["old_phys"], rec["input_phys"])
        bias_new = per_lead_bias(rec["new_phys"], rec["input_phys"])
        focus_idx = int(np.argmax(np.abs(rmse_old - rmse_new)))
        frame_titles = [(case.t0 + timedelta(days=i)).strftime("%m-%d") for i in range(rec["input_phys"].shape[0])]

        rows.append(
            {
                "case_name": case.name,
                "date": case.t0.strftime("%Y-%m-%d"),
                "focus_lead": focus_idx + 1,
                "focus_frame": frame_titles[focus_idx],
                "mask_frac": prepared["selected_candidate"]["mask_frac"],
                "score": prepared["selected_candidate"]["score"],
                "old_mae": float(mae_old.mean()),
                "new_mae": float(mae_new.mean()),
                "old_rmse": float(rmse_old.mean()),
                "new_rmse": float(rmse_new.mean()),
                "old_p95": float(p95_old.mean()),
                "new_p95": float(p95_new.mean()),
                "old_grad": float(grad_old.mean()),
                "new_grad": float(grad_new.mean()),
                "old_bias": float(bias_old.mean()),
                "new_bias": float(bias_new.mean()),
                "crop_extent_lon_min": prepared["crop_extent"][0],
                "crop_extent_lon_max": prepared["crop_extent"][1],
                "crop_extent_lat_min": prepared["crop_extent"][2],
                "crop_extent_lat_max": prepared["crop_extent"][3],
            }
        )

        prepared_cases.append(
            {
                **prepared,
                **rec,
                "focus_idx": focus_idx,
                "frame_titles": frame_titles,
                "metrics": {
                    "mae_old": mae_old,
                    "mae_new": mae_new,
                    "rmse_old": rmse_old,
                    "rmse_new": rmse_new,
                    "p95_old": p95_old,
                    "p95_new": p95_new,
                },
            }
        )

    results_df = pd.DataFrame(rows)

    err_vmax = max(
        max(compressed_percentile(case["old_abs_err"], 99), compressed_percentile(case["new_abs_err"], 99))
        for case in prepared_cases
    )
    improve_vmax = max(
        compressed_percentile(np.ma.abs(case["old_abs_err"] - case["new_abs_err"]), 99) for case in prepared_cases
    )

    fig, axes = plt.subplots(len(prepared_cases), 6, figsize=(15.2, 3.8 * len(prepared_cases)), dpi=220, layout="constrained")
    if len(prepared_cases) == 1:
        axes = np.asarray([axes])

    col_titles = [
        "Input",
        "Old recon",
        "New recon",
        "Old abs err",
        "New abs err",
        "Old abs err - New abs err",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight="semibold")

    state_ims = []
    err_ims = []
    diff_ims = []
    for row_idx, case in enumerate(prepared_cases):
        focus_idx = case["focus_idx"]
        focus_title = case["frame_titles"][focus_idx]
        extent = case["crop_extent"]
        panels = [
            (case["input_phys"][focus_idx], "YlGnBu", search_cfg.clip[0], search_cfg.clip[1], "RZSM (m3/m3)"),
            (
                case["old_phys"][focus_idx],
                "YlGnBu",
                search_cfg.clip[0],
                search_cfg.clip[1],
                "RZSM (m3/m3)",
            ),
            (
                case["new_phys"][focus_idx],
                "YlGnBu",
                search_cfg.clip[0],
                search_cfg.clip[1],
                "RZSM (m3/m3)",
            ),
            (case["old_abs_err"][focus_idx], "magma", 0.0, err_vmax, "Absolute error (m3/m3)"),
            (case["new_abs_err"][focus_idx], "magma", 0.0, err_vmax, "Absolute error (m3/m3)"),
            (
                case["old_abs_err"][focus_idx] - case["new_abs_err"][focus_idx],
                "coolwarm",
                -improve_vmax,
                improve_vmax,
                "Improvement (m3/m3)",
            ),
        ]
        for col_idx, (panel, cmap, vmin, vmax, cbar_label) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin="upper", interpolation="nearest", aspect="auto")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.tick_params(labelsize=7)
            if col_idx == 0:
                ax.text(
                    -0.30,
                    0.5,
                    f"{case['case'].name}\n{case['case'].t0:%Y-%m-%d}\nlead {focus_idx + 1} ({focus_title})",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    fontweight="semibold",
                )
            if col_idx < 3:
                state_ims.append((im, cbar_label))
            elif col_idx < 5:
                err_ims.append((im, cbar_label))
            else:
                diff_ims.append((im, cbar_label))

    fig.colorbar(state_ims[0][0], ax=axes[:, 0:3], fraction=0.020, pad=0.02, label=state_ims[0][1])
    fig.colorbar(err_ims[0][0], ax=axes[:, 3:5], fraction=0.020, pad=0.02, label=err_ims[0][1])
    fig.colorbar(diff_ims[0][0], ax=axes[:, 5], fraction=0.035, pad=0.02, label=diff_ims[0][1])
    fig.suptitle("Stage-1 tokenizer case studies with georeferenced focus leads", fontsize=12)

    png_path = output_dir / f"{save_stem}.png"
    pdf_path = output_dir / f"{save_stem}.pdf"
    csv_path = output_dir / f"{save_stem}.csv"
    results_df.to_csv(csv_path, index=False)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "device": used_device,
        "cases": len(prepared_cases),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
    }
    with (output_dir / f"{save_stem}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return results_df, summary


def make_benchmark_figure(
    ckpt_old: Path,
    ckpt_new: Path,
    cases: Iterable[CaseSpec],
    base_dir: Path,
    output_dir: Path,
    benchmark_crops_per_case: int = 6,
    search: SearchConfig | None = None,
    device: str | None = None,
    save_stem: str = "fig4_stage1_benchmark",
) -> tuple[pd.DataFrame, dict]:
    search_cfg = SearchConfig() if search is None else search
    output_dir.mkdir(parents=True, exist_ok=True)

    ae_old, ae_new, used_device = load_models(ckpt_old, ckpt_new, device=device)
    rows = []
    for case in cases:
        prepared = _prepare_case(case, base_dir=base_dir, search=search_cfg)
        benchmark_candidates = prepared["candidate_stats"][: min(benchmark_crops_per_case, len(prepared["candidate_stats"]))]
        for rank, cand in enumerate(benchmark_candidates, start=1):
            y0, x0 = cand["y0"], cand["x0"]
            cand_patch_n = prepared["frames_n"][:, y0 : y0 + search_cfg.crop, x0 : x0 + search_cfg.crop]
            cand_patch_mask = prepared["mask_r"][y0 : y0 + search_cfg.crop, x0 : x0 + search_cfg.crop]
            input_patch = np.ma.array(
                cand_patch_n,
                mask=np.broadcast_to(cand_patch_mask[None, ...], cand_patch_n.shape),
            )

            rec = reconstruct_pair(ae_old, ae_new, input_patch, search=search_cfg)
            mae_old, rmse_old = per_lead_mae_rmse(rec["old_phys"], rec["input_phys"])
            mae_new, rmse_new = per_lead_mae_rmse(rec["new_phys"], rec["input_phys"])
            p95_old = per_lead_abs_p95(rec["old_phys"], rec["input_phys"])
            p95_new = per_lead_abs_p95(rec["new_phys"], rec["input_phys"])
            grad_old = per_lead_grad_mae(rec["old_phys"], rec["input_phys"])
            grad_new = per_lead_grad_mae(rec["new_phys"], rec["input_phys"])

            rows.append(
                {
                    "case_name": case.name,
                    "date": case.t0.strftime("%Y-%m-%d"),
                    "rank": rank,
                    "score": cand["score"],
                    "mask_frac": cand["mask_frac"],
                    "temporal_p95": cand["temporal_p95"],
                    "spatial_p95": cand["spatial_p95"],
                    "old_mae": float(mae_old.mean()),
                    "new_mae": float(mae_new.mean()),
                    "old_rmse": float(rmse_old.mean()),
                    "new_rmse": float(rmse_new.mean()),
                    "old_p95": float(p95_old.mean()),
                    "new_p95": float(p95_new.mean()),
                    "old_grad": float(grad_old.mean()),
                    "new_grad": float(grad_new.mean()),
                }
            )

    results_df = pd.DataFrame(rows)
    results_df["delta_rmse"] = results_df["old_rmse"] - results_df["new_rmse"]
    results_df["delta_mae"] = results_df["old_mae"] - results_df["new_mae"]
    results_df["delta_p95"] = results_df["old_p95"] - results_df["new_p95"]
    results_df["delta_grad"] = results_df["old_grad"] - results_df["new_grad"]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), dpi=220, layout="constrained")

    metrics = [
        ("old_rmse", "new_rmse", "RMSE", axes[0, 0]),
        ("old_grad", "new_grad", "grad-MAE", axes[0, 1]),
    ]
    for old_col, new_col, label, ax in metrics:
        x = results_df[old_col].to_numpy(dtype=np.float32)
        y = results_df[new_col].to_numpy(dtype=np.float32)
        colors = results_df["score"].to_numpy(dtype=np.float32)
        scatter = ax.scatter(x, y, c=colors, cmap="viridis", s=42, edgecolor="black", linewidth=0.25)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
        ax.set_xlabel(f"Old mean {label} (m3/m3)")
        ax.set_ylabel(f"New mean {label} (m3/m3)")
        ax.set_title(f"Paired {label} comparison")
        fig.colorbar(scatter, ax=ax, label="Crop score")

    delta_data = [
        results_df["delta_rmse"].to_numpy(dtype=np.float32),
        results_df["delta_mae"].to_numpy(dtype=np.float32),
        results_df["delta_p95"].to_numpy(dtype=np.float32),
        results_df["delta_grad"].to_numpy(dtype=np.float32),
    ]
    delta_labels = ["RMSE", "MAE", "p95", "grad-MAE"]
    axes[1, 0].boxplot(delta_data, labels=delta_labels, showmeans=True)
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_ylabel("Old - New (m3/m3)")
    axes[1, 0].set_title("Distribution of improvement deltas")

    win_rates = [
        float(np.mean(results_df["new_rmse"] < results_df["old_rmse"])),
        float(np.mean(results_df["new_mae"] < results_df["old_mae"])),
        float(np.mean(results_df["new_p95"] < results_df["old_p95"])),
        float(np.mean(results_df["new_grad"] < results_df["old_grad"])),
    ]
    axes[1, 1].bar(delta_labels, win_rates, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("Fraction of crops where new < old")
    axes[1, 1].set_title("Win rates across benchmark crops")

    png_path = output_dir / f"{save_stem}.png"
    pdf_path = output_dir / f"{save_stem}.pdf"
    csv_path = output_dir / f"{save_stem}.csv"
    results_df.to_csv(csv_path, index=False)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "device": used_device,
        "num_cases": int(results_df["case_name"].nunique()),
        "num_crops": int(len(results_df)),
        "mean_old_rmse": float(results_df["old_rmse"].mean()),
        "mean_new_rmse": float(results_df["new_rmse"].mean()),
        "mean_old_mae": float(results_df["old_mae"].mean()),
        "mean_new_mae": float(results_df["new_mae"].mean()),
        "mean_old_p95": float(results_df["old_p95"].mean()),
        "mean_new_p95": float(results_df["new_p95"].mean()),
        "mean_old_grad": float(results_df["old_grad"].mean()),
        "mean_new_grad": float(results_df["new_grad"].mean()),
        "mean_delta_rmse": float(results_df["delta_rmse"].mean()),
        "mean_delta_mae": float(results_df["delta_mae"].mean()),
        "mean_delta_p95": float(results_df["delta_p95"].mean()),
        "mean_delta_grad": float(results_df["delta_grad"].mean()),
        "delta_rmse_ci95": bootstrap_mean_ci(results_df["delta_rmse"].to_numpy(dtype=np.float32)),
        "delta_mae_ci95": bootstrap_mean_ci(results_df["delta_mae"].to_numpy(dtype=np.float32)),
        "delta_p95_ci95": bootstrap_mean_ci(results_df["delta_p95"].to_numpy(dtype=np.float32)),
        "delta_grad_ci95": bootstrap_mean_ci(results_df["delta_grad"].to_numpy(dtype=np.float32)),
        "rmse_win_rate": float(np.mean(results_df["new_rmse"] < results_df["old_rmse"])),
        "mae_win_rate": float(np.mean(results_df["new_mae"] < results_df["old_mae"])),
        "p95_win_rate": float(np.mean(results_df["new_p95"] < results_df["old_p95"])),
        "grad_win_rate": float(np.mean(results_df["new_grad"] < results_df["old_grad"])),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
    }
    with (output_dir / f"{save_stem}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return results_df, summary
