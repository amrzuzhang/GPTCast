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


PUBLICATION_RCPARAMS = {
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
}

UNIT_LABEL = "m³ m⁻³"

CAPTION_TABLE1_EN = (
    "Table 1. Unified stage-1 comparison between the MAE-based and PHuber-based RZSM tokenizers. "
    "Metrics are computed in physical space over benchmark crops drawn from multiple East China case studies. "
    "For each metric, we report the mean performance of the old and new tokenizers, the old-minus-new delta, "
    "a bootstrap 95 % confidence interval for the delta, and the fraction of crops where the PHuber tokenizer performs better."
)
CAPTION_TABLE1_ZH = (
    "表1. 第一阶段 RZSM tokenizer 的统一定量比较。表中在统一物理空间评估口径下，对 MAE tokenizer 与 PHuber tokenizer 在多个华东代表性案例的 benchmark crop 上的表现进行汇总，"
    "包括 old/new 的平均指标、old-minus-new 差值、bootstrap 95% 置信区间，以及 PHuber tokenizer 优于 MAE tokenizer 的样本比例。"
)

CAPTION_FIG2_EN = (
    "Figure 2. Quantitative comparison of the stage-1 RZSM tokenizers. "
    "Panels (a) and (b) show paired crop-level RMSE and MAE for the MAE-based tokenizer versus the PHuber-based tokenizer; "
    "points below the diagonal favor the PHuber tokenizer. Panel (c) summarizes the fraction of benchmark crops for which the PHuber tokenizer "
    "outperforms the MAE tokenizer across four evaluation metrics. The corresponding mean old-minus-new improvements are reported in Table 1."
)
CAPTION_FIG2_ZH = (
    "图2. 第一阶段 RZSM tokenizer 的定量对比。子图 (a) 和 (b) 分别给出 crop 级别的 RMSE 和 MAE 配对散点图，"
    "横轴为 MAE tokenizer，纵轴为 PHuber tokenizer，落在对角线下方的点表示 PHuber tokenizer 更优。子图 (c) 给出 PHuber tokenizer 在四个评估指标上优于 MAE tokenizer 的样本比例；"
    "对应的平均 old-minus-new 改进值见表1。"
)

CAPTION_FIG3_EN = (
    "Figure 3. Representative georeferenced stage-1 reconstruction cases for the RZSM tokenizers. "
    "The two cases shown here correspond to benchmark crops with the largest RMSE improvements in favor of the PHuber tokenizer. "
    "For each case, the figure compares the input field, reconstructions from the MAE-based and PHuber-based tokenizers, "
    "their absolute-error maps, and the error-reduction map. Positive values in the reduction panels indicate locations where the PHuber tokenizer reduces reconstruction error relative to the MAE tokenizer."
)
CAPTION_FIG3_ZH = (
    "图3. 第一阶段 RZSM tokenizer 的代表性地理配准重建案例图。图中展示的是在 benchmark crop 中相对于 MAE tokenizer 具有最大 RMSE 改善的两个代表案例。对于每个案例，图中对比了输入场、MAE tokenizer 重建、PHuber tokenizer 重建、"
    "对应的绝对误差图，以及误差降低图。降低图中的正值表示 PHuber tokenizer 相比 MAE tokenizer 降低了该位置的重建误差。"
)

CAPTION_SUPP_TRAINING_EN = (
    "Supplementary Figure S1. Validation-side training audit for the stage-1 tokenizer runs. "
    "The curves show reconstruction, perceptual, and quantization losses for the MAE-based and PHuber-based tokenizers. "
    "These plots are used to document training stability rather than to define the final model selection criterion."
)
CAPTION_SUPP_TRAINING_ZH = (
    "补充图 S1. 第一阶段 tokenizer 训练过程审计图。曲线展示了 MAE tokenizer 与 PHuber tokenizer 在验证集上的重建误差、感知误差和量化误差。"
    "该图用于说明训练稳定性，不作为最终模型优劣判断的唯一依据。"
)


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


def _panel_label(ax, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
    )


def _write_caption_files(output_dir: Path, save_stem: str, caption_en: str, caption_zh: str | None = None) -> dict[str, str]:
    txt_path = output_dir / f"{save_stem}.caption.txt"
    md_path = output_dir / f"{save_stem}.caption.md"
    txt_path.write_text(caption_en + "\n")
    md_path.write_text(caption_en + "\n")
    payload = {"caption_txt_path": str(txt_path), "caption_md_path": str(md_path)}
    if caption_zh is not None:
        zh_txt_path = output_dir / f"{save_stem}.caption.zh.txt"
        zh_md_path = output_dir / f"{save_stem}.caption.zh.md"
        zh_txt_path.write_text(caption_zh + "\n")
        zh_md_path.write_text(caption_zh + "\n")
        payload.update({"caption_zh_txt_path": str(zh_txt_path), "caption_zh_md_path": str(zh_md_path)})
    return payload


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
    summary_rows = []
    with plt.rc_context(PUBLICATION_RCPARAMS):
        fig, axes = plt.subplots(1, len(tags), figsize=(14.0, 4.0), dpi=260, layout="constrained")
        panel_labels = ["(a)", "(b)", "(c)"]

        for panel_idx, (ax, tag) in enumerate(zip(np.atleast_1d(axes), tags)):
            for label, run_dir in run_map.items():
                series = read_scalar_series(run_dir, tag)
                ax.plot(series["step"], series["value"], label=label, linewidth=1.8)
                best_idx = int(series["value"].idxmin())
                ax.scatter(
                    [float(series.loc[best_idx, "step"])],
                    [float(series.loc[best_idx, "value"])],
                    s=20,
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
            _panel_label(ax, panel_labels[panel_idx])
            ax.set_title(tag.replace("val/", "").replace("_", " "))
            ax.set_xlabel("Training step")
            ax.set_ylabel("Validation loss")
            ax.grid(alpha=0.25, linestyle="--")
            ax.legend(frameon=False)

        fig.suptitle("Supplementary training audit for the stage-1 tokenizer runs", fontsize=12)

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
        "caption_en": CAPTION_SUPP_TRAINING_EN,
        "caption_zh": CAPTION_SUPP_TRAINING_ZH,
    }
    summary.update(_write_caption_files(output_dir, save_stem, CAPTION_SUPP_TRAINING_EN, CAPTION_SUPP_TRAINING_ZH))
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
    plot_top_n_cases: int = 2,
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
    results_df["delta_rmse"] = results_df["old_rmse"] - results_df["new_rmse"]
    results_df["delta_mae"] = results_df["old_mae"] - results_df["new_mae"]
    plot_indices = (
        results_df.sort_values(["delta_rmse", "delta_mae"], ascending=False)
        .head(min(int(plot_top_n_cases), len(results_df)))
        .index.tolist()
    )
    plot_cases = [prepared_cases[idx] for idx in plot_indices]

    err_vmax = max(
        max(compressed_percentile(case["old_abs_err"], 99), compressed_percentile(case["new_abs_err"], 99))
        for case in plot_cases
    )
    improve_vmax = max(
        compressed_percentile(np.ma.abs(case["old_abs_err"] - case["new_abs_err"]), 99) for case in plot_cases
    )

    with plt.rc_context(PUBLICATION_RCPARAMS):
        nrows = len(plot_cases)
        fig = plt.figure(figsize=(16.2, 3.15 * nrows), dpi=260, layout="constrained")
        gs = fig.add_gridspec(nrows=nrows, ncols=7, width_ratios=[1.70, 1, 1, 1, 1, 1, 1.02])
        text_axes: list[plt.Axes] = []
        axes = np.empty((nrows, 6), dtype=object)
        for row_idx in range(nrows):
            text_axes.append(fig.add_subplot(gs[row_idx, 0]))
            for col_idx in range(6):
                axes[row_idx, col_idx] = fig.add_subplot(gs[row_idx, col_idx + 1])

        col_titles = [
            "Input",
            "MAE tokenizer\nreconstruction",
            "PHuber tokenizer\nreconstruction",
            "MAE tokenizer\nabsolute error",
            "PHuber tokenizer\nabsolute error",
            "Error reduction",
        ]
        panel_labels = [f"({chr(ord('a') + idx)})" for idx in range(nrows * len(col_titles))]
        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(title, fontsize=9.5, fontweight="semibold")

        state_ims = []
        err_ims = []
        diff_ims = []
        for row_idx, case in enumerate(plot_cases):
            focus_idx = case["focus_idx"]
            focus_title = case["frame_titles"][focus_idx]
            extent = case["crop_extent"]
            rmse_delta = float(case["metrics"]["rmse_old"][focus_idx] - case["metrics"]["rmse_new"][focus_idx])
            mae_delta = float(case["metrics"]["mae_old"][focus_idx] - case["metrics"]["mae_new"][focus_idx])
            case_tag = f"Case {['I', 'II', 'III', 'IV', 'V'][row_idx]}"

            text_ax = text_axes[row_idx]
            text_ax.axis("off")
            text_ax.text(0.98, 0.76, case_tag, ha="right", va="center", fontsize=9.0, fontweight="bold")
            text_ax.text(0.98, 0.62, case["case"].name, ha="right", va="center", fontsize=10.0, fontweight="semibold")
            text_ax.text(0.98, 0.46, f"{case['case'].t0:%Y-%m-%d} | lead {focus_idx + 1} ({focus_title})", ha="right", va="center", fontsize=8.6)
            text_ax.text(0.98, 0.28, f"RMSE: {float(case['metrics']['rmse_old'][focus_idx]):.4f} -> {float(case['metrics']['rmse_new'][focus_idx]):.4f}", ha="right", va="center", fontsize=8.2)
            text_ax.text(0.98, 0.10, f"ΔRMSE = {rmse_delta:+.4f}\nΔMAE = {mae_delta:+.4f}", ha="right", va="bottom", fontsize=8.2, fontweight="semibold")

            panels = [
                (case["input_phys"][focus_idx], "YlGnBu", search_cfg.clip[0], search_cfg.clip[1], f"RZSM ({UNIT_LABEL})"),
                (
                    case["old_phys"][focus_idx],
                    "YlGnBu",
                    search_cfg.clip[0],
                    search_cfg.clip[1],
                    f"RZSM ({UNIT_LABEL})",
                ),
                (
                    case["new_phys"][focus_idx],
                    "YlGnBu",
                    search_cfg.clip[0],
                    search_cfg.clip[1],
                    f"RZSM ({UNIT_LABEL})",
                ),
                (case["old_abs_err"][focus_idx], "magma", 0.0, err_vmax, f"Absolute error ({UNIT_LABEL})"),
                (case["new_abs_err"][focus_idx], "magma", 0.0, err_vmax, f"Absolute error ({UNIT_LABEL})"),
                (
                    case["old_abs_err"][focus_idx] - case["new_abs_err"][focus_idx],
                    "coolwarm",
                    -improve_vmax,
                    improve_vmax,
                    f"Error reduction ({UNIT_LABEL})",
                ),
            ]
            for col_idx, (panel, cmap, vmin, vmax, cbar_label) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                im = ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin="upper", interpolation="nearest", aspect="auto")
                ax.tick_params(labelsize=7)
                _panel_label(ax, panel_labels[row_idx * len(col_titles) + col_idx])
                if row_idx == nrows - 1:
                    ax.set_xlabel("Longitude")
                else:
                    ax.set_xlabel("")
                    ax.set_xticklabels([])
                if col_idx == 0:
                    ax.set_ylabel("Latitude")
                else:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
                if col_idx < 3:
                    state_ims.append((im, cbar_label))
                elif col_idx < 5:
                    err_ims.append((im, cbar_label))
                else:
                    diff_ims.append((im, cbar_label))

        fig.colorbar(state_ims[0][0], ax=axes[:, 0:3].ravel().tolist(), fraction=0.020, pad=0.015, label=state_ims[0][1])
        fig.colorbar(err_ims[0][0], ax=axes[:, 3:5].ravel().tolist(), fraction=0.020, pad=0.015, label=err_ims[0][1])
        diff_cbar = fig.colorbar(
            diff_ims[0][0],
            ax=axes[:, 5].ravel().tolist(),
            fraction=0.035,
            pad=0.015,
            label=f"Old |error| - New |error| ({UNIT_LABEL})",
        )
        diff_cbar.ax.set_title("positive:\nPHuber better", fontsize=8, pad=6)
        fig.suptitle("Representative georeferenced reconstruction cases for the stage-1 RZSM tokenizers", fontsize=11.2)

    png_path = output_dir / f"{save_stem}.png"
    pdf_path = output_dir / f"{save_stem}.pdf"
    csv_path = output_dir / f"{save_stem}.csv"
    results_df.to_csv(csv_path, index=False)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "device": used_device,
        "cases_total": len(prepared_cases),
        "cases_plotted": len(plot_cases),
        "plotted_case_names": [case["case"].name for case in plot_cases],
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "caption_en": CAPTION_FIG3_EN,
        "caption_zh": CAPTION_FIG3_ZH,
    }
    summary.update(_write_caption_files(output_dir, save_stem, CAPTION_FIG3_EN, CAPTION_FIG3_ZH))
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

    with plt.rc_context(PUBLICATION_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.15), dpi=260, layout="constrained")
        panel_labels = ["(a)", "(b)", "(c)"]
        case_names = sorted(results_df["case_name"].unique())
        case_palette = {
            name: color
            for name, color in zip(case_names, ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#dd8452"])
        }

        metrics = [
            ("old_rmse", "new_rmse", "RMSE", axes[0]),
            ("old_mae", "new_mae", "MAE", axes[1]),
        ]
        for panel_idx, (old_col, new_col, label, ax) in enumerate(metrics):
            for case_name in case_names:
                subset = results_df[results_df["case_name"] == case_name]
                x = subset[old_col].to_numpy(dtype=np.float32)
                y = subset[new_col].to_numpy(dtype=np.float32)
                ax.scatter(
                    x,
                    y,
                    color=case_palette[case_name],
                    s=48,
                    edgecolor="black",
                    linewidth=0.25,
                    label=case_name if panel_idx == 0 else None,
                )
            x_all = results_df[old_col].to_numpy(dtype=np.float32)
            y_all = results_df[new_col].to_numpy(dtype=np.float32)
            lo = float(min(x_all.min(), y_all.min()))
            hi = float(max(x_all.max(), y_all.max()))
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
            _panel_label(ax, panel_labels[panel_idx])
            ax.set_xlabel(f"MAE tokenizer mean {label} ({UNIT_LABEL})")
            ax.set_ylabel(f"PHuber tokenizer mean {label} ({UNIT_LABEL})")
            ax.set_title(f"Paired {label} comparison")
            ax.text(
                0.03,
                0.03,
                "below diagonal: PHuber better",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.2,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
            )
            if panel_idx == 0:
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.86))

        win_rates = [
            float(np.mean(results_df["new_rmse"] < results_df["old_rmse"])),
            float(np.mean(results_df["new_mae"] < results_df["old_mae"])),
            float(np.mean(results_df["new_p95"] < results_df["old_p95"])),
            float(np.mean(results_df["new_grad"] < results_df["old_grad"])),
        ]
        metric_labels = ["RMSE", "MAE", "p95", "grad-MAE"]
        bars = axes[2].bar(metric_labels, win_rates, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
        axes[2].set_ylim(0.0, 1.08)
        _panel_label(axes[2], panel_labels[2])
        axes[2].set_ylabel("PHuber win rate")
        axes[2].set_title("Win rates across metrics")
        axes[2].grid(axis="y", alpha=0.25, linestyle="--")
        for bar, win_rate in zip(bars, win_rates):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2.0,
                min(win_rate + 0.02, 1.04),
                f"{100.0 * win_rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.4,
                fontweight="semibold",
            )

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
        "caption_en": CAPTION_FIG2_EN,
        "caption_zh": CAPTION_FIG2_ZH,
    }
    summary.update(_write_caption_files(output_dir, save_stem, CAPTION_FIG2_EN, CAPTION_FIG2_ZH))
    with (output_dir / f"{save_stem}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return results_df, summary


def make_stage1_summary_table(
    benchmark_df: pd.DataFrame,
    output_dir: Path,
    save_stem: str = "table1_stage1_summary",
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    metric_map = {
        "RMSE": ("old_rmse", "new_rmse", "delta_rmse"),
        "MAE": ("old_mae", "new_mae", "delta_mae"),
        "p95 abs err": ("old_p95", "new_p95", "delta_p95"),
        "grad-MAE": ("old_grad", "new_grad", "delta_grad"),
    }
    for label, (old_col, new_col, delta_col) in metric_map.items():
        old_vals = benchmark_df[old_col].to_numpy(dtype=np.float32)
        new_vals = benchmark_df[new_col].to_numpy(dtype=np.float32)
        delta_vals = benchmark_df[delta_col].to_numpy(dtype=np.float32)
        ci_lo, ci_hi = bootstrap_mean_ci(delta_vals)
        raw_rows.append(
            {
                "metric": label,
                "old_mean": float(old_vals.mean()),
                "old_std": float(old_vals.std(ddof=1)) if old_vals.size > 1 else 0.0,
                "new_mean": float(new_vals.mean()),
                "new_std": float(new_vals.std(ddof=1)) if new_vals.size > 1 else 0.0,
                "delta_mean_old_minus_new": float(delta_vals.mean()),
                "delta_ci95_low": ci_lo,
                "delta_ci95_high": ci_hi,
                "new_better_fraction": float(np.mean(new_vals < old_vals)),
            }
        )

    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(
        {
            "Metric": raw_df["metric"],
            "MAE tokenizer": [f"{m:.5f} ± {s:.5f}" for m, s in zip(raw_df["old_mean"], raw_df["old_std"])],
            "PHuber tokenizer": [f"{m:.5f} ± {s:.5f}" for m, s in zip(raw_df["new_mean"], raw_df["new_std"])],
            "Δ (old - new)": [f"{v:+.5f}" for v in raw_df["delta_mean_old_minus_new"]],
            "95% CI": [
                f"[{lo:+.5f}, {hi:+.5f}]"
                for lo, hi in zip(raw_df["delta_ci95_low"], raw_df["delta_ci95_high"])
            ],
            "PHuber win rate": [f"{100.0 * v:.1f}%" for v in raw_df["new_better_fraction"]],
        }
    )
    csv_path = output_dir / f"{save_stem}.csv"
    raw_csv_path = output_dir / f"{save_stem}.raw.csv"
    md_path = output_dir / f"{save_stem}.md"
    tex_path = output_dir / f"{save_stem}.tex"
    json_path = output_dir / f"{save_stem}.json"
    summary_df.to_csv(csv_path, index=False)
    raw_df.to_csv(raw_csv_path, index=False)
    with md_path.open("w") as f:
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")
    with tex_path.open("w") as f:
        f.write(summary_df.to_latex(index=False, escape=False))
        f.write("\n")
    payload = {
        "csv_path": str(csv_path),
        "raw_csv_path": str(raw_csv_path),
        "md_path": str(md_path),
        "tex_path": str(tex_path),
        "json_path": str(json_path),
        "num_samples": int(len(benchmark_df)),
        "num_cases": int(benchmark_df["case_name"].nunique()) if "case_name" in benchmark_df.columns else None,
        "caption_en": CAPTION_TABLE1_EN,
        "caption_zh": CAPTION_TABLE1_ZH,
    }
    payload.update(_write_caption_files(output_dir, save_stem, CAPTION_TABLE1_EN, CAPTION_TABLE1_ZH))
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return summary_df, payload
