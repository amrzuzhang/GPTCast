#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptcast.data.era5land_hydro import Era5LandHydro  # noqa: E402
from gptcast.models import GPTCast, VAEGANVQ  # noqa: E402
from gptcast.models.components import GPT, GPTCastConfig  # noqa: E402
from gptcast.utils.plotting import plot_era5land_panel  # noqa: E402


@dataclass
class EvalSample:
    sample_idx: int
    start_key: str
    lead: int
    persistence_mae: float
    persistence_rmse: float
    model_mae: float
    model_rmse: float


def parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {text!r}")


def parse_float_list(text: str) -> tuple[float, ...]:
    items = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    return tuple(items)


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def norm_to_phys(
    x_norm: np.ndarray | np.ma.MaskedArray,
    *,
    clip: tuple[float, float],
    norm: tuple[float, float],
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


def per_lead_mae_rmse(
    pred: np.ma.MaskedArray,
    tgt: np.ma.MaskedArray,
) -> tuple[np.ndarray, np.ndarray]:
    diff = pred - tgt
    mae = np.ma.mean(np.abs(diff), axis=(1, 2)).filled(np.nan)
    rmse = np.sqrt(np.ma.mean(diff ** 2, axis=(1, 2))).filled(np.nan)
    return np.asarray(mae, dtype=np.float64), np.asarray(rmse, dtype=np.float64)


def infer_transformer_from_ckpt(ckpt_path: Path) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    tok_emb = state_dict["transformer.tok_emb.weight"]
    vocab_size, n_embd = tok_emb.shape
    block_size = state_dict["transformer.pos_emb"].shape[-2]

    n_layer = 0
    pattern = re.compile(r"transformer.blocks.(\d+)")
    for key in state_dict:
        match = pattern.search(key)
        if match is not None:
            n_layer = max(n_layer, int(match.group(1)))
    n_layer += 1

    return GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=GPTCastConfig.n_head,
        n_embd=n_embd,
    )


def load_gptcast(
    *,
    gpt_ckpt: Path,
    first_stage_ckpt: Path,
    device: str,
) -> GPTCast:
    transformer = infer_transformer_from_ckpt(gpt_ckpt)
    first_stage = VAEGANVQ.load_from_pretrained(str(first_stage_ckpt), device=device).to(device).eval()
    model = GPTCast.load_from_checkpoint(
        str(gpt_ckpt),
        transformer=transformer,
        first_stage=first_stage,
        map_location=device,
    ).to(device).eval()
    return model


def build_dataset(args: argparse.Namespace) -> Era5LandHydro:
    return Era5LandHydro(
        base_dir=str(args.base_dir),
        metadata_path_or_df=str(args.metadata_path),
        image_variable_key=args.image_variable_key,
        forcing_variable_keys=args.forcing_variable_keys,
        static_variable_keys=args.static_variable_keys,
        static_dir=None if args.static_dir is None else str(args.static_dir),
        normalize_forcing=args.normalize_forcing,
        seq_len=args.context_steps + args.forecast_steps,
        stack_seq=None,
        clip_and_normalize=tuple(args.clip_and_normalize),
        resize=args.resize,
        crop=args.crop,
        smart_crop=args.smart_crop,
        max_mask_fraction=args.max_mask_fraction,
        smart_crop_attempts=args.smart_crop_attempts,
        center_crop=args.center_crop,
        random_rotate90=False,
        drop_incomplete=True,
        max_open_years=args.max_open_years,
    )


def select_indices(dataset_len: int, max_samples: Optional[int], seed: int) -> list[int]:
    indices = list(range(dataset_len))
    if max_samples is None or max_samples >= dataset_len:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = sorted(indices[:max_samples])
    return chosen


def maybe_prepare_forcing_context(
    example: dict,
    *,
    forcing_variable_keys: list[str],
    static_variable_keys: list[str],
    context_steps: int,
    forecast_steps: int,
    expected_channels: Optional[int],
) -> Optional[np.ndarray]:
    if "forcing" not in example or (len(forcing_variable_keys) == 0 and len(static_variable_keys) == 0):
        return None

    forcing = np.asarray(example["forcing"], dtype=np.float32)
    total_steps = context_steps + forecast_steps
    n_vars = len(forcing_variable_keys)
    n_static = len(static_variable_keys)
    h, w, channels = forcing.shape
    expected_total_channels = n_vars * total_steps + n_static
    if channels != expected_total_channels:
        raise ValueError(
            f"Forcing tensor shape mismatch: got {channels} channels, expected "
            f"{n_vars} variables x {total_steps} steps + {n_static} static channels = {expected_total_channels}"
        )

    forcing_context = None
    if n_vars > 0:
        dynamic = forcing[..., : n_vars * total_steps]
        forcing_4d = dynamic.reshape(h, w, n_vars, total_steps)
        forcing_context = forcing_4d[..., :context_steps].reshape(h, w, n_vars * context_steps)

    if n_static > 0:
        static = forcing[..., n_vars * total_steps :]
        forcing_context = static if forcing_context is None else np.concatenate([forcing_context, static], axis=2)

    if expected_channels is not None and forcing_context.shape[-1] != expected_channels:
        raise ValueError(
            f"Conditioning channel mismatch: checkpoint expects {expected_channels}, "
            f"but the evaluation forcing context provides {forcing_context.shape[-1]}"
        )
    return forcing_context


def predict_rollout(
    model: GPTCast,
    *,
    context: np.ma.MaskedArray,
    forcing_context: Optional[np.ndarray],
    forecast_steps: int,
    temperature: float,
    top_k: Optional[int],
    verbosity: int,
) -> np.ma.MaskedArray:
    x = torch.tensor(context.data, dtype=torch.float32, device=model.device)
    x = rearrange(x, "s h w -> h w s")

    forcing_tensor = None
    if forcing_context is not None:
        forcing_tensor = torch.tensor(forcing_context, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        result = model.predict_sequence(
            x,
            forcing_seq=forcing_tensor,
            steps=forecast_steps,
            future=True,
            window_size=None,
            temperature=temperature,
            top_k=top_k,
            verbosity=verbosity,
        )

    pred = result["pred_sequence_nopad"].detach().cpu().numpy().squeeze()
    if forecast_steps == 1 and pred.ndim == 2:
        pred = pred[None, ...]
    output_mask = np.broadcast_to(context.mask[-1], pred.shape)
    return np.ma.array(pred, mask=output_mask)


def save_panel(
    *,
    output_dir: Path,
    sample_idx: int,
    start_key: str,
    target_phys: np.ma.MaskedArray,
    persist_phys: np.ma.MaskedArray,
    model_phys: np.ma.MaskedArray,
) -> None:
    lead_titles = [f"D+{i + 1}" for i in range(target_phys.shape[0])]
    safe_key = start_key.replace(":", "").replace("/", "_")
    panel_path = output_dir / f"panel_{sample_idx:04d}_{safe_key}.png"
    plot_era5land_panel(
        [target_phys, persist_phys, model_phys],
        row_titles=["Ground truth", "Persistence", "GPTCast"],
        frame_titles=lead_titles,
        title=f"Batch eval sample {sample_idx} | start {start_key}",
        colorbar=True,
        savepath=panel_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation for ERA5-Land GPTCast rollouts.")
    parser.add_argument("--gpt-ckpt", type=Path, required=True)
    parser.add_argument("--first-stage-ckpt", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/0.1/1")
    parser.add_argument("--image-variable-key", default="swvl1")
    parser.add_argument("--forcing-variable-keys", default="")
    parser.add_argument("--static-variable-keys", default="")
    parser.add_argument("--static-dir", type=Path, default=None)
    parser.add_argument("--normalize-forcing", type=parse_bool, default=True)
    parser.add_argument("--clip-and-normalize", default="0.0,0.8,-1.0,1.0")
    parser.add_argument("--context-steps", type=int, default=8)
    parser.add_argument("--forecast-steps", type=int, default=7)
    parser.add_argument("--resize", type=int, default=720)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--smart-crop", type=parse_bool, default=False)
    parser.add_argument("--max-mask-fraction", type=float, default=0.40)
    parser.add_argument("--smart-crop-attempts", type=int, default=50)
    parser.add_argument("--center-crop", type=parse_bool, default=True)
    parser.add_argument("--max-open-years", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", default="1")
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--save-panels", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    args.forcing_variable_keys = parse_str_list(args.forcing_variable_keys)
    args.static_variable_keys = parse_str_list(args.static_variable_keys)
    parsed_clip = parse_float_list(args.clip_and_normalize)
    if len(parsed_clip) != 4:
        raise ValueError("--clip-and-normalize must have 4 comma-separated floats")
    args.clip_and_normalize = parsed_clip
    args.top_k = None if str(args.top_k).strip().lower() == "none" else int(args.top_k)
    return args


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ROOT / "logs" / "eval" / f"gptcast_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_gptcast(
        gpt_ckpt=args.gpt_ckpt,
        first_stage_ckpt=args.first_stage_ckpt,
        device=args.device,
    )

    dataset = build_dataset(args)
    indices = select_indices(len(dataset), args.max_samples, args.seed)
    if len(indices) == 0:
        raise RuntimeError("No evaluation samples were selected.")

    expected_forcing_channels = None
    if bool(getattr(model.hparams, "use_forcing_conditioning", False)):
        expected_forcing_channels = int(model.hparams.forcing_channels)

    rows: list[EvalSample] = []
    clip = (float(args.clip_and_normalize[0]), float(args.clip_and_normalize[1]))
    norm = (float(args.clip_and_normalize[2]), float(args.clip_and_normalize[3]))

    print(f"Evaluating {len(indices)} samples -> {output_dir}")
    print(f"Model checkpoint: {args.gpt_ckpt}")
    print(f"First-stage checkpoint: {args.first_stage_ckpt}")

    for order, sample_idx in enumerate(indices):
        example = dataset[sample_idx]
        image = np.asarray(example["image"], dtype=np.float32)
        image_seq = np.transpose(image, (2, 0, 1))
        mask = np.asarray(example["mask"], dtype=bool)
        mask_seq = np.broadcast_to(mask, image_seq.shape)

        context = np.ma.array(image_seq[: args.context_steps], mask=mask_seq[: args.context_steps])
        target = np.ma.array(
            image_seq[args.context_steps : args.context_steps + args.forecast_steps],
            mask=mask_seq[args.context_steps : args.context_steps + args.forecast_steps],
        )
        persistence = np.ma.array(
            np.broadcast_to(context[-1][None, ...], target.shape),
            mask=np.broadcast_to(mask, target.shape),
        )

        forcing_context = maybe_prepare_forcing_context(
            example,
            forcing_variable_keys=args.forcing_variable_keys,
            static_variable_keys=args.static_variable_keys,
            context_steps=args.context_steps,
            forecast_steps=args.forecast_steps,
            expected_channels=expected_forcing_channels,
        )

        pred = predict_rollout(
            model,
            context=context,
            forcing_context=forcing_context,
            forecast_steps=args.forecast_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            verbosity=args.verbosity,
        )

        target_phys = norm_to_phys(target, clip=clip, norm=norm)
        persist_phys = norm_to_phys(persistence, clip=clip, norm=norm)
        pred_phys = norm_to_phys(pred, clip=clip, norm=norm)

        persist_mae, persist_rmse = per_lead_mae_rmse(persist_phys, target_phys)
        model_mae, model_rmse = per_lead_mae_rmse(pred_phys, target_phys)

        start_key = str(example["file_path_"])
        for lead in range(args.forecast_steps):
            rows.append(
                EvalSample(
                    sample_idx=sample_idx,
                    start_key=start_key,
                    lead=lead + 1,
                    persistence_mae=float(persist_mae[lead]),
                    persistence_rmse=float(persist_rmse[lead]),
                    model_mae=float(model_mae[lead]),
                    model_rmse=float(model_rmse[lead]),
                )
            )

        if order < int(args.save_panels):
            save_panel(
                output_dir=output_dir,
                sample_idx=sample_idx,
                start_key=start_key,
                target_phys=target_phys,
                persist_phys=persist_phys,
                model_phys=pred_phys,
            )

    per_sample_df = pd.DataFrame([asdict(row) for row in rows])
    per_sample_path = output_dir / "per_sample_metrics.csv"
    per_sample_df.to_csv(per_sample_path, index=False)

    summary_df = (
        per_sample_df.groupby("lead", as_index=False)[
            ["persistence_mae", "persistence_rmse", "model_mae", "model_rmse"]
        ]
        .mean()
        .sort_values("lead")
    )
    summary_df["rmse_improvement_vs_persistence"] = (
        summary_df["persistence_rmse"] - summary_df["model_rmse"]
    )
    summary_df["mae_improvement_vs_persistence"] = (
        summary_df["persistence_mae"] - summary_df["model_mae"]
    )
    summary_path = output_dir / "summary_by_lead.csv"
    summary_df.to_csv(summary_path, index=False)

    config_payload = {
        "gpt_ckpt": str(args.gpt_ckpt),
        "first_stage_ckpt": str(args.first_stage_ckpt),
        "metadata_path": str(args.metadata_path),
        "base_dir": str(args.base_dir),
        "image_variable_key": args.image_variable_key,
        "forcing_variable_keys": args.forcing_variable_keys,
        "static_variable_keys": args.static_variable_keys,
        "static_dir": None if args.static_dir is None else str(args.static_dir),
        "normalize_forcing": bool(args.normalize_forcing),
        "clip_and_normalize": list(args.clip_and_normalize),
        "context_steps": int(args.context_steps),
        "forecast_steps": int(args.forecast_steps),
        "resize": int(args.resize) if args.resize is not None else None,
        "crop": int(args.crop) if args.crop is not None else None,
        "smart_crop": bool(args.smart_crop),
        "max_mask_fraction": float(args.max_mask_fraction),
        "smart_crop_attempts": int(args.smart_crop_attempts),
        "center_crop": bool(args.center_crop),
        "max_open_years": int(args.max_open_years),
        "max_samples": None if args.max_samples is None else int(args.max_samples),
        "seed": int(args.seed),
        "temperature": float(args.temperature),
        "top_k": args.top_k,
        "verbosity": int(args.verbosity),
        "device": args.device,
        "output_dir": str(output_dir),
    }
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)

    overall = {
        "samples": int(len(indices)),
        "mean_persistence_rmse": float(per_sample_df["persistence_rmse"].mean()),
        "mean_model_rmse": float(per_sample_df["model_rmse"].mean()),
        "mean_persistence_mae": float(per_sample_df["persistence_mae"].mean()),
        "mean_model_mae": float(per_sample_df["model_mae"].mean()),
    }
    overall["rmse_improvement_vs_persistence"] = (
        overall["mean_persistence_rmse"] - overall["mean_model_rmse"]
    )
    overall["mae_improvement_vs_persistence"] = (
        overall["mean_persistence_mae"] - overall["mean_model_mae"]
    )
    with (output_dir / "overall_summary.json").open("w") as f:
        json.dump(overall, f, indent=2)

    print("\nLead-wise summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nOverall summary:")
    print(json.dumps(overall, indent=2))
    print(f"\nSaved per-sample metrics to: {per_sample_path}")
    print(f"Saved lead-wise summary to: {summary_path}")


if __name__ == "__main__":
    main()
