from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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


CAPTION_TABLE2_EN = (
    "Table 2. Summary of the second-stage RZSM forecasting results. The table compares the clean baseline and the physical-context-aware static enhancement under the same evaluation protocol, including overall MAE/RMSE, PBIAS, NSE, improvements relative to persistence, and the baseline-to-static deltas."
)
CAPTION_TABLE2_ZH = (
    "表2. 第二阶段 RZSM 预报结果汇总表。该表在统一评估协议下对 clean baseline 与 physical-context-aware static 增强版进行对比，包括整体 MAE/RMSE、PBIAS、NSE、相对于 persistence 的改进，以及 static 相对 baseline 的差值。"
)
CAPTION_FIG4_EN = (
    "Figure 4. Lead-wise comparison of the second-stage forecasting models. Panels summarize the evolution of RMSE and MAE across forecast leads for persistence, the clean baseline, and the physical-context-aware static enhancement, together with lead-wise improvements of the static model over the baseline. Lead-wise PBIAS and NSE are exported in the companion CSV for table-based reporting."
)
CAPTION_FIG4_ZH = (
    "图4. 第二阶段模型的 lead-wise 定量对比。各子图展示 persistence、clean baseline 和 physical-context-aware static 增强版在不同预报 lead 上的 RMSE 和 MAE 变化，以及 static 模型相对 baseline 的 lead-wise 改进。lead-wise 的 PBIAS 和 NSE 已导出到配套 CSV，可用于表格汇报。"
)
CAPTION_FIG5_EN = (
    "Figure 5. Representative second-stage forecast cases. The figure compares saved evaluation panels from the clean baseline and the physical-context-aware static enhancement for selected samples, allowing qualitative inspection of target fields, persistence, and SoilCast forecasts."
)
CAPTION_FIG5_ZH = (
    "图5. 第二阶段代表性预报案例图。该图对比 clean baseline 与 physical-context-aware static 增强版在代表性样本上的评估面板，以便定性检查目标场、persistence 以及 SoilCast 预报结果。"
)
CAPTION_SUPP_STAGE2_TRAINING_EN = (
    "Supplementary Figure S2. Training audit for the second-stage forecasting runs. The curves document training loss, validation loss, and teacher-forced physical monitoring metrics for the baseline and static-aware models."
)
CAPTION_SUPP_STAGE2_TRAINING_ZH = (
    "补充图 S2. 第二阶段预报模型训练审计图。曲线展示 baseline 与 static-aware 模型的训练损失、验证损失以及 teacher-forced 物理监控指标。"
)


def _write_caption_files(output_dir: Path, save_stem: str, caption_en: str, caption_zh: str) -> dict[str, str]:
    txt_path = output_dir / f"{save_stem}.caption.txt"
    md_path = output_dir / f"{save_stem}.caption.md"
    zh_txt_path = output_dir / f"{save_stem}.caption.zh.txt"
    zh_md_path = output_dir / f"{save_stem}.caption.zh.md"
    txt_path.write_text(caption_en + "\n")
    md_path.write_text(caption_en + "\n")
    zh_txt_path.write_text(caption_zh + "\n")
    zh_md_path.write_text(caption_zh + "\n")
    return {
        "caption_txt_path": str(txt_path),
        "caption_md_path": str(md_path),
        "caption_zh_txt_path": str(zh_txt_path),
        "caption_zh_md_path": str(zh_md_path),
    }


@dataclass(frozen=True)
class EvalBundle:
    name: str
    eval_dir: Path
    summary_by_lead: pd.DataFrame
    per_sample: pd.DataFrame
    overall: dict
    run_config: dict


def load_eval_bundle(name: str, eval_dir: Path) -> EvalBundle:
    eval_dir = Path(eval_dir)
    summary_by_lead = pd.read_csv(eval_dir / "summary_by_lead.csv")
    per_sample = pd.read_csv(eval_dir / "per_sample_metrics.csv")
    overall = json.loads((eval_dir / "overall_summary.json").read_text())
    run_config = json.loads((eval_dir / "run_config.json").read_text())
    return EvalBundle(
        name=name,
        eval_dir=eval_dir,
        summary_by_lead=summary_by_lead,
        per_sample=per_sample,
        overall=overall,
        run_config=run_config,
    )


def _read_scalar_series(run_dir: Path, tag: str) -> pd.DataFrame:
    event_path = next((run_dir / "tensorboard" / "version_0").glob("events.out.tfevents.*"))
    ea = EventAccumulator(str(event_path))
    ea.Reload()
    values = ea.Scalars(tag)
    return pd.DataFrame({"step": [v.step for v in values], "value": [v.value for v in values]})


def plot_stage2_training_audit(
    run_map: dict[str, Path],
    output_dir: Path,
    save_stem: str = "figS2_stage2_training_audit",
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = ["train/loss_epoch", "val/loss_epoch", "val/tf_phys_mae_mean", "val/tf_phys_rmse_mean"]
    rows = []

    with plt.rc_context(PUBLICATION_RCPARAMS):
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), dpi=260, layout="constrained")
        axes = axes.ravel()
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]
        for ax, label, tag in zip(axes, panel_labels, tags):
            for run_label, run_dir in run_map.items():
                series = _read_scalar_series(run_dir, tag)
                ax.plot(series["step"], series["value"], linewidth=1.8, label=run_label)
                best_idx = int(series["value"].idxmin())
                rows.append(
                    {
                        "run_label": run_label,
                        "run_dir": str(run_dir),
                        "tag": tag,
                        "best_step": float(series.loc[best_idx, "step"]),
                        "best_value": float(series.loc[best_idx, "value"]),
                        "last_step": float(series["step"].iloc[-1]),
                        "last_value": float(series["value"].iloc[-1]),
                    }
                )
            _panel_label(ax, label)
            ax.set_title(tag.replace("_", " "))
            ax.set_xlabel("Training step")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.25, linestyle="--")
            ax.legend(frameon=False)
        fig.suptitle("Supplementary training audit for the stage-2 forecasting runs", fontsize=12)

    png_path = output_dir / f"{save_stem}.png"
    pdf_path = output_dir / f"{save_stem}.pdf"
    csv_path = output_dir / f"{save_stem}.csv"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)
    summary = {
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "csv_path": str(csv_path),
        "caption_en": CAPTION_SUPP_STAGE2_TRAINING_EN,
        "caption_zh": CAPTION_SUPP_STAGE2_TRAINING_ZH,
    }
    summary.update(_write_caption_files(output_dir, save_stem, CAPTION_SUPP_STAGE2_TRAINING_EN, CAPTION_SUPP_STAGE2_TRAINING_ZH))
    with (output_dir / f"{save_stem}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary_df, summary


def make_stage2_summary_table(
    baseline_bundle: EvalBundle,
    static_bundle: EvalBundle,
    output_dir: Path,
    save_stem: str = "table2_stage2_summary",
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        (
            "Overall RMSE",
            float(baseline_bundle.overall["mean_model_rmse"]),
            float(static_bundle.overall["mean_model_rmse"]),
            float(baseline_bundle.overall["mean_model_rmse"] - static_bundle.overall["mean_model_rmse"]),
        ),
        (
            "Overall MAE",
            float(baseline_bundle.overall["mean_model_mae"]),
            float(static_bundle.overall["mean_model_mae"]),
            float(baseline_bundle.overall["mean_model_mae"] - static_bundle.overall["mean_model_mae"]),
        ),
        (
            "Overall PBIAS",
            float(baseline_bundle.overall["mean_model_pbias"]),
            float(static_bundle.overall["mean_model_pbias"]),
            float(baseline_bundle.overall["mean_model_pbias"] - static_bundle.overall["mean_model_pbias"]),
        ),
        (
            "Overall NSE",
            float(baseline_bundle.overall["mean_model_nse"]),
            float(static_bundle.overall["mean_model_nse"]),
            float(static_bundle.overall["mean_model_nse"] - baseline_bundle.overall["mean_model_nse"]),
        ),
        (
            "RMSE improvement vs persistence",
            float(baseline_bundle.overall["rmse_improvement_vs_persistence"]),
            float(static_bundle.overall["rmse_improvement_vs_persistence"]),
            float(static_bundle.overall["rmse_improvement_vs_persistence"] - baseline_bundle.overall["rmse_improvement_vs_persistence"]),
        ),
        (
            "MAE improvement vs persistence",
            float(baseline_bundle.overall["mae_improvement_vs_persistence"]),
            float(static_bundle.overall["mae_improvement_vs_persistence"]),
            float(static_bundle.overall["mae_improvement_vs_persistence"] - baseline_bundle.overall["mae_improvement_vs_persistence"]),
        ),
        (
            "|PBIAS| reduction vs persistence",
            float(baseline_bundle.overall["abs_pbias_reduction_vs_persistence"]),
            float(static_bundle.overall["abs_pbias_reduction_vs_persistence"]),
            float(static_bundle.overall["abs_pbias_reduction_vs_persistence"] - baseline_bundle.overall["abs_pbias_reduction_vs_persistence"]),
        ),
        (
            "NSE gain vs persistence",
            float(baseline_bundle.overall["nse_gain_vs_persistence"]),
            float(static_bundle.overall["nse_gain_vs_persistence"]),
            float(static_bundle.overall["nse_gain_vs_persistence"] - baseline_bundle.overall["nse_gain_vs_persistence"]),
        ),
    ]

    rows = []
    for metric, baseline_val, static_val, delta in metrics:
        rows.append(
            {
                "Metric": metric,
                "Baseline": f"{baseline_val:.5f}",
                "Physical-context-aware": f"{static_val:.5f}",
                "Δ (static - baseline)": f"{delta:+.5f}",
            }
        )

    summary_df = pd.DataFrame(rows)
    csv_path = output_dir / f"{save_stem}.csv"
    md_path = output_dir / f"{save_stem}.md"
    tex_path = output_dir / f"{save_stem}.tex"
    json_path = output_dir / f"{save_stem}.json"
    summary_df.to_csv(csv_path, index=False)
    with md_path.open("w") as f:
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")
    with tex_path.open("w") as f:
        f.write(summary_df.to_latex(index=False, escape=False))
        f.write("\n")
    payload = {
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "tex_path": str(tex_path),
        "json_path": str(json_path),
        "caption_en": CAPTION_TABLE2_EN,
        "caption_zh": CAPTION_TABLE2_ZH,
        "baseline_eval_dir": str(baseline_bundle.eval_dir),
        "static_eval_dir": str(static_bundle.eval_dir),
    }
    payload.update(_write_caption_files(output_dir, save_stem, CAPTION_TABLE2_EN, CAPTION_TABLE2_ZH))
    with (output_dir / f"{save_stem}.json").open("w") as f:
        json.dump(payload, f, indent=2)
    return summary_df, payload


def make_stage2_leadwise_figure(
    baseline_bundle: EvalBundle,
    static_bundle: EvalBundle,
    output_dir: Path,
    save_stem: str = "fig4_stage2_leadwise_comparison",
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = baseline_bundle.summary_by_lead.copy()
    static = static_bundle.summary_by_lead.copy()
    merged = baseline.merge(static, on="lead", suffixes=("_baseline", "_static"))
    merged["delta_model_rmse"] = merged["model_rmse_baseline"] - merged["model_rmse_static"]
    merged["delta_model_mae"] = merged["model_mae_baseline"] - merged["model_mae_static"]

    with plt.rc_context(PUBLICATION_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=260, layout="constrained")
        panel_labels = ["(a)", "(b)", "(c)"]

        # RMSE by lead
        ax = axes[0]
        ax.plot(merged["lead"], merged["persistence_rmse_baseline"], marker="o", linewidth=1.8, label="Persistence")
        ax.plot(merged["lead"], merged["model_rmse_baseline"], marker="o", linewidth=1.8, label="Baseline")
        ax.plot(merged["lead"], merged["model_rmse_static"], marker="o", linewidth=1.8, label="Static-aware")
        _panel_label(ax, panel_labels[0])
        ax.set_title("Lead-wise RMSE")
        ax.set_xlabel("Forecast lead")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False)

        # MAE by lead
        ax = axes[1]
        ax.plot(merged["lead"], merged["persistence_mae_baseline"], marker="o", linewidth=1.8, label="Persistence")
        ax.plot(merged["lead"], merged["model_mae_baseline"], marker="o", linewidth=1.8, label="Baseline")
        ax.plot(merged["lead"], merged["model_mae_static"], marker="o", linewidth=1.8, label="Static-aware")
        _panel_label(ax, panel_labels[1])
        ax.set_title("Lead-wise MAE")
        ax.set_xlabel("Forecast lead")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False)

        # Static vs baseline gains by lead
        ax = axes[2]
        ax.bar(merged["lead"] - 0.15, merged["delta_model_rmse"], width=0.3, label="RMSE gain")
        ax.bar(merged["lead"] + 0.15, merged["delta_model_mae"], width=0.3, label="MAE gain")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        _panel_label(ax, panel_labels[2])
        ax.set_title("Static-aware gain over baseline")
        ax.set_xlabel("Forecast lead")
        ax.set_ylabel("Baseline - Static-aware")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(frameon=False)

    png_path = output_dir / f"{save_stem}.png"
    pdf_path = output_dir / f"{save_stem}.pdf"
    csv_path = output_dir / f"{save_stem}.csv"
    merged.to_csv(csv_path, index=False)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "caption_en": CAPTION_FIG4_EN,
        "caption_zh": CAPTION_FIG4_ZH,
        "baseline_eval_dir": str(baseline_bundle.eval_dir),
        "static_eval_dir": str(static_bundle.eval_dir),
    }
    payload.update(_write_caption_files(output_dir, save_stem, CAPTION_FIG4_EN, CAPTION_FIG4_ZH))
    with (output_dir / f"{save_stem}.json").open("w") as f:
        json.dump(payload, f, indent=2)
    return merged, payload


def _panel_key(panel_path: Path) -> str:
    match = re.match(r"panel_\\d+_(.+)\\.png$", panel_path.name)
    return panel_path.stem if match is None else match.group(1)


def make_stage2_case_study_figure(
    baseline_eval_dir: Path,
    static_eval_dir: Path,
    output_dir: Path,
    save_stem: str = "fig5_stage2_case_studies",
    max_cases: int = 2,
) -> tuple[pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_bundle = load_eval_bundle("baseline", baseline_eval_dir)
    static_bundle = load_eval_bundle("static", static_eval_dir)

    b = baseline_bundle.per_sample.groupby(["sample_idx", "start_key"], as_index=False)[["model_rmse", "model_mae"]].mean()
    s = static_bundle.per_sample.groupby(["sample_idx", "start_key"], as_index=False)[["model_rmse", "model_mae"]].mean()
    merged = b.merge(s, on=["sample_idx", "start_key"], suffixes=("_baseline", "_static"))
    merged["delta_rmse"] = merged["model_rmse_baseline"] - merged["model_rmse_static"]
    merged["delta_mae"] = merged["model_mae_baseline"] - merged["model_mae_static"]
    selected = merged.sort_values(["delta_rmse", "delta_mae"], ascending=False).head(int(max_cases)).reset_index(drop=True)

    baseline_panels = { _panel_key(p): p for p in sorted(Path(baseline_eval_dir).glob("panel_*.png")) }
    static_panels = { _panel_key(p): p for p in sorted(Path(static_eval_dir).glob("panel_*.png")) }

    with plt.rc_context(PUBLICATION_RCPARAMS):
        fig, axes = plt.subplots(len(selected), 2, figsize=(12.0, 4.2 * len(selected)), dpi=220, layout="constrained")
        if len(selected) == 1:
            axes = np.asarray([axes])
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]
        for row_idx, row in selected.iterrows():
            key = str(row["start_key"]).replace(":", "").replace("/", "_")
            for col_idx, (name, panel_map) in enumerate([("Baseline", baseline_panels), ("Static-aware", static_panels)]):
                ax = axes[row_idx, col_idx]
                ax.axis("off")
                img = mpimg.imread(panel_map[key])
                ax.imshow(img)
                ax.set_title(name, fontsize=10, fontweight="semibold")
                _panel_label(ax, panel_labels[row_idx * 2 + col_idx])
                if col_idx == 0:
                    ax.text(
                        -0.05,
                        0.5,
                        (
                            f"Case {row_idx + 1}\n"
                            f"sample={int(row['sample_idx'])}\n"
                            f"ΔRMSE={float(row['delta_rmse']):+.4f}\n"
                            f"ΔMAE={float(row['delta_mae']):+.4f}"
                        ),
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=8.5,
                        fontweight="semibold",
                    )

        fig.suptitle("Representative second-stage forecast cases", fontsize=12)

    png_path = output_dir / f"{save_stem}.png"
    pdf_path = output_dir / f"{save_stem}.pdf"
    csv_path = output_dir / f"{save_stem}.csv"
    selected.to_csv(csv_path, index=False)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "caption_en": CAPTION_FIG5_EN,
        "caption_zh": CAPTION_FIG5_ZH,
        "baseline_eval_dir": str(baseline_bundle.eval_dir),
        "static_eval_dir": str(static_bundle.eval_dir),
    }
    payload.update(_write_caption_files(output_dir, save_stem, CAPTION_FIG5_EN, CAPTION_FIG5_ZH))
    with (output_dir / f"{save_stem}.json").open("w") as f:
        json.dump(payload, f, indent=2)
    return selected, payload
