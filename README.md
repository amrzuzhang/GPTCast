<div align="center">

# GPTCast-SWVL1: a weather language model for soil moisture forecasting (ERA5-Land)

_Fork of GPTCast (Franch et al., GMD 2025), adapted to ERA5-Land `swvl1` (volumetric soil water, layer 1)._

[![简体中文](https://img.shields.io/badge/简体中文-README.zh--CN.md-0A7E8C?style=for-the-badge)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/繁體中文-README.zh--TW.md-0A7E8C?style=for-the-badge)](README.zh-TW.md)
[![日本語](https://img.shields.io/badge/日本語-README.ja.md-0A7E8C?style=for-the-badge)](README.ja.md)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Upstream Paper](http://img.shields.io/badge/upstream%20paper-GMD-B31B1B.svg)](https://doi.org/10.5194/gmd-18-5351-2025)

</div>

<br>

## This Work

This fork is not a generic dataset swap. It repurposes GPTCast from precipitation/radar nowcasting to
**daily short-range soil moisture forecasting over East China**, with emphasis on **hydrologically meaningful
state prediction** instead of image extrapolation.

Current mainline:

- **Stage 1**: train a root-zone tokenizer on `rzsm_0_100cm`
- **Stage 2 baseline**: train a `state + forcing` daily GPT forecaster
- **Stage 2 enhancement**: add **physical-context-aware static information** (terrain / soil background) through a
  dedicated static encoder branch

Scientific target:

- predict `D+1 ... D+7` soil water states from recent soil moisture history
- focus on `swvl1`, `rzsm_0_100cm`, and `soil_water_storage_0_100cm_mm`
- improve physical realism with **forcing design + static context**, not with ad hoc homemade physics losses

Method summary:

- **State sequence**: recent soil moisture states tokenized by a first-stage VAE/VQ tokenizer
- **Dynamic forcing**: precipitation, evapotranspiration, runoff, temperature, and radiation
- **Static physical context**: land mask, geography, and now terrain/soil properties such as
  `sand_fraction`, `clay_fraction`, `silt_fraction`, and `porosity`
- **Second-stage model**: GPT-style autoregressive token forecasting with separate dynamic/static conditioning paths
- **Physical evaluation**: teacher-forced `tf_phys_*` monitoring during validation plus rollout-based downstream evaluation

Repository intent:

- keep the upstream GPTCast code structure where possible
- add hydro-specific data modules, configs, and notebooks
- make the project publishable as a soil-moisture forecasting paper rather than a precipitation-nowcasting fork

## Description

This repository is a **fork** of the original GPTCast codebase.

The upstream project targets **precipitation/radar nowcasting**. This fork keeps the overall code structure and
adapts it to **forecast daily soil moisture** using **ERA5-Land**:
- target variable: `swvl1` (volumetric soil water layer 1)
- typical use case: predict `d+1 ... d+7` from a 7-day context window (see notes in configs/notebooks)

### Upstream Reference (GPTCast paper)

```
@Article{gmd-18-5351-2025,
AUTHOR = {Franch, G. and Tomasi, E. and Wanjari, R. and Poli, V. and Cardinali, C. and Alberoni, P. P. and Cristoforetti, M.},
TITLE = {GPTCast: a weather language model for precipitation nowcasting},
JOURNAL = {Geoscientific Model Development},
VOLUME = {18},
YEAR = {2025},
NUMBER = {16},
PAGES = {5351--5371},
URL = {https://gmd.copernicus.org/articles/18/5351/2025/},
DOI = {10.5194/gmd-18-5351-2025}
}
```

<b>upstream paper</b>: [https://gmd.copernicus.org/articles/18/5351/2025/](https://doi.org/10.5194/gmd-18-5351-2025)

## Fork Notes (ERA5-Land Soil Moisture / Hydro)

This repository is a fork of the original GPTCast codebase.
In addition to the upstream precipitation/radar nowcasting workflow, this fork adapts the project to
**ERA5-Land soil moisture forecasting**, with the current mainline centered on:
- surface soil moisture (`swvl1`)
- root-zone moisture / storage (`rzsm_0_100cm`, `soil_water_storage_0_100cm_mm`)
- hydro-aware second-stage GPT experiments with explicit forcing inputs

What is added in this fork:
- Dataset + LightningDataModule:
  - `gptcast/data/era5land_swvl1.py`
  - `gptcast/data/era5land_swvl1_datamodule.py`
  - `gptcast/data/era5land_hydro.py`
  - `gptcast/data/era5land_hydro_datamodule.py`
- Hydra configs (same overall structure as the upstream experiments):
  - `configs/data/era5land_swvl1.yaml` (legacy surface-only path)
  - `configs/data/era5land_hydro.yaml` (current hydro/root-zone path)
  - `configs/experiment/vae_mae_swvl1.yaml` (baseline first stage)
  - `configs/experiment/vae_phuber_swvl1.yaml` (main first stage)
  - `configs/experiment/vae_mae_rzsm.yaml` / `configs/experiment/vae_phuber_rzsm.yaml`
  - `configs/experiment/gptcast_16x16_swvl1_hydro.yaml`
  - `configs/experiment/gptcast_16x16_rzsm_hydro.yaml`
  - `configs/experiment/gptcast_16x16_era5land_swvl1.yaml` (legacy surface-only second stage)
- Notebooks that **mirror the original notebook structure and plotting style**:
  - `notebooks/swvl1/example_autoencoder_reconstruction.ipynb`
  - `notebooks/swvl1/example_gptcast_forecast.ipynb`

Data is intentionally **not included** in this repo because it is large.

## Why This Fork Exists

This fork is not a simple dataset swap. The goal is to move GPTCast from
**weather image forecasting** toward **hydrologically meaningful state forecasting**.

The current research focus is:

- use recent soil moisture states as `state`
- use precipitation, evapotranspiration, runoff, temperature, and radiation as `forcing`
- predict future `D+1 ... D+7` soil water states, especially root-zone moisture

The most important targets are:

- `swvl1`: surface soil moisture
- `rzsm_0_100cm`: root-zone soil moisture
- `soil_water_storage_0_100cm_mm`: root-zone water storage

Why this matters:

- agriculture depends more directly on root-zone water availability than on rainfall images
- drought and flash-drought monitoring require soil moisture memory, not only precipitation
- flood, runoff, and landslide risk depend strongly on antecedent wetness conditions
- hydrologic forecasting benefits from realistic short-range land-state background fields

So the scientific aim of this fork is to explore a **state + forcing generative forecasting framework**
for short-range hydrologic state prediction, rather than only extrapolating precipitation patterns.

For a fuller research-motivation note, including historical context and real DOI-backed references, see
[`RESEARCH_SIGNIFICANCE.md`](RESEARCH_SIGNIFICANCE.md).


## How to run

### Environment

This fork is commonly used with a Conda environment (single GPU is supported).

Hydra expects the env var `PROJECT_ROOT` to be set:

```bash
cd /path/to/GPTCast
export PROJECT_ROOT="$(pwd)"
```

Example Conda setup:

```bash
conda create -n gptcast python=3.12 -y
conda activate gptcast

# Install package + dependencies
pip install -e .
```

### ERA5-Land SWVL1 (Soil Moisture) Notebooks (This Fork)

These notebooks are the SWVL1 equivalents of the upstream examples and keep the same cell order/logic as much as possible:
- `notebooks/swvl1/example_autoencoder_reconstruction.ipynb`
- `notebooks/swvl1/example_gptcast_forecast.ipynb`

Note: `pysteps` is **optional** and only needed for the original precipitation plotting utilities.
The SWVL1 notebooks use `plot_era5land(...)` / `plot_mutiple_era5land(...)` from `gptcast/utils/plotting.py`.


## Training

### ERA5-Land SWVL1 Dataset Layout (This Fork)

This fork assumes you already have the yearly ERA5-Land NetCDFs locally.

Expected layout:

```
data/0.1/1/land_surface/<YEAR>/volumetric_soil_water_layer_1.nc
```

The NetCDF is expected to contain:
- variable `swvl1`
- coordinates including `time` (daily), `latitude`, `longitude`

If you need to *download* swvl1 / hydro variables into this layout, use:

```bash
# Copernicus CDS (recommended; yearly requests, needs ~/.cdsapirc)
python data/download_era5land_landbench_style_cds.py \
  --download-profile hydrology \
  --year-start 1979 --year-end 2020 \
  --area 42.0,105.0,20.0,125.0 \
  --time 12:00 \
  --max-download-attempts 12
```

Generate the MIARAD-style metadata CSVs (yearly rows) with:

```bash
python data/make_era5land_swvl1_csv_yearly.py
```

To download a more hydrology-oriented ERA5-Land dataset layout for the next stage
(multi-layer soil moisture + forcing + runoff/ET), use:

```bash
python data/download_era5land_landbench_style_cds.py \
  --download-profile hydrology \
  --years 1979,1980,1981 \
  --area 42.0,105.0,20.0,125.0 \
  --time 12:00 \
  --num-workers 1 \
  --max-download-attempts 8
```

Profiles:
- `baseline`: original forcing-only layout
- `hydrology`: baseline + `swvl1-4`, evapotranspiration, runoff, root-zone derivatives
- `full`: hydrology + ET component breakdown

The new generic hydro datamodule lives in:

```text
gptcast/data/era5land_hydro.py
gptcast/data/era5land_hydro_datamodule.py
```

It supports:
- a main image/state variable such as `swvl1` or `rzsm_0_100cm`
- optional forcing variables returned as a separate `forcing` tensor
- future state/forcing experiments without changing the baseline SWVL1 datamodule

If CDS/object-store downloads are unstable, you can make retries more tolerant:

```bash
python data/download_era5land_landbench_style_cds.py \
  --download-profile hydrology \
  --year-start 1979 --year-end 2020 \
  --max-download-attempts 12 \
  --retry-sleep-seconds 10 \
  --retry-backoff 2.0 \
  --retry-jitter-seconds 2.0
```

### Train the VAE
Train the first stage (the VAE) with one of the following configurations contained in the folder [configs/experiment/](configs/experiment/):
- [vaeganvq_mae](configs/experiment/vaeganvq_mae.yaml) - Mean Absolute Error loss

```bash
# train the original upstream MAE tokenizer on GPU
# the result (including model checkpoints) will be saved in the folder `logs/train/`
python gptcast/train.py trainer=gpu experiment=vaeganvq_mae
```

Train the SWVL1 VAE (this fork):

```bash
python gptcast/train.py trainer=gpu experiment=vae_mae_swvl1
python gptcast/train.py trainer=gpu experiment=vae_phuber_swvl1

# quick smoke test (Hydra strict mode requires '+' for new keys)
python gptcast/train.py trainer=gpu experiment=vae_phuber_swvl1 \
  trainer.max_epochs=1 +trainer.limit_train_batches=50 +trainer.limit_val_batches=10 \
  data.batch_size=2 data.num_workers=0
```

Recommended A800 80G commands (server):

```bash
# connect from local terminal
ssh user@server

cd /path/to/GPTCast
export PROJECT_ROOT="$PWD"
```

```bash
# Stage 1A: MAE tokenizer
python gptcast/train.py \
  experiment=vae_mae_swvl1 \
  test=False \
  data.batch_size=8 \
  data.num_workers=8 \
  data.pin_memory=true \
  data.center_crop_val=true \
  model.base_learning_rate=2.25e-6
```

```bash
# Stage 1B: PHuber tokenizer
python gptcast/train.py \
  experiment=vae_phuber_swvl1 \
  test=False \
  data.batch_size=8 \
  data.num_workers=8 \
  data.pin_memory=true \
  data.center_crop_val=true \
  model.base_learning_rate=2.25e-6
```

Reason: the code scales effective learning rate with batch size, so these
commands keep the original default effective LR while using a larger A800-friendly batch.

### Next Hydro Step: Root-Zone Tokenizer

After downloading the `hydrology` profile, you can switch the first-stage target
from surface `swvl1` to root-zone moisture (`rzsm_0_100cm`):

```bash
python gptcast/train.py trainer=gpu experiment=vae_mae_rzsm test=False
python gptcast/train.py trainer=gpu experiment=vae_phuber_rzsm test=False
```

These experiments use the generic hydro datamodule and are the recommended path
for moving from surface-state prediction toward more hydrologically meaningful storage prediction.

### Hydro GPT: State + Forcing

The next step is to train the second stage with the generic hydro datamodule and
explicit forcing inputs. Two configs are provided:

- `configs/experiment/gptcast_16x16_swvl1_hydro.yaml`
- `configs/experiment/gptcast_16x16_rzsm_hydro.yaml`

The `swvl1_hydro` config is useful for validating the conditional pipeline with
your current files. The `rzsm_hydro` config is the preferred hydrologic path
once the root-zone variables are fully downloaded.

The current route-A enhancement is:

- `configs/experiment/gptcast_16x16_rzsm_hydro.yaml`
  - clean baseline (`state + forcing`, no static terrain/soil context)
- `configs/experiment/gptcast_16x16_rzsm_hydro_static.yaml`
  - physical-context-aware enhancement
  - uses a separate static encoder branch for terrain/soil background fields
  - currently expects 7 static channels:
    `land_mask`, `latitude_norm`, `longitude_norm`, `sand_fraction`,
    `clay_fraction`, `silt_fraction`, `porosity`

Important wording:
- this static-aware route is a **physical-context-aware enhancement**
- it is **not** a hard physics-loss / hard conservation method
- `tf_phys_*` metrics remain **physical monitoring**, not physical constraints

### Static Terrain / Soil Context

The current static-feature builder supports two modes:

1. minimal geography only:
   - `land_mask`
   - `latitude_norm`
   - `longitude_norm`
2. enhanced terrain / soil context when ancillary NetCDFs are available

To build the minimal static features:

```bash
python data/build_era5land_static_features.py \
  --base-dir data/0.1/1 \
  --year 1979
```

If you have external ancillary rasters already converted to NetCDF files on disk,
place them in a directory such as:

```text
data/0.1/1/static_raw/
  elevation_m.nc
  sand_fraction.nc
  clay_fraction.nc
  silt_fraction.nc
  porosity.nc
  field_capacity.nc
  wilting_point.nc
  depth_to_bedrock_m.nc
  depth_to_water_table_m.nc
  topographic_wetness_index.nc
```

Then run:

```bash
python data/build_era5land_static_features.py \
  --base-dir data/0.1/1 \
  --year 1979 \
  --ancillary-dir data/0.1/1/static_raw
```

Notes:
- `elevation_m.nc` is special: if present, the script also derives
  `slope_degrees`, `aspect_sin`, and `aspect_cos` automatically.
- Each ancillary file should contain either a variable with the same name as the
  filename stem, or a single 2D data variable that can be inferred automatically.
- The hydro datamodule now recognizes terrain/soil static keys such as
  `elevation_m`, `slope_degrees`, `aspect_sin`, `aspect_cos`, `sand_fraction`,
  `clay_fraction`, `silt_fraction`, `porosity`, `field_capacity`,
  `wilting_point`, `depth_to_bedrock_m`, `depth_to_water_table_m`, and
  `topographic_wetness_index`.

If you want to automatically download a subset of soil static properties aligned to
the ERA5-Land grid, use the SoilGrids helper:

```bash
python data/download_soilgrids_static.py \
  --base-dir data/0.1/1 \
  --year 1979
```

This downloads/derives:
- `sand_fraction`
- `clay_fraction`
- `silt_fraction`
- `porosity` (derived from SoilGrids bulk density)

By default it also regenerates the minimal static files into `data/0.1/1/static/`.

Important:
- This is **not** part of the CDS ERA5-Land product, so the original CDS downloader
  cannot fetch these fields.
- The current `soilgrids` Python wrapper/service map does **not** expose
  `wv003` / `wv1500` as service ids, so `field_capacity` / `wilting_point`
  are not downloaded automatically by this helper.
- DEM / topographic wetness / bedrock / water-table depth still need their own
  sources. The static builder can already ingest those once you provide NetCDFs.

For the hydro GPT configs, the transformer budget is expanded to
`block_size=2304`, allowing a small multi-token forcing prefix while keeping
the original 16x16 spatial tokenization.

Example:

```bash
python gptcast/train.py trainer=gpu experiment=gptcast_16x16_swvl1_hydro \
  model.first_stage.ckpt_path=<path_to_swvl1_tokenizer_checkpoint> \
  test=False
```

```bash
python gptcast/train.py trainer=gpu experiment=gptcast_16x16_rzsm_hydro \
  model.first_stage.ckpt_path=<path_to_rzsm_tokenizer_checkpoint> \
  test=False
```

### Train GPTCast
After training the VAE, train the GPTCast model with one of the following configurations contained in the folder [configs/experiment/](configs/experiment/):
- [gptcast_8x8](configs/experiment/gptcast_8x8.yaml) - 8x8 token spatial context (128x128 pixels)
- [gptcast_16x16](configs/experiment/gptcast_16x16.yaml) - 16x16 token spatial context (256x256 pixels)

```bash
# train GPTCast with a 16x16 token spatial context on GPU
# the result (including model checkpoints) will be saved in the folder `logs/train/`
# the VAE checkpoint path should be provided
python gptcast/train.py trainer=gpu experiment=gptcast_16x16 model.first_stage.ckpt_path=<path_to_vae_checkpoint>
```

Preferred second-stage path in this fork:

```bash
python gptcast/train.py trainer=gpu experiment=gptcast_16x16_swvl1_hydro \
  model.first_stage.ckpt_path=<path_to_swvl1_vae_checkpoint> \
  test=False

python gptcast/train.py trainer=gpu experiment=gptcast_16x16_rzsm_hydro \
  model.first_stage.ckpt_path=<path_to_rzsm_vae_checkpoint> \
  test=False
```

Legacy surface-only GPT baseline (kept for comparison only):

```bash
python gptcast/train.py trainer=gpu experiment=gptcast_16x16_era5land_swvl1 \
  model.first_stage.ckpt_path=<path_to_swvl1_vae_checkpoint>
```

Recommended A800 80G command (server, hydro mainline):

```bash
# pick the newest first-stage checkpoint on the server
FIRST_STAGE_CKPT=$(find logs/train/runs -path '*/checkpoints/*.ckpt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
echo "$FIRST_STAGE_CKPT"
```

```bash
# Stage 2: hydro GPT forecaster
python gptcast/train.py \
  experiment=gptcast_16x16_rzsm_hydro \
  model.first_stage.ckpt_path="$FIRST_STAGE_CKPT" \
  test=False \
  data.batch_size=8 \
  data.num_workers=8 \
  data.pin_memory=true \
  data.center_crop_val=true \
  model.base_learning_rate=2.8125e-6
```

Recommended two-GPU run (baseline vs physical-context-aware static enhancement):

```bash
cd /path/to/GPTCast
export PROJECT_ROOT="$PWD"

FIRST_STAGE_CKPT=/path/to/phuber_rzsm_best.ckpt
```

```bash
# GPU 0: clean baseline
CUDA_VISIBLE_DEVICES=0 python gptcast/train.py \
  experiment=gptcast_16x16_rzsm_hydro \
  model.first_stage.ckpt_path="$FIRST_STAGE_CKPT" \
  test=False \
  data.batch_size=8 \
  data.num_workers=8 \
  data.pin_memory=true \
  data.center_crop_val=true \
  model.base_learning_rate=2.8125e-6
```

```bash
# GPU 1: physical-context-aware static enhancement
CUDA_VISIBLE_DEVICES=1 python gptcast/train.py \
  experiment=gptcast_16x16_rzsm_hydro_static \
  model.first_stage.ckpt_path="$FIRST_STAGE_CKPT" \
  test=False \
  data.batch_size=8 \
  data.num_workers=8 \
  data.pin_memory=true \
  data.center_crop_val=true \
  model.base_learning_rate=2.8125e-6
```

Compared with the clean baseline, the current static-aware configuration adds:
- 7 static terrain/soil channels on top of the 40 dynamic forcing channels
- a separate static encoder branch that is fused with the dynamic forcing encoder

So the conditioning changes from:
- baseline: `40 = 5 forcing vars x 8 context steps`

to:
- physical-context-aware: `47 = 40 dynamic forcing + 7 static terrain/soil channels`

If you prefer a larger batch on the same GPU:

```bash
# use batch_size=10 and keep effective LR unchanged
data.batch_size=10 model.base_learning_rate=2.25e-6
```

Practical note: the legacy surface-only inference path is still designed around a maximum context length of
**7 steps** with `block_size=2048`. The hydro GPT configs increase the transformer budget to
`block_size=2304` to accommodate a small forcing prefix while keeping the notebook defaults at 7 forecast/eval steps.

## TensorBoard

Each Hydra run writes TensorBoard logs under:

```bash
logs/train/runs/<timestamp>/tensorboard/
```

To open TensorBoard for the newest run automatically:

```bash
# on server
cd /path/to/GPTCast
LATEST_RUN=$(ls -td logs/train/runs/* | head -n 1)
echo "$LATEST_RUN"
tensorboard --logdir "$LATEST_RUN/tensorboard" --port 6006
```

To view TensorBoard for a run that is training on the server, start TensorBoard on the server:

```bash
# on server
cd /path/to/GPTCast
tensorboard --logdir logs/train --port 6006
```

Then forward the port from your local terminal:

```bash
# on local machine
ssh -L 6006:localhost:6006 user@server
```

Finally open TensorBoard in your local browser:

```text
http://localhost:6006
```
