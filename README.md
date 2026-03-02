<div align="center">

# GPTCast-SWVL1: a weather language model for soil moisture forecasting (ERA5-Land)

_Fork of GPTCast (Franch et al., GMD 2025), adapted to ERA5-Land `swvl1` (volumetric soil water, layer 1)._

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Upstream Paper](http://img.shields.io/badge/upstream%20paper-GMD-B31B1B.svg)](https://doi.org/10.5194/gmd-18-5351-2025)
[![Upstream Data](http://img.shields.io/badge/upstream%20data-Zenodo-4b44ce.svg)](https://doi.org/10.5281/zenodo.13692016)
[![Upstream Models](http://img.shields.io/badge/upstream%20models-Zenodo-4b44ce.svg)](https://doi.org/10.5281/zenodo.13594332)

</div>

<br>

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

<b>upstream data</b>: https://doi.org/10.5281/zenodo.13692016

<b>upstream models</b>: https://doi.org/10.5281/zenodo.13594332


## Fork Notes (ERA5-Land SWVL1)

This repository is a fork of the original GPTCast codebase.
In addition to the upstream precipitation/radar nowcasting workflow, this fork adds **ERA5-Land soil moisture**
support for experimenting with **`swvl1` (volumetric soil water, layer 1)** forecasting.

What is added in this fork:
- Dataset + LightningDataModule:
  - `gptcast/data/era5land_swvl1.py`
  - `gptcast/data/era5land_swvl1_datamodule.py`
- Hydra configs (same overall structure as the upstream experiments):
  - `configs/data/era5land_swvl1.yaml`
  - `configs/experiment/vaeganvq_mwae_era5land_swvl1.yaml` (first stage)
  - `configs/experiment/gptcast_16x16_era5land_swvl1.yaml` (second stage)
- Notebooks that **mirror the original notebook structure and plotting style**:
  - `notebooks/swvl1/example_autoencoder_reconstruction.ipynb`
  - `notebooks/swvl1/example_gptcast_forecast.ipynb`

Data is intentionally **not included** in this repo because it is large.


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

### Use The Pretrained Models (Upstream, Precip/Radar)

Check the notebooks in the [notebooks](notebooks/) folder on how to use the pretrained models.

- See the notebook [notebooks/example_gptcast_forecast.ipynb](notebooks/example_gptcast_forecast.ipynb) for running the models on a test batch and generating a forecast.

- See the notebook [notebooks/example_autoencoder_reconstruction.ipynb](notebooks/example_autoencoder_reconstruction.ipynb) for a test on the VAE reconstruction.


### ERA5-Land SWVL1 (Soil Moisture) Notebooks (This Fork)

These notebooks are the SWVL1 equivalents of the upstream examples and keep the same cell order/logic as much as possible:
- `notebooks/swvl1/example_autoencoder_reconstruction.ipynb`
- `notebooks/swvl1/example_gptcast_forecast.ipynb`

Note: `pysteps` is **optional** and only needed for the original precipitation plotting utilities.
The SWVL1 notebooks use `plot_era5land(...)` / `plot_mutiple_era5land(...)` from `gptcast/utils/plotting.py`.


## Training

### Upstream Dataset (Precip/Radar)

To train the model on the original dataset, run the script in the [data](data/) folder to download it:

```bash
# download the dataset
python data/download_data.py
```

### ERA5-Land SWVL1 Dataset Layout (This Fork)

This fork assumes you already have the yearly ERA5-Land NetCDFs locally.

Expected layout:

```
data/0.1/1/land_surface/<YEAR>/volumetric_soil_water_layer_1.nc
```

The NetCDF is expected to contain:
- variable `swvl1`
- coordinates including `time` (daily), `latitude`, `longitude`

Generate the MIARAD-style metadata CSVs (yearly rows) with:

```bash
python data/make_era5land_swvl1_csv_yearly.py
```

### Train the VAE
Train the first stage (the VAE) with one of the following configurations contained in the folder [configs/experiment/](configs/experiment/):
- [vaeganvq_mae](configs/experiment/vaeganvq_mae.yaml) - Mean Absolute Error loss
- [vaeganvq_mwae](configs/experiment/vaeganvq_mwae.yaml) - Magnitude Weighted Absolute Error loss

```bash
# train a VAE with WMAE reconstruction loss on GPU
# the result (including model checkpoints) will be saved in the folder `logs/train/`
python gptcast/train.py trainer=gpu experiment=vaeganvq_mwae
```

Train the SWVL1 VAE (this fork):

```bash
python gptcast/train.py trainer=gpu experiment=vaeganvq_mwae_era5land_swvl1

# quick smoke test (Hydra strict mode requires '+' for new keys)
python gptcast/train.py trainer=gpu experiment=vaeganvq_mwae_era5land_swvl1 \
  trainer.max_epochs=1 +trainer.limit_train_batches=50 +trainer.limit_val_batches=10 \
  data.batch_size=2 data.num_workers=0
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

Train the SWVL1 GPTCast forecaster (this fork):

```bash
python gptcast/train.py trainer=gpu experiment=gptcast_16x16_era5land_swvl1 \
  model.first_stage.ckpt_path=<path_to_swvl1_vae_checkpoint>

# quick smoke test
python gptcast/train.py trainer=gpu experiment=gptcast_16x16_era5land_swvl1 \
  model.first_stage.ckpt_path=<path_to_swvl1_vae_checkpoint> \
  trainer.max_epochs=1 +trainer.limit_train_batches=50 +trainer.limit_val_batches=10 \
  data.batch_size=2 data.num_workers=0
```

Practical note: the provided inference utilities in `gptcast/models/gptcast.py` are designed for a maximum context length of **7 steps** (with `block_size=2048`), even though training uses 8 stacked frames.
