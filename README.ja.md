# GPTCast-SWVL1: ERA5-Land に基づく土壌水分予測

[![English](https://img.shields.io/badge/English-README.md-0A7E8C?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/简体中文-README.zh--CN.md-0A7E8C?style=for-the-badge)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/繁體中文-README.zh--TW.md-0A7E8C?style=for-the-badge)](README.zh-TW.md)

## このリポジトリの目的

このリポジトリは、上流の GPTCast をそのまま別データセットに差し替えたものではありません。
もともとの **降水・レーダー nowcasting** 用 GPTCast を、
**中国東部を対象とした日次土壌水分予測モデル**へ拡張することが目的です。

現在の主な研究課題は次のとおりです。

- 直近 7 日間の土壌水分状態を `state` として使う
- 降水、蒸発散、流出、気温、放射を `forcing` として使う
- 将来 `D+1 ... D+7` の土壌水状態を予測する
- 表層だけでなく根域土壌水分を重視する

## 現在の方法

### Stage 1: Tokenizer

まず第一段階で土壌水分場を離散 token に変換する tokenizer を学習します。

主な対象変数：

- `swvl1`
- `rzsm_0_100cm`
- `soil_water_storage_0_100cm_mm`

### Stage 2: GPT 予測器

第二段階では単純な画像外挿ではなく、**state + forcing** による token 生成予測を行います。

- baseline: `soilcast_16x16_rzsm_hydro`
- 強化版: `soilcast_16x16_rzsm_hydro_static`

強化版では、ad hoc な physics loss を追加するのではなく、
**物理的に意味のある静的背景情報**を導入します。

- `land_mask`
- `latitude_norm`
- `longitude_norm`
- `sand_fraction`
- `clay_fraction`
- `silt_fraction`
- `porosity`

これらは **独立した static encoder** で処理され、動的 forcing 分岐と融合されます。

## なぜ重要か

このプロジェクトで重視しているのは「見た目が似ているか」ではなく、

- 根域の水分状態を短期的に安定して予測できるか
- 農業・干ばつ監視にとって意味のある予測になっているか
- 時系列パターンの記憶だけでなく、土壌・地形背景を利用できるか

という点です。

## 主要コンポーネント

- データ:
  - `gptcast/data/era5land_hydro.py`
  - `gptcast/data/era5land_hydro_datamodule.py`
- 設定:
  - `configs/data/era5land_hydro.yaml`
  - `configs/experiment/vae_mae_rzsm.yaml`
  - `configs/experiment/vae_phuber_rzsm.yaml`
  - `configs/experiment/soilcast_16x16_rzsm_hydro.yaml`
  - `configs/experiment/soilcast_16x16_rzsm_hydro_static.yaml`
- 静的特徴:
  - `data/build_era5land_static_features.py`
  - `data/download_soilgrids_static.py`

## 使い方

環境構築、学習コマンド、SoilGrids の静的データ取得、2GPU 実行方法などの詳細は英語版 README を参照してください。

- [README.md](README.md)

要約すると、このリポジトリは次のことを目指しています。

> GPTCast の token 生成フレームワークを使って、ERA5-Land の根域土壌水分を日次で予測し、土壌・地形などの物理的背景情報を段階的に取り込む。
