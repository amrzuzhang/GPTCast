# GPTCast-SWVL1：基於 ERA5-Land 的土壤濕度預報

[![English](https://img.shields.io/badge/English-README.md-0A7E8C?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/简体中文-README.zh--CN.md-0A7E8C?style=for-the-badge)](README.zh-CN.md)
[![日本語](https://img.shields.io/badge/日本語-README.ja.md-0A7E8C?style=for-the-badge)](README.ja.md)

## 專案定位

這個倉庫不是單純把上游 GPTCast 換一個資料集，而是把 GPTCast 從**降水/雷達外推**
改造成**面向華東區域的日尺度土壤濕度預報模型**。

目前主線目標是：

- 用最近 7 天的土壤濕度狀態作為 `state`
- 用降水、蒸散、徑流、溫度、輻射作為 `forcing`
- 預報未來 `D+1 ... D+7` 的土壤水狀態
- 重點放在根區土壤濕度，而不只是表層

## 目前方法

### 第一階段：Tokenizer

第一階段先訓練土壤濕度 tokenizer，把連續土壤濕度場離散成 token。

目前重點變數：

- `swvl1`
- `rzsm_0_100cm`
- `soil_water_storage_0_100cm_mm`

### 第二階段：GPT 預報器

第二階段不是直接做影像外推，而是做 **state + forcing** 的 token 生成預報：

- baseline：`gptcast_16x16_rzsm_hydro`
- 增強版：`gptcast_16x16_rzsm_hydro_static`

增強版不是硬塞自製 physics loss，而是加入**更有物理意義的靜態背景資訊**：

- `land_mask`
- `latitude_norm`
- `longitude_norm`
- `sand_fraction`
- `clay_fraction`
- `silt_fraction`
- `porosity`

並透過**獨立的 static encoder 分支**與動態 forcing 分支融合。

## 為什麼這樣做

這個專案關心的不是「圖看起來像不像」，而是：

- 根區水分能不能被短期穩定預報
- 預報是否更接近農業與乾旱監測的真實需求
- 模型是否能利用土壤與地形背景，而不是只記住時間序列紋理

## 程式結構

目前新增的關鍵模組包括：

- 資料：
  - `gptcast/data/era5land_hydro.py`
  - `gptcast/data/era5land_hydro_datamodule.py`
- 設定：
  - `configs/data/era5land_hydro.yaml`
  - `configs/experiment/vae_mae_rzsm.yaml`
  - `configs/experiment/vae_phuber_rzsm.yaml`
  - `configs/experiment/gptcast_16x16_rzsm_hydro.yaml`
  - `configs/experiment/gptcast_16x16_rzsm_hydro_static.yaml`
- 靜態特徵：
  - `data/build_era5land_static_features.py`
  - `data/download_soilgrids_static.py`

## 使用說明

完整環境設定、訓練指令、SoilGrids 靜態資料下載、雙卡訓練方式，請看英文主 README：

- [README.md](README.md)

如果只想快速理解這個倉庫在做什麼，可以概括為：

> 用 GPTCast 的 token 生成框架，做 ERA5-Land 根區土壤濕度的日尺度狀態預報，並逐步加入土壤/地形等更有物理意義的背景資訊。
