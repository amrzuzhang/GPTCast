# GPTCast-SWVL1：基于 ERA5-Land 的土壤湿度预测

[![English](https://img.shields.io/badge/English-README.md-0A7E8C?style=for-the-badge)](README.md)
[![繁體中文](https://img.shields.io/badge/繁體中文-README.zh--TW.md-0A7E8C?style=for-the-badge)](README.zh-TW.md)
[![日本語](https://img.shields.io/badge/日本語-README.ja.md-0A7E8C?style=for-the-badge)](README.ja.md)

## 项目定位

这个仓库不是简单把上游 GPTCast 换一个数据集，而是把 GPTCast 从**降水/雷达外推**改造成
**面向华东区域的日尺度土壤湿度预测模型**。

当前主线目标是：

- 用最近 7 天的土壤湿度状态作为 `state`
- 用降水、蒸散、径流、温度、辐射作为 `forcing`
- 预测未来 `D+1 ... D+7` 的土壤水状态
- 重点关注根区土壤湿度而不是只看表层

## 当前方法

### 第一阶段：Tokenizer

第一阶段先训练土壤湿度 tokenizer，把连续土壤湿度场离散成 token。

当前重点变量：

- `swvl1`
- `rzsm_0_100cm`
- `soil_water_storage_0_100cm_mm`

### 第二阶段：GPT 预报器

第二阶段不是直接做图像外推，而是做 **state + forcing** 的 token 生成预报：

- baseline：`gptcast_16x16_rzsm_hydro`
- 增强版：`gptcast_16x16_rzsm_hydro_static`

增强版的设计不是硬塞一个自定义 physics loss，而是加入**更有物理意义的静态背景信息**：

- `land_mask`
- `latitude_norm`
- `longitude_norm`
- `sand_fraction`
- `clay_fraction`
- `silt_fraction`
- `porosity`

并通过**单独的 static encoder 分支**和动态 forcing 分支融合。

## 为什么这样做

这个项目关心的不是“图像看起来像不像”，而是：

- 根区水分能不能被短期稳定预测
- 预测是否更接近农业和干旱监测的真实需求
- 模型是否能利用土壤和地形背景，而不是只记忆时间序列纹理

## 代码结构

当前新增的关键模块包括：

- 数据：
  - `gptcast/data/era5land_hydro.py`
  - `gptcast/data/era5land_hydro_datamodule.py`
- 配置：
  - `configs/data/era5land_hydro.yaml`
  - `configs/experiment/vae_mae_rzsm.yaml`
  - `configs/experiment/vae_phuber_rzsm.yaml`
  - `configs/experiment/gptcast_16x16_rzsm_hydro.yaml`
  - `configs/experiment/gptcast_16x16_rzsm_hydro_static.yaml`
- 静态特征：
  - `data/build_era5land_static_features.py`
  - `data/download_soilgrids_static.py`

## 使用说明

详细环境配置、训练命令、SoilGrids 静态数据下载、双卡训练命令，请查看英文主 README：

- [README.md](README.md)

如果你只想快速理解本仓库做什么，可以把它概括为：

> 用 GPTCast 的 token 生成框架，做 ERA5-Land 根区土壤湿度的日尺度状态预报，并逐步加入土壤/地形等更有物理意义的背景信息。
