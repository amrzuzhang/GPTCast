# SoilCast: GPTCast-derived ERA5-Land Soil Moisture Forecasting

## 论文引言草稿（中文）

本研究面向中国东部日尺度根区土壤湿度短期预测问题，尝试将生成式时空预测框架从传统的天气图像外推任务推进到更具水文意义的陆面状态预测任务。土壤湿度，尤其是根区土壤湿度（root-zone soil moisture, RZSM），是连接降水强迫、陆面储水状态、蒸散发反馈以及产流风险的重要中间变量，在农业灌溉管理、干旱与闪旱监测、洪涝风险评估以及流域水文模拟中具有基础性作用。与短时降水预测相比，根区土壤湿度更直接表征作物可利用水和土壤水分记忆，因此对农业和水文应用往往具有更高的解释价值和决策相关性。

近年来，围绕土壤湿度的研究已经从站点观测、陆面模式和再分析产品，逐步发展到遥感反演与机器学习估计。ERA5-Land 等高时空分辨率陆面再分析产品为区域尺度连续土壤湿度建模提供了统一的数据基础。与此同时，越来越多研究表明，根区土壤湿度相较表层湿度更能反映陆气相互作用中的慢变量特征，也更贴近农业干旱、水分胁迫和前期湿润度对洪涝响应的实际需求。然而，现有研究仍主要集中于土壤湿度估计、遥感反演或点尺度预测，对于“未来数日根区土壤状态场”的区域化、空间连续、生成式预测仍相对不足。

从方法角度看，现有深度学习时空预测框架大多服务于降水 nowcasting、视频预测或通用图像序列建模，而较少针对水文状态变量构建显式的 `state + forcing` 预测结构。对于土壤湿度而言，仅依赖历史状态做 persistence 或统计外推，难以充分体现降水、蒸散发、径流、辐射和温度等外部强迫的驱动作用；而如果完全将问题视为一般图像生成任务，又容易忽略地形、土壤属性和陆面背景对状态演化的长期调制。因此，如何在生成式建模框架中显式组织过去状态、动态强迫和静态物理背景，是当前土壤湿度短期预测中一个值得系统探索的问题。

基于上述背景，本文以中国东部为研究区域，构建了一个面向日尺度根区土壤湿度预测的生成式 `state + forcing + physical context` 框架。选择中国东部作为实验区域主要基于三点考虑：第一，该区域农业生产、水资源管理以及洪涝和干旱风险评估对根区土壤湿度的短期预测具有直接需求；第二，区域内部同时包含华北平原、长江中下游和东南丘陵等不同下垫面与湿润度条件，能够为方法提供一定程度的水文异质性检验；第三，作为面向水文状态预测的第一步工作，区域聚焦有助于在保持数据定义和任务设置一致的前提下，更清晰地评估模型框架本身的有效性，而不是在全国或全球尺度下先被更强的气候与数据异质性所主导。

在具体实现上，本文以 SoilCast 的 latent token 与自回归预测框架为基础，将上游 GPTCast 的降水/雷达图像外推任务迁移到 ERA5-Land 根区土壤湿度任务。模型首先通过第一阶段 tokenizer 学习根区土壤湿度状态场的紧凑表示，再在第二阶段结合过去状态序列与动态强迫信息进行未来 `D+1` 至 `D+7` 的状态预测，并进一步探索静态地理与土壤背景信息对预测性能的增强作用。需要强调的是，本文当前工作的核心贡献并不在于提出一个严格物理约束的水文模型，而在于探索一种更接近陆面过程组织方式的生成式预测框架，即用显式的状态变量、外部强迫和物理背景信息，共同驱动短期根区土壤湿度预报。

本文的主要贡献可以概括为以下三点：  
（1）将原本面向天气图像 nowcasting 的生成式 token 预测框架迁移到 ERA5-Land 根区土壤湿度预测任务，构建了一个面向水文状态变量的区域尺度日预报流程；  
（2）提出并验证了 `state + forcing` 的第二阶段建模方式，并进一步引入 physical-context-aware 的静态背景编码，以增强模型对地理和土壤背景的利用能力；  
（3）在中国东部这一具有实际应用需求且具备一定水文异质性的区域内，系统评估该框架在根区土壤湿度短期预测中的有效性，为后续扩展到更广区域和跨气候带泛化奠定基础。

## 这个实验到底在做什么

这个项目不是简单地把上游 GPTCast 的降水雷达输入换成了 ERA5-Land 数据，而是在尝试把一个基于 latent token 和自回归预测的生成式框架，迁移到更有水文意义的目标上：

- 预测未来几天的表层土壤湿度 `swvl1`
- 预测未来几天的根区土壤湿度 `rzsm_0_100cm`
- 进一步扩展到根区储水量 `soil_water_storage_0_100cm_mm`

从当前代码路线看，最核心的方向是：

- 用过去一段时间的土壤状态作为 `state`
- 用降水、蒸散发、径流、温度、辐射等作为 `forcing`
- 预测未来 `D+1 ~ D+7` 的根区水分状态

因此，这个实验的本质不是“天气图像生成”，而是“面向水文状态变量的短期预测”。

这里要进一步区分第一阶段和第二阶段：

- **第一阶段**不是完整意义上的水文建模，而是**面向水文状态变量的表示学习**  
  它的目标是把 `swvl1` / `rzsm_0_100cm` 这样的状态场压缩为稳定的 token 表示，并尽可能高质量地重建回来。
- **第二阶段**才更接近水文建模  
  因为它显式建模了 `state + forcing (+ physical context)` 到未来状态的演化关系。

因此，在论文里更安全的表述应该是：

> The first stage learns a compact latent representation for hydrologically meaningful land-state variables, whereas the second stage performs the actual state evolution forecasting.

## 它现实中可以预测什么

如果这条路线最终跑通，它最直接能提供的是未来几天的土壤湿度背景场，特别是根区土壤湿度背景场。

这意味着它可以服务于以下实际问题：

1. 农业灌溉和作物水分管理

- 判断未来几天根区是否持续偏干
- 判断是否需要提前灌溉
- 识别作物潜在水分胁迫风险

2. 干旱和闪旱监测

- 降水只是输入，真正影响植被和农业的是土壤中可利用的水
- 根区湿度比表层湿度更接近农业干旱监测的真实需求

3. 洪水、山洪和滑坡的前期湿润背景分析

- 很多灾害并不是单纯由一次降雨大小决定
- 更关键的是降雨来临前土壤是否已经接近饱和
- 根区储水和前期湿润度能更好反映产流和失稳风险背景

4. 流域水文模拟和预报的背景状态

- 土壤湿度是流域水文过程中的关键中间状态
- 如果能提前预测土壤湿度背景，就能为后续 runoff 或 streamflow 预报提供更合理的初始条件

## 为什么值得做这个实验

## 1. 土壤湿度是连接天气 forcing 和地表响应的关键状态量

很多应用场景关心的不是“明天会不会下雨”，而是“未来几天地里还有多少水”“土壤是不是会继续变干”“土壤是不是已经湿到容易产流”。

从这个角度说：

- 降水是 forcing
- 土壤湿度是状态
- 产流、蒸散和农业影响是响应

所以，预测土壤湿度比单纯预测降水图像更接近水文和农业应用的真实目标。

## 2. 根区土壤湿度比表层湿度更有应用价值

只做 `swvl1` 时，模型学到的更多是近地表浅层湿润信号。

而 `rzsm_0_100cm` 更接近：

- 作物根系可利用水
- 土壤水分记忆
- 流域前期湿润程度
- 陆气相互作用中的慢变量

因此，根区土壤湿度比表层湿度更适合支撑“水文意义”与“应用意义”的论文叙事。

## 3. state + forcing 的结构比纯图像外推更接近过程机理

如果只做 persistence，本质上只是利用土壤水分的时间记忆。

如果只做纯图像 GPT，本质上仍然偏向统计外推。

而当前项目正在构建的是：

- 过去状态 `state`
- 外部驱动 `forcing`
- 未来状态 `future state`

这样的结构更接近一个简化的水文动力系统 surrogate model。虽然它不是显式物理模型，但它至少在建模结构上已经比“纯像素预测”更贴近真实陆面过程。

## 这个实验相对上游 GPTCast 的提升在哪里

上游 GPTCast 的应用对象是雷达降水 nowcasting，本质上是天气图像序列的 token 化和生成。

而当前项目的迁移，不应被理解成简单的“换数据集”，而应该被理解成三个层次的升级：

1. 预测目标升级

- 从 precipitation image forecasting
- 变成 hydrologically meaningful state forecasting

2. 变量语义升级

- 从单一图像序列
- 变成 `state + forcing` 的耦合系统

3. 应用落点升级

- 从天气外推
- 变成农业、水资源和流域过程相关的状态变量预测

这也是这个项目真正具备科研价值的原因。

## 论文里可以怎么说它的研究意义

如果以后写论文，引言里可以把意义概括成下面这段逻辑：

1. 土壤湿度，尤其是根区土壤湿度，是连接降水 forcing、陆面储水状态、蒸散发反馈和产流风险的重要中间变量。
2. 与短时降水预测相比，根区土壤湿度预测更贴近农业干旱、水资源调度和流域水文风险分析的实际需求。
3. 当前深度学习时空预测框架大多集中在 precipitation nowcasting 或通用图像序列预测，对“水文状态变量”的生成式建模与解释仍然不足。
4. 本项目尝试把 latent token + autoregressive forecasting 的生成式框架从降水雷达场迁移到 ERA5-Land 土壤湿度任务，并引入 forcing conditioning 与更具物理意义的目标变量，从而探索一种更接近陆面过程的短期土壤水分预测路径。

## 这个实验当前最合理的论文定位

现阶段最合理的定位不是：

- “我们用 GPT 做了一个新的 AI 模型”

而是：

- “我们探索了一种面向根区土壤湿度预测的、具备 state + forcing 耦合结构的生成式时空预测框架”

更具体一点，可以概括为：

> 这是一个尝试从 weather image forecasting 走向 hydrologic state forecasting 的实验。

## 当前边界也要说清楚

为了避免把项目表述得过度，下面这些边界也应该明确：

- 它目前仍然是数据驱动模型，不是显式物理水文模型
- 它学到的是统计意义上的状态演化规律，不等于严格满足水量平衡
- 它当前最有希望服务的是短期状态预测和背景场预估，而不是替代完整流域模拟器
- 它的真正价值，取决于后续是否能证明：
  - 比 persistence 更强
  - 比单纯表层目标更有意义
  - forcing conditioning 确实带来增益
  - 输出结果与已知水文过程相符

## 为什么选择中国东部作为当前实验区域

这一点在论文里一定会被问到，而且属于合理的审稿问题。  
因此，选择中国东部不能只解释成“数据方便”或“计算量更容易控制”，而应该被明确表述成一个**应用驱动、区域异质性明确、且适合作为方法验证场景**的科学选择。

### 1. 中国东部具有直接而强烈的应用需求

中国东部同时面对以下几类与根区土壤湿度高度相关的实际问题：

- 农业灌溉与作物水分胁迫监测
- 干旱与闪旱风险识别
- 强降雨条件下的前期湿润背景评估
- 洪涝、山洪、滑坡等灾害的 antecedent wetness 分析

也就是说，这一区域并不是随意选取的“方便样区”，而是根区土壤湿度短期预测本身就具有现实意义的区域。

### 2. 中国东部并不是一个完全同质的区域

虽然当前研究把中国东部作为一个统一实验区，但该区域内部并不单一，而是包含了明显不同的下垫面与水文地理背景，例如：

- 华北平原
- 长江中下游地区
- 东南丘陵与山地边缘

这意味着区域内部已经具有：

- 不同湿润度条件
- 不同地形背景
- 不同土壤与下垫面特征
- 不同前期湿润度与 forcing 响应模式

因此，这个实验虽然不是全国或全球尺度研究，但也不是在一个完全均一、缺乏挑战的小范围内完成的。

### 3. 作为第一步工作，区域聚焦比一开始做全国/全球更合理

当前论文的核心目标不是直接证明模型可以无条件泛化到所有气候区，而是先回答一个更基础、更重要的问题：

> 一个生成式 `state + forcing + physical context` 框架，能否在一个具有明确应用需求且内部存在一定水文异质性的区域内，稳定地预测未来几天的根区土壤湿度状态？

如果这个问题都没有在一个清晰可控的区域内回答好，那么直接扩展到全国甚至全球，反而只会让：

- 数据链更复杂
- 变量差异更大
- 气候态变化更强
- 结论更难解释

因此，把中国东部作为当前研究区域，更符合“先验证框架有效性，再讨论更广泛泛化能力”的正常研究路径。

### 4. 在论文里最安全的写法

如果后面需要在论文引言、研究区介绍或讨论中解释这一点，最稳妥的中文表述可以直接写成：

> 本研究选择中国东部作为实验区域，主要基于三点考虑。  
> 第一，该区域农业生产、水资源管理以及洪涝和干旱风险评估对根区土壤湿度的短期预测具有直接需求。  
> 第二，虽然本文将其视为一个统一研究区，但区域内部同时包含平原、河湖密集区和丘陵山地边缘等不同下垫面与湿润度条件，能够为方法提供一定程度的水文异质性检验。  
> 第三，作为面向水文状态预测的第一步工作，区域聚焦有助于在保持数据定义和任务设置一致的前提下，更清晰地评估 `state + forcing + physical context` 框架本身的有效性。  
> 因此，本文应被理解为一个区域尺度、方法验证导向的研究，而不是对全国或全球泛化能力的直接宣称。

### 5. 这一段的真正作用

把这段写进研究意义或论文引言，有两个直接好处：

1. 它能帮助审稿人理解：
   - 你不是随便挑了一个区域
   - 而是在一个有现实意义且内部已有一定复杂性的区域里验证方法

2. 它也能主动限制 claim：
   - 当前工作是区域尺度验证
   - 更广泛的跨气候带泛化能力属于后续工作

这比在主文里含糊其辞，或者被审稿人追问后再被动解释，要稳得多。

## 一句话总结

这个实验之所以值得做，是因为它把研究目标从“预测天气图像”推进到了“预测有水文意义的土壤状态”，而根区土壤湿度正是农业干旱、陆面储水和流域风险分析中最关键的状态变量之一。

## 从历史研究和应用需求看，这个实验为什么有必要

从历史上看，土壤湿度研究经历了一个非常清晰的演进路径：

1. 先是依赖站点观测
2. 后来发展到陆面模式和再分析产品
3. 再到基于遥感和机器学习的大尺度估计
4. 现在才逐步进入“面向未来几天的土壤状态预测”

这条演进路径本身说明了一件事：土壤湿度不是一个边缘变量，而是被越来越多研究和业务系统视为关键状态量。

例如，Drought.gov 的官方页面明确指出，土壤湿度在农业监测、干旱和洪水预报、森林火险和供水管理中都具有重要作用，并且常常能在传统指标触发之前提供预警信号。  
官方来源：<https://www.drought.gov/topics/soil-moisture>

而 ECMWF 对 ERA5-Land 的官方说明和数据论文也明确强调，ERA5-Land 之所以重要，正是因为它以较高空间和时间分辨率持续描述陆面水热过程，可服务于水资源、土地和环境管理等应用。  
ECMWF 官方说明：<https://www.ecmwf.int/en/era5-land>  
数据论文 DOI：<https://doi.org/10.5194/essd-13-4349-2021>

如果从更偏水文过程和应用端的文献看，这个必要性会更明显。

- Ghannam et al. (2016) 讨论了 root-zone soil moisture 的记忆与持续性，指出根区湿度是季节尺度陆气相互作用和预报中的关键慢变量。  
  DOI：<https://doi.org/10.1002/2015WR017983>
- Ran et al. (2022) 在长江中下游流域表明，随着流域面积增大，洪水生成机制会从降雨主导逐步转向前期土壤湿度更重要。  
  DOI：<https://doi.org/10.5194/hess-26-4919-2022>
- Webb et al. (2025) 在美国西海岸大气河流域中发现，湿润的前期土壤条件会显著放大洪峰流量，许多流域存在明显的 antecedent soil moisture threshold。  
  DOI：<https://doi.org/10.1175/JHM-D-24-0078.1>

这些研究共同说明：很多现实问题并不是单纯由“未来降水多少”决定，而是由“未来几天土壤会保持多湿、多干、还是继续变湿”决定。

## 别人是怎么研究这个场景的

现有研究大体可以分成 4 类。

### 1. 站点观测和经验方法

最传统的方法是依赖原位观测站、探针和经验阈值分析。

优点：

- 精度高
- 物理意义明确

缺点：

- 空间覆盖差
- 难以支持区域尺度应用
- 很难直接生成连续空间场

### 2. 再分析和陆面模式

以 ERA5-Land 为代表的陆面再分析提供了长期连续、物理一致性较强的土壤湿度背景场。

优点：

- 时空连续
- 变量体系完整
- 适合作为背景状态与训练数据

缺点：

- 更像“状态重建”或“背景产品”
- 不是专门针对某个区域未来 1–7 天状态预测优化出来的
- 分辨率、误差结构和区域适应性仍然受限

代表文献：

- Muñoz-Sabater et al. (2021), *ERA5-Land: a state-of-the-art global reanalysis dataset for land applications*  
  DOI：<https://doi.org/10.5194/essd-13-4349-2021>

### 3. 遥感反演与根区湿度估计

由于卫星更容易感知表层湿度，很多研究集中在“如何从表层信息反推出根区湿度”。

Li et al. (2023) 的综述把这一类方法总结为：

- empirical
- semi-empirical
- physics-based
- machine learning

并指出 RZSM 在农业、生态和水文上很关键，但大尺度、高分辨率、深层根区湿度仍然难以直接获得。  
DOI：<https://doi.org/10.3390/rs15225361>

此外，也有工作使用深度学习来做 RZSM 估计，而不是预测。例如：

- *Root-zone soil moisture estimation based on remote sensing data and deep learning*  
  DOI：<https://doi.org/10.1016/j.envres.2022.113278>

这类方法的重要贡献在于：

- 证明表层和根区之间可以通过数据驱动方式建立联系

但其主要问题是：

- 多数工作聚焦于估计/反演，而不是未来多日预测
- 很多工作不是直接做空间场的生成式预测

### 4. 机器学习与深度学习预测

近年来越来越多研究开始做土壤湿度预测，而不只是估计。

例如：

- Cai et al. (2019) 用深度学习预测 soil moisture，强调其对农业生产和水资源管理的重要性。  
  DOI：<https://doi.org/10.1371/journal.pone.0214508>
- Ahmed et al. (2021) 用 CNN 和 GRU 进行土壤湿度预测。  
  DOI：<https://doi.org/10.3390/rs13040554>
- Xu et al. (2023) 结合深度学习和 S2S 模型做 surface/root-zone soil moisture forecasting，指出更长 lead time 的土壤湿度预报对农业和风险准备有价值。  
  DOI：<https://doi.org/10.3390/rs15133410>
- Wang et al. (2024) 对 deep learning soil moisture prediction 做了系统研究。  
  DOI：<https://doi.org/10.5194/hess-28-917-2024>
- Zheng et al. (2024) 提出了 GRU-Transformer 做 root-zone soil moisture forecasting。  
  DOI：<https://doi.org/10.3390/agronomy14030432>

这说明“用深度学习做土壤湿度预测”已经不是空白方向，但问题仍然很多。

## 现有方法的主要问题在哪里

虽然已有大量研究，但从“短期、区域尺度、根区状态、可解释预测”这个组合目标来看，现有方法仍有明显不足。

### 1. 很多工作做的是估计，不是预测

不少工作重点是：

- 从表层遥感估计根区湿度
- 从多源产品融合出当前根区状态

这类工作对监测很有意义，但不等于你现在的目标。你的目标是：

- 给定过去状态和驱动
- 预测未来几天状态如何演化

这是更接近业务决策的问题。

### 2. 很多预测工作是点位级、站点级或表格型

很多 ML / DL 土壤湿度研究是在：

- 单站点
- 少量站点
- 时序表格输入

上做的。

这类研究的问题是：

- 缺少空间结构
- 难以表达区域尺度湿润/干燥传播模式
- 很难直接服务空间分布图、流域湿润背景或区域农业管理

### 3. 很多工作仍然偏表层，不够根区

表层湿度当然重要，但农业和流域应用更关心的是根区湿度与储水状态。

根区状态更接近：

- 植物可利用水
- 陆面水分记忆
- 前期湿润度
- 产流敏感背景

所以只做 surface soil moisture，应用意义是有限的。

### 4. 很多方法更像统计拟合，缺少 state + forcing 耦合结构

现有很多方法虽然可以预测，但往往没有明确区分：

- 当前状态
- 外部 forcing
- 未来状态

而这个区分非常关键，因为土壤湿度本质上是一个受 forcing 驱动、带有强记忆的状态变量。

### 5. 泛化性问题依然突出

已有研究已经注意到，土壤湿度与：

- 地形
- 土壤属性
- 植被
- 气候区

高度相关。

例如 Celik et al. (2022) 在深度学习土壤湿度预测中已经把 climate、soil texture 和 topography 一并纳入。  
DOI：<https://doi.org/10.3390/rs14215584>

这从侧面说明：如果模型只学局部统计关系，而不理解下垫面和地理背景，它的跨区域泛化通常会受限。

## 我们这个实验试图解决什么问题

你当前这条 SoilCast（GPTCast -> ERA5-Land）路线，最值得强调的不是“用 GPT”，而是你在尝试同时解决以下几个痛点：

1. 从估计走向预测  
- 不只是恢复当前根区湿度，而是预测未来 `D+1 ~ D+7`

2. 从表层走向根区  
- 不再停留在 `swvl1`
- 目标推进到 `rzsm_0_100cm` 甚至 `storage`

3. 从点位/表格走向空间分布场  
- 直接预测区域尺度状态图

4. 从纯 persistence / 纯统计外推，走向 `state + forcing` 耦合  
- 让模型在结构上更像一个水文状态演化代理模型

5. 从“只有数值 skill”走向“更有水文解释意义”  
- 后续可以继续加入物理监控、物理辅助 loss、地形静态信息等

## 做成以后可以改变什么

如果这项工作真正做成，它最现实的改变不是“替代所有物理水文模型”，而是提供一个更快、更细、更贴近决策需求的土壤状态预测层。

它可能带来的改变包括：

### 1. 农业管理更早决策

- 提前几天知道根区是否会持续偏干
- 辅助灌溉时机和强度判断
- 更早识别作物水分胁迫风险

### 2. 干旱监测更接近实际受灾过程

- 不再只看降水异常
- 而是看植物真正能利用的根区水

### 3. 洪水、山洪和滑坡风险判断更完整

- 不只看即将到来的降雨
- 还看前期土壤是否已经湿润到接近临界状态

### 4. 流域模型和业务预报有更好的背景场

- 为后续 runoff / streamflow 预报提供更合理的初始土壤状态
- 改善短期风险评估链条中的陆面背景信息

## 这一页在论文里最适合支持什么论点

如果以后要把这部分写进论文引言，这一页最适合支撑下面这条主线：

1. 土壤湿度，尤其是根区土壤湿度，是连接天气 forcing 和水文/农业响应的关键状态量。
2. 现有大量研究已经证明其在干旱、农业、水资源和洪水风险中的重要作用。
3. 现有研究虽然已有再分析、遥感估计和深度学习方法，但仍缺乏对“未来几天根区土壤状态”的高分辨率、空间化、state + forcing 耦合预测。
4. 因此，把生成式时空预测框架从 precipitation nowcasting 迁移到 hydrologic state forecasting 是有现实意义且具有研究空缺的。

## 可直接引用的参考文献与 DOI

下面这些 DOI 在写论文或开题时可以直接作为真实参考文献使用：

- Muñoz-Sabater, J. et al. (2021). *ERA5-Land: a state-of-the-art global reanalysis dataset for land applications*. Earth System Science Data, 13, 4349-4383.  
  DOI: <https://doi.org/10.5194/essd-13-4349-2021>

- Ghannam, K. et al. (2016). *Persistence and memory timescales in root-zone soil moisture dynamics*. Water Resources Research, 52, 1427-1445.  
  DOI: <https://doi.org/10.1002/2015WR017983>

- Ran, Q. et al. (2022). *The relative importance of antecedent soil moisture and precipitation in flood generation in the middle and lower Yangtze River basin*. Hydrology and Earth System Sciences, 26, 4919-4931.  
  DOI: <https://doi.org/10.5194/hess-26-4919-2022>

- Webb, M. J. et al. (2025). *Wet antecedent soil moisture increases atmospheric river streamflow magnitudes non-linearly*. Journal of Hydrometeorology.  
  DOI: <https://doi.org/10.1175/JHM-D-24-0078.1>

- Li, M., Sun, H., and Zhao, R. (2023). *A Review of Root Zone Soil Moisture Estimation Methods Based on Remote Sensing*. Remote Sensing, 15(22), 5361.  
  DOI: <https://doi.org/10.3390/rs15225361>

- Cai, Y. et al. (2019). *Research on soil moisture prediction model based on deep learning*. PLOS ONE, 14(4), e0214508.  
  DOI: <https://doi.org/10.1371/journal.pone.0214508>

- Ahmed, A. A. M. et al. (2021). *Deep Learning Forecasts of Soil Moisture: Convolutional Neural Network and Gated Recurrent Unit Models Coupled with Satellite-Derived MODIS, Observations and Synoptic-Scale Climate Index Data*. Remote Sensing, 13(4), 554.  
  DOI: <https://doi.org/10.3390/rs13040554>

- Xu, L. et al. (2023). *Hybrid Deep Learning and S2S Model for Improved Sub-Seasonal Surface and Root-Zone Soil Moisture Forecasting*. Remote Sensing, 15(13), 3410.  
  DOI: <https://doi.org/10.3390/rs15133410>

- Wang, Y. et al. (2024). *A comprehensive study of deep learning for soil moisture prediction*. Hydrology and Earth System Sciences, 28, 917-943.  
  DOI: <https://doi.org/10.5194/hess-28-917-2024>

- Zheng, W. et al. (2024). *GRU-Transformer: A Novel Hybrid Model for Predicting Soil Moisture Content in Root Zones*. Agronomy, 14(3), 432.  
  DOI: <https://doi.org/10.3390/agronomy14030432>

- Celik, M. F. et al. (2022). *Soil Moisture Prediction from Remote Sensing Images Coupled with Climate, Soil Texture and Topography via Deep Learning*. Remote Sensing, 14(21), 5584.  
  DOI: <https://doi.org/10.3390/rs14215584>

- Franch, G. et al. (2025). *GPTCast: a weather language model for precipitation nowcasting*. Geoscientific Model Development, 18, 5351-5371.  
  DOI: <https://doi.org/10.5194/gmd-18-5351-2025>

## 当前第二阶段 physical-context-aware 设计参考了哪些论文

这一部分必须说清楚，否则很容易在论文里把“受文献启发”写成“直接复现”。

### 直接参考的主线设计

当前第二阶段主线不是 homemade physics loss，也不是显式质量守恒约束。  
主线的核心思想是：

- 保留生成式 `state + forcing` 预测框架
- 把与水文过程直接相关的 forcing 输入显式接入
- 进一步引入更有物理意义的静态背景信息（地理、地形、土壤）
- 让模型在结构上区分动态驱动和静态背景，而不是把所有信息混成同一种输入

这个思路和下列文献最一致：

1. **Lesinger & Tian (2025, Nature Communications)**  
   这篇工作最关键的启发不是“加一个物理损失函数”，而是：
   - 用历史 RZSM 观测
   - 用大气预测因子
   - 用 dynamic model guidance
   共同构成一个 hybrid 输入设计

   对我们当前项目最重要的启发是：
   > 物理一致性更应来自输入设计和变量组织，而不是 ad hoc 的训练惩罚项。

   需要严格说明的是：  
   我们**没有**直接复现它的 weekly / subseasonal forecasting 范式，也**没有**复现 ECMWF/GEFS guidance 路线；我们只是吸收了它“用更物理的条件输入组织模型”的思想。  
   DOI: <https://doi.org/10.1038/s41467-025-62761-3>

2. **Wi & Steinschneider (2024, HESS)**  
   这篇工作更偏方法论层面，强调 hydrologic deep learning 的物理真实性，不应只依赖后验解释，而应更多来自：
   - 结构设计
   - 输入变量设计
   - 物理合理的学习偏置

   它对我们当前 second stage 的直接意义是：
   - separate dynamic/static conditioning 比“一股脑拼接所有变量”更符合水文建模直觉
   - static-aware 分支是比随手加硬约束更稳的主线增强方向

   DOI: <https://doi.org/10.5194/hess-28-479-2024>

3. **Celik et al. (2022, Remote Sensing)**  
   这篇工作明确把：
   - climate
   - soil texture
   - topography
   一起接入深度学习模型做 soil moisture prediction。

   它支持我们当前把 static-aware 从：
   - `land_mask + latitude_norm + longitude_norm`
   逐步升级到：
   - `sand_fraction`
   - `clay_fraction`
   - `silt_fraction`
   - `porosity`
   - 后续再接 `DEM / slope / aspect`

   DOI: <https://doi.org/10.3390/rs14215584>

4. **Scientific Data 2023 / 2025 的 multilayer soil moisture 数据工作**  
   这些工作的重要价值不在于提供 forecasting 模型，而在于反复说明：
   - 地形
   - 土壤性质
   - bedrock
   - water-table depth
   对不同深度层的土壤湿度分布具有实质影响。

   它们给了我们把 second stage 做成 terrain/soil-aware 的合理依据。  
   DOI: <https://doi.org/10.1038/s41597-023-02011-7>  
   DOI: <https://doi.org/10.1038/s41597-025-06436-0>

### 文献启发但当前没有直接实现的方向

下面这些文献对后续 physical-aware 增强有启发，但**不应在当前主线里被写成“已经实现”**。

1. **Zhang et al. (2025, HESS)**  
   启发点是单调性/方向性软约束。  
   这类思想可以转化成：
   - 强湿润 forcing 下不应预测显著变干
   - 强干燥 forcing 下不应预测显著变湿

   但目前本仓库主线**尚未正式引入**这类 soft monotonicity loss。  
   DOI: <https://doi.org/10.5194/hess-29-5955-2025>

2. **PhySoilNet (2024)**  
   启发点是 boundary constraint，也就是把超过物理范围的土壤湿度预测扣分。  
   这类设计对以后加入 `porosity / field_capacity / wilting_point` 后的软边界约束有启发。  
   但当前主线**尚未正式实现** pixel-wise boundary loss。  
   DOI: <https://doi.org/10.1016/j.jag.2024.104147>

3. **Wang et al. (2024, HESS)**  
   启发点是：与其在训练目标里硬加复杂约束，不如先把输入、区域、变量和评估体系做对。  
   这进一步支持当前项目的路线：
   - 先做 clean baseline
   - 再做 physical-context-aware static enhancement
   - 最后才考虑是否需要更强的物理正则

   DOI: <https://doi.org/10.5194/hess-28-917-2024>

### 论文里最安全的写法

如果后面要在论文方法部分概括当前 second stage，最稳妥的说法应该是：

> The second-stage enhancement in this project is **physical-context-aware**, not physics-constrained in the strict sense.  
> It is primarily motivated by literature showing that hydrologically relevant forcing variables, terrain context, and soil-property background are key determinants of soil moisture evolution.  
> In the current implementation, these insights are incorporated through explicit `state + forcing` conditioning and a separate static-context encoding branch, rather than through hard conservation constraints or ad hoc physical penalties.

翻成中文就是：

> 当前第二阶段增强版的核心定位是“physical-context-aware”，而不是严格意义上的 physics-constrained。  
> 它主要参考的是：水文相关 forcing、地形背景、土壤属性对土壤湿度演化具有决定作用，因此我们通过显式的 `state + forcing` 建模和独立的 static-context 分支，把这些物理背景信息接入模型，而不是通过硬守恒或自定义物理惩罚项来强行约束训练。

### 这一部分为什么值得写进 RESEARCH_SIGNIFICANCE

因为这会直接影响审稿人如何理解你的贡献边界：

- 不是“我们发明了一个新的 physics loss”
- 而是“我们把生成式 token forecasting 框架组织成了更接近陆面过程的 `state + forcing + physical context` 结构”

这对当前项目的定位其实更强，也更稳。

## 第一阶段 PHuber 设计到底有没有起作用

这个问题值得单独说清，因为很容易被误解成“我们只是随手换了个 loss 名字”。

当前 `vae_phuber_rzsm` 配置的关键超参数是：

- `pixel_loss: phuber`
- `pixelloss_weight: 10.0`
- `perceptual_weight: 0.1`
- `phuber_delta: 0.03`
- `grad_weight: 0.25`
- `range_weight: 0.10`

这些参数的物理含义是：

1. **不是在归一化后的 token 空间里算误差，而是在物理空间里算**

代码里 `phuber(...)` 会先把输入从 `[-1, 1]` 反变换回 `0.0 ~ 0.8 m3/m3` 的物理空间，再计算 Huber、梯度误差和越界惩罚。  
这意味着：

- “数值很小所以权重不起作用”的风险，比直接在归一化空间里要小得多
- `delta=0.03` 的尺度也是按物理湿度单位解释的，而不是按 token 归一化值解释的

2. **基础误差项是主导项，梯度项是辅助项，范围项是轻约束**

当前实现等价于：

> `base huber + 0.25 * gradient huber + 0.10 * range penalty`

因此：

- `base` 一定是主要贡献
- `gradient` 会稳定地影响优化，但不会压过主重建项
- `range_penalty` 只有在预测超出物理合理范围时才会显著发挥作用

3. **哪一项可能“看起来弱”？**

最可能显得弱的是 `range_weight = 0.10` 这一项。  
原因不是“数值太小”，而是：

- 如果大多数预测本来就在 `0 ~ 0.8` 范围内
- 那越界惩罚本来就该很小

这说明它是一个 guardrail，而不是主优化来源。  
从设计上说，这是合理的。

4. **哪一项最值得保留？**

当前最有价值的是：

- 在物理空间算 `Huber`
- 加入轻量的 `gradient consistency`

这两项都能直接帮助：

- 抑制少数极端异常误差
- 保住湿干边界和空间结构

也正因此，在我们第一阶段的统一评估里，`vae_phuber_rzsm` 不只是 `RMSE` 更低，`grad-MAE` 也更低。

### 最安全的结论

因此，更准确的判断不是：

- “这些权重会不会完全没用”

而是：

- **主重建项一定在起作用**
- **梯度项大概率在起作用，并且是值得保留的**
- **范围项本来就是轻约束，如果多数样本不越界，它小是正常的**

如果以后真的要继续优化第一阶段，这一部分最可能值得微调的是：

- `phuber_delta`
- `grad_weight`

而不是先怀疑“权重完全没起作用”。
