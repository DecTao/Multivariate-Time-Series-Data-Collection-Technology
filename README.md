# Multivariate-Time-Series-Data-Collection-Technology

## 项目简介

这是一个关于多变量时间序列数据收集技术的项目，旨在提供高效、准确的多变量时间序列数据收集工具，适用于数据分析、机器学习、时间序列预测等场景。

## 目录结构

```
Multivariate-Time-Series-Data-Collection-Technology/
├── 代码/                   # 项目源代码
│   ├── approximate_algorithm.py       # 近似算法实现
│   ├── data.py                       # 数据预处理相关代码
│   ├── dataset_tester.py             # 数据集评估工具
│   ├── electric_cleaned.py           # 电力数据清洗代码
│   ├── exact_algorithm.py            # 精确算法实现
│   ├── optimized_exact_algorithm.py  # 优化后的精确算法实现
│   ├── select_attributes.py          # 属性选择工具
│   └── test_dataset.py               # 数据集测试脚本
├── 数据/                   # 数据集文件
│   ├── DSMTS.csv
│   ├── ETTh1.csv
│   ├── FRED.csv
│   ├── Portland_weather_merged.csv
│   ├── electric_cleaned.csv
│   ├── exchange_rate.csv
│   ├── national_illness.csv
│   ├── electricity.csv
│   ├── traffic.csv
│   └── weather.csv
├── 结果/                   # 数据处理和分析结果
│   ├── autocorr_plots/              # 自相关图
│   ├── DSMTS_corr_heatmap.png
│   ├── DSMTS_result.txt
│   ├── ETTh1_corr_heatmap.png
│   ├── ETTh1_result.txt
│   ├── FRED_5_result.txt
│   ├── FRED_9_result.txt
│   ├── FRED_corr_heatmap.png
│   ├── FRED_result.txt
│   ├── Portland_weather_merged_corr_heatmap.png
│   ├── Portland_weather_merged_result.txt
│   ├── traffic_corr_heatmap.png
│   ├── traffic_result.txt
│   ├── electric_cleaned_5_result.txt
│   ├── electric_cleaned_7_result.txt
│   ├── electric_cleaned_9_result.txt
│   ├── electric_cleaned_corr_heatmap.png
│   ├── electric_cleaned_result.txt
│   ├── national_illness_corr_heatmap.png
│   └── national_illness_result.txt
└── 适合度评分框架.png        # 适合度评分框架图
```

## 数据集说明

  * **DSMTS.csv** ：风力涡轮数据集。
  * **ETTh1**.csv ：工业设备运行状态的数据集。
  * **FRED.csv** ：经济数据集。
  * **Portland_weather_merged.csv** ：波特兰天气数据集，包含多个天气相关变量。
  * **electric_cleaned.csv** ：清洗后的电力数据集。
  * **national_illness.csv** ：全国疾病数据集。
  * **traffic.csv** ：交通流量数据集。
  * **electricity.csv** ：电力数据集。

## 使用方法

  1. 克隆项目仓库：`git clone https://github.com/DecTao/Multivariate-Time-Series-Data-Collection-Technology.git`
  2. 安装项目依赖：`pip install -r requirements.txt`
  3. 运行数据收集和处理代码：可以运行代码目录下的 Python 脚本，例如 `python dataset_tester.py` 评估数据集是否适用于本算法，或`python test_dataset.py` 进行算法测试等。
  4. 查看结果：处理结果会保存在结果目录下，例如相关性热力图（如 `DSMTS_corr_heatmap.png`）和结果文本文件（如 `DSMTS_result.txt`）。

## 项目用途

  * 用于收集多变量时间序列数据，可应用于数据分析、机器学习模型训练、时间序列预测等场景。

## 技术实现

项目采用 Python 编写，主要使用了以下技术栈：

  * Python 3.8
  * NumPy：用于数值计算
  * Pandas：用于数据处理和分析
  * Matplotlib：用于数据可视化（生成热力图等）

## 贡献指南

  * 欢迎提交议题（Issues）报告问题或提出建议。
  * 如果要贡献代码，请先创建一个新的分支，然后提交 Pull Request。
