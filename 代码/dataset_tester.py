import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import statsmodels.api as sm
import os

def test_dataset_suitability(file_path, time_column, time_format, use_first_row_as_header=False, granger_lag=5,result_dir="results"):
    """测试数据集是否适合时间序列收集算法，添加了Granger因果检验。"""
    os.makedirs(result_dir, exist_ok=True) 
    base_name = os.path.splitext(os.path.basename(file_path))[0] 
    results={}
    # 1. 加载数据
    try:
        data = pd.read_csv(file_path).iloc[:,:10]
        if use_first_row_as_header:
            data.columns = data.iloc[0]
            data = data.iloc[1:]        
        print(f"\n{'='*50}")
        print(f"数据集基本信息:")
        print(f"总记录数: {len(data)}")
        print(f"列数: {len(data.columns)}")
        print(f"列名: {list(data.columns)}")
        
        # 2. 预处理时间序列
        data[time_column] = pd.to_datetime(data[time_column], format=time_format)
        data.set_index(time_column, inplace=True)
        data_numeric = data.select_dtypes(include=np.number)
        
        # 3. 检查数据完整性
        missing_ratio = data_numeric.isnull().mean()
        print("\n缺失值比例:")
        print(missing_ratio)
        
        # 4. 检查时间连续性
        time_diffs = data_numeric.index.to_series().diff().dropna()
        time_stats = {
            'min_interval': time_diffs.min(),
            'max_interval': time_diffs.max(),
            'median_interval': time_diffs.median(),
            'interval_consistency': (time_diffs.nunique() == 1)
        }
        print("\n时间间隔统计:")
        print(f"最小间隔: {time_stats['min_interval']}")
        print(f"最大间隔: {time_stats['max_interval']}")
        print(f"中位数间隔: {time_stats['median_interval']}")
        print(f"间隔是否一致: {'是' if time_stats['interval_consistency'] else '否'}")
        
        # 5. 平稳性测试
        stationary_results = {}
        for col in data_numeric.columns:
            print(f"\n=== 列 '{col}' 的平稳性测试 ===")
            # ADF检验
            adf_result = adfuller(data_numeric[col].dropna(), autolag='AIC')
            # KPSS检验
            kpss_result = kpss(data_numeric[col].dropna(), regression='c', nlags="auto")            
            stationary_results[col] = {
                'ADF_statistic': adf_result[0],
                'ADF_pvalue': adf_result[1],
                'KPSS_statistic': kpss_result[0],
                'KPSS_pvalue': kpss_result[1],
                'is_stationary': (adf_result[1] < 0.05) and (kpss_result[1] > 0.05)
            }            
            print(f"ADF统计量: {adf_result[0]:.4f}, p值: {adf_result[1]:.4f}")
            print(f"KPSS统计量: {kpss_result[0]:.4f}, p值: {kpss_result[1]:.4f}")
            print(f"是否平稳: {'是' if stationary_results[col]['is_stationary'] else '否'}")
        
        # 6. 自相关性和季节性分析
        for col in data_numeric.columns:
            print(f"\n=== 列 '{col}' 的自相关性和季节性分析 ===")
            plt.figure(figsize=(12, 6))
            plot_acf(data_numeric[col].dropna(), lags=40, ax=plt.gca())
            plt.title(f"'{col}' 自相关函数(ACF)")
            plt.show()            
            plt.figure(figsize=(12, 6))
            plot_pacf(data_numeric[col].dropna(), lags=40, ax=plt.gca())
            plt.title(f"'{col}' 偏自相关函数(PACF)")
            plt.show()
            
            # 7. 季节性检测
            try:
                seasonal_period = detect_seasonal_period(data_numeric[col])
                # STL分解可视化
                stl = STL(data_numeric[col].dropna(), period=seasonal_period, robust=True)
                res = stl.fit()
                plt.figure(figsize=(12, 8))
                res.plot()
                plt.suptitle(f"'{col}' 的STL分解")
                plt.close()
            except Exception as e:
                print(f"季节性分析失败: {str(e)}")
        
        # 8. 多变量相关性分析
        ("变量间相关性矩阵")
        corr_matrix = data_numeric.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("heatmap")
        heatmap_path = os.path.join(result_dir, f"{base_name}_corr_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        results['heatmap_path'] = heatmap_path

        # 9. Granger因果检验
        for col1 in data_numeric.columns:
            for col2 in data_numeric.columns:
                if col1 != col2:
                    result = grangercausalitytests(data_numeric[[col1, col2]].dropna(), maxlag=granger_lag, verbose=False)
        
        # 10. 计算适合度评分 (0-1)
        score = 0
        total_possible = 0
        
        # 数据完整性评分 (权重: 0.3)
        completeness_score = 1 - missing_ratio.mean()
        score += completeness_score * 0.3
        total_possible += 0.3
        
        # 时间连续性评分 (权重: 0.2)
        continuity_score = 1 if time_stats['interval_consistency'] else 0.5
        score += continuity_score * 0.2
        total_possible += 0.2
        
        # 平稳性评分 (权重: 0.2)
        stationary_cols = sum(1 for col in stationary_results if stationary_results[col]['is_stationary'])
        stationary_score = stationary_cols / len(data_numeric.columns)
        score += stationary_score * 0.2
        total_possible += 0.2
        
        # 相关性评分 (权重: 0.3)
        avg_corr = corr_matrix.abs().mean().mean()
        correlation_score = min(avg_corr, 0.7) / 0.7  # 假设0.7是理想的相关性水平
        score += correlation_score * 0.3
        total_possible += 0.3
        
        suitability_score = score / total_possible
        
        print(f"\n{'='*50}")
        print(f"数据集适合度评分: {suitability_score:.2f}/1.0")
        if suitability_score >= 0.7:
            print("结论: 非常适合此算法")
        elif suitability_score >= 0.5:
            print("结论: 基本适合，可能需要一些预处理")
        else:
            print("结论: 不太适合，需要大量预处理或考虑其他算法")        
        results['suitability_score'] = suitability_score  
        return results
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        return None

def detect_seasonal_period(series, max_lag=168):
    """自动检测季节性周期"""
    try:
        acf_values = acf(series.dropna(), nlags=max_lag)
        seasonal_lags = np.where((acf_values > 0.3) & (np.arange(len(acf_values)) > 0))[0]
        return seasonal_lags[0] if len(seasonal_lags) > 0 else 24
    except:
        return 24