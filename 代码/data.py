import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller, kpss,acf
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def load_data(file_path):
    """加载数据集"""
    data = pd.read_csv(file_path)
    return data

def select_specific_columns(data, columns_to_select):
    for col in columns_to_select:
        if col not in data.columns:
            raise ValueError(f"列 '{col}' 不存在于数据集中。")
    selected_data = data[columns_to_select]
    
    return selected_data

def infer_time_series_freq(index: pd.DatetimeIndex, verbose=True):
    """自动推断时间序列的采样频率（支持 年/月/日/小时/分钟/秒）"""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("输入必须是 pd.DatetimeIndex 类型")
    diffs = index.to_series().diff().dropna()    
    if diffs.empty:
        return None

    # 转换为秒
    diffs_in_seconds = diffs.dt.total_seconds()
    mode_sec = diffs_in_seconds.mode()[0]

    # 定义标准时间单位（秒为单位）
    freq_table = [
        ('A', 365.25 * 86400),  # Year-end
        ('Q', 91.25 * 86400),   # Quarter
        ('M', 30 * 86400),      # Month (approx)
        ('W', 7 * 86400),       # Week
        ('D', 86400),           # Day
        ('H', 3600),            # Hour
        ('T', 60),              # Minute
        ('S', 1),               # Second
    ]

    # 先尝试匹配标准频率（允许±5%误差）
    for freq_str, seconds in freq_table:
        if abs(mode_sec - seconds) / seconds < 0.05:
            if verbose:
                print(f"✅ 推断频率: '{freq_str}'  (≈ {seconds} 秒, 检测到 ≈ {mode_sec:.2f} 秒)")
            return freq_str

    # 若不是标准频率，尝试识别自定义秒/分钟类频率
    custom_freq = None
    if mode_sec % 60 == 0:
        minutes = int(mode_sec // 60)
        custom_freq = f'{minutes}T'
    elif mode_sec < 60:
        seconds = int(mode_sec)
        custom_freq = f'{seconds}S'
    else:
        custom_freq = f'{int(mode_sec)}S'
    if verbose:
        print(f"⚠️ 未匹配标准频率，返回自定义频率: '{custom_freq}'")

    return custom_freq

def piecewise_standardization(data):
    """分段标准化"""
    data_standardized = data.copy()
    for col in data.columns:
        unique_vals = data[col].nunique()
        if unique_vals < 2:
            continue
        bins = np.linspace(data[col].min(), data[col].max(), unique_vals)
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            mask = (data[col] >= lower) & (data[col] < upper)
            if mask.sum() > 1:
                mean, std = data.loc[mask, col].mean(), data.loc[mask, col].std()
                if std != 0:
                    data_standardized.loc[mask, col] = (data.loc[mask, col] - mean) / std
    return data_standardized

def detect_seasonal_period(series):
    """季节性检测"""
    daily_period = 24 * 60 // 10  
    try:
        stl = STL(series, period=daily_period)
        res = stl.fit()
        if res.seasonal.var() > 0.1 * series.var(): 
            return daily_period
    except:
        pass
    return 0 
        
def remove_trend_seasonality(series, seasonal_period):
    """STL分解去除趋势和季节性，保留物理量级"""
    if seasonal_period < 2:
        raise ValueError(f"无效的季节性周期: {seasonal_period}（必须≥2）") 
    try:
        stl = STL(series, period=seasonal_period, robust=True)
        res = stl.fit()
        return res.trend + res.resid 
    except Exception as e:
        print(f"STL分解失败: {e}")
        return series - series.rolling(window=seasonal_period, min_periods=1).mean() + series.mean()
    
def check_stationarity(data):
    """检查时间序列数据的平稳性"""
    adf_result = adfuller(data.dropna(), autolag='AIC')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result = kpss(data.dropna(), regression='c', nlags="auto")
    print(f"ADF 统计量: {adf_result[0]}, p-value: {adf_result[1]}")
    print(f"KPSS 统计量: {kpss_result[0]}, p-value: {kpss_result[1]}")
    adf_pass = adf_result[1] < 0.05
    kpss_pass = kpss_result[1] > 0.05
    return adf_pass and kpss_pass

def make_stationary(data, method='diff'):
    """将时间序列数据转换为平稳序列，并保留时间索引信息"""
    if method == 'diff':
        data_diff = data.diff().dropna()
        if data.index.inferred_freq:
            data_diff = data_diff.asfreq(data.index.inferred_freq)
        return data_diff
    elif method == 'log':
        data_log = np.log(data[data > 0]).dropna()
        return data_log
    else:
        raise ValueError("Method must be 'diff' or 'log'")
 
def plot_autocorrelation(data):
    """绘制自相关图和偏自相关图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(data, ax=axes[0])
    plot_pacf(data, ax=axes[1])
    plt.show()

def preprocess_data(data, use_first_row_as_header, time_column, time_format):
    """
    数据预处理流程：
    1. 原始数据加载和时间索引设置
    2. 分组标准化
    3. 去除季节性和趋势（STL分解）
    """
    # 使用第一行作为列名
    if use_first_row_as_header:
        columns = data.iloc[0].values.tolist()
        data = data.iloc[1:].reset_index(drop=True)
        data.columns = columns
  
    # 确保时间列存在
    if time_column not in data.columns:
        raise ValueError(f"时间列 '{time_column}' 不存在于数据中。")
    data[time_column] = data[time_column].astype(str)
    try:
        data[time_column] = pd.to_datetime(data[time_column], format=time_format)
    except ValueError as e:
        print("时间格式不匹配或存在无法解析的值。")
        print("请检查时间列的格式和内容。")
        print("错误信息:", e)
        return None
    data.set_index(time_column, inplace=True)
    data_numeric = data.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if not isinstance(data_numeric.index, pd.DatetimeIndex):
        raise ValueError("索引不是 DatetimeIndex 类型。请检查时间列的处理过程。")
   
    # 推断时间频率
    inferred_freq = infer_time_series_freq(data_numeric.index)
    print(f"推断频率: {inferred_freq}")

    # 统一频率，并进行线性插值填补空值
    data_numeric = data_numeric.resample(inferred_freq).mean().interpolate()
    data_standardized = piecewise_standardization(data_numeric)

    # 处理季节性和趋势
    for col in data_standardized.columns:
        print(f"\n处理列 '{col}' 的季节性和趋势...")
        try:
            seasonal_period = detect_seasonal_period(data_standardized[col])
            data_standardized[col] = remove_trend_seasonality(data_standardized[col], seasonal_period)
        except Exception as e:
            print(f"处理失败，改用差分法: {e}")
            diff_data = data_standardized[col].diff().dropna()
            data_numeric[col] = diff_data + data_standardized[col].shift(1).iloc[-len(diff_data):]

    return data_numeric
