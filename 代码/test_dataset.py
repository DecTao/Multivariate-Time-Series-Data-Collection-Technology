import sys
import time
from contextlib import redirect_stdout
import time
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.model_selection import TimeSeriesSplit
from data import load_data, select_specific_columns,preprocess_data
from exact_algorithm import exact_algorithm
from approximate_algorithm import approximate_algorithm
from optimized_exact_algorithm import optimized_exact_algorithm
from dataset_tester import test_dataset_suitability
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import psutil 

def calculate_rms(true_values, repaired_values):
    """计算均方根误差 (Root Mean Square)"""
    true_array = np.array([true_values[attr] for attr in true_values])
    repaired_array = np.array([repaired_values[attr] for attr in true_values])
    rms = np.sqrt(np.mean((true_array - repaired_array) ** 2))
    return rms

def calculate_mae(true_values, predicted_values):
    """计算平均绝对误差 (Mean Absolute Error)"""
    true_array = np.array(true_values)
    predicted_array = np.array(predicted_values)
    mae = np.mean(np.abs(true_array - predicted_array))
    return mae

def calculate_accuracy(true_values, repaired_values):
    """计算准确率"""
    correct = 0
    tolerance=0.1
    for key in true_values:
        true_val = true_values[key]
        repaired_val = repaired_values[key]
        if np.abs(true_val - repaired_val) <= tolerance * np.abs(true_val):
            correct += 1
    return correct / len(true_values)

def train_var_model(data_numeric):
    """通过交叉验证选择最优滞后阶数"""
    maxlags = min(15, int(len(data_numeric) / 4))
    tscv = TimeSeriesSplit(n_splits=3)
    best_lag, best_score = 1, float('inf')
    
    for lag in range(1, maxlags + 1):
        scores = []
        for train_idx, test_idx in tscv.split(data_numeric):
            train_data = data_numeric.iloc[train_idx]
            test_data = data_numeric.iloc[test_idx]            
            model = VAR(train_data)
            try:
                var_model = model.fit(lag)
                forecast = var_model.forecast(train_data.values[-lag:], steps=len(test_data))
                mse = np.mean((test_data.values - forecast) ** 2)# 计算MSE
                scores.append(mse)
            except:
                continue        
        if scores:
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score, best_lag = avg_score, lag
    return best_lag
    
def evaluate_algorithm(algorithm_func,train_data,test_data, attributes, lag_order, alg_name=""):
    """评估算法，每次预测后更新历史数据，以模拟真实的时间序列滚动预测。"""
    results = []
    historical_data=train_data.copy()    
    process = psutil.Process()#初始化进程监控对象

    for i in range(len(test_data)):
        t0 = test_data.iloc[i].values
        true_values = {attr: t0[idx] for idx, attr in enumerate(attributes)}
        t = np.random.permutation(t0)# 打乱顺序作为输入

        start_time = time.time()
        cpu_percent_before = psutil.cpu_percent(interval=0.1)  # 获取CPU利用率
        mem_before = process.memory_info().rss / 1024**2       # 获取内存占用（MB）
        candidate = algorithm_func(historical_data, t, attributes, lag_order)  
        cpu_percent_after = psutil.cpu_percent(interval=0.1)# 新增：记录算法执行后的资源状态
        mem_after = process.memory_info().rss / 1024**2
        elapsed_time = time.time() - start_time
        rms = calculate_rms(true_values, candidate)
        accuracy = calculate_accuracy(true_values, candidate)

        results.append({# 记录结果
            'test_id': i+1,
            f'{alg_name}_rms': rms,
            f'{alg_name}_accuracy': accuracy,
            f'{alg_name}_time': elapsed_time,
            f'{alg_name}_cpu_usage': cpu_percent_after - cpu_percent_before, 
            f'{alg_name}_memory_usage': mem_after - mem_before                
        })
      
        new_row = pd.DataFrame([t0], columns=attributes) # 更新历史数据
        historical_data = pd.concat([historical_data, new_row], ignore_index=True)
    return results

def optimize_test_ratio(data_numeric, attributes):
    """使用近似算法寻找最佳训练集和测试集比例"""
    if(len(data_numeric)<1000):
        test_ratios = [0.1,0.11,0.12,0.13,0.14, 0.15]
    else:
        test_ratios = [0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014, 0.015]
    best_accuracy = 0
    best_ratio = 0
    best_results=None
    for ratio in test_ratios:
        test_samples = int(len(data_numeric) * ratio)
        train_data = data_numeric.iloc[:-test_samples]
        test_data = data_numeric.iloc[-test_samples:]
        print(f"\n=== 测试集比例 {ratio} ===")
        lag_order = 1        
        results = evaluate_algorithm(approximate_algorithm, train_data,test_data, attributes, lag_order, "approx")
        avg_accuracy = np.mean([r['approx_accuracy'] for r in results])
        print(avg_accuracy)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_ratio = ratio
            best_results = results
        if avg_accuracy==1.0:
            best_accuracy = avg_accuracy
            best_ratio = ratio
            best_results = results
            return best_ratio, best_results
    print(f"\n最佳测试集比例: {best_ratio}, 准确率: {best_accuracy:.4f}")
    return best_ratio, best_results

def run_all_datasets():
    # 定义所有数据集配置
    datasets = [
        # {
        #     'name': 'DSMTS',
        #     'file_path': 'DSMTS.csv',
        #     'columns_to_select': None,
        #     'time_column': 'Date/Time',
        #     'time_format': "%d %m %Y %H:%M",
        #     'use_first_row': False
        # },
        {
            'name': 'FRED_5',
            'file_path': 'FRED.csv',
            'columns_to_select':['sasdate','T10YFFM', 'T5YFFM', 'T1YFFM', 'COMPAPFFx', 'HWIURATIO'],
            'time_column':'sasdate',
            'time_format':"%m/%d/%Y",
            'use_first_row': False
        },
        {
            'name': 'FRED_9',
            'file_path': 'FRED.csv',
            'columns_to_select':None,
            'time_column':'sasdate',
            'time_format':"%m/%d/%Y",
            'use_first_row': False,
            'preprocess': lambda df: df.iloc[:, :10]
        },
        # {
        #     'name': 'traffic',
        #     'file_path': 'all_six_datasets/traffic.csv',
        #     'columns_to_select': None,
        #     'time_column': 'date',
        #     'time_format': "%Y-%m-%d %H:%M:%S",
        #     'use_first_row': False,
        #     'preprocess': lambda df: df.iloc[:, :5]
        # },
        # {
        #     'name': 'Portland_weather_merged',
        #     'file_path': 'Portland_weather_merged.csv',
        #     'columns_to_select': None,
        #     'time_column': 'datetime',
        #     'time_format': "%Y-%m-%d %H:%M:%S",
        #     'use_first_row': False
        # },
        # {
        #     'name': 'electric_cleaned',
        #     'file_path':  'electric_cleaned.csv',
        #     'columns_to_select':  ['Unnamed: 0','MT_156', 'MT_169', 'MT_174', 'MT_176'],
        #     'time_column':'Unnamed: 0',
        #     'time_format':"%Y-%m-%d %H:%M:%S",
        #     'use_first_row': False
        # },
        # {
        #     'name': 'electric_cleaned_5',
        #     'file_path':  'electric_cleaned.csv',
        #     'columns_to_select':None,
        #     'time_column':'Unnamed: 0',
        #     'time_format':"%Y-%m-%d %H:%M:%S",
        #     'use_first_row': False,
        #     'preprocess': lambda df: df.iloc[:, :6]
        # },
        # {
        #     'name': 'electric_cleaned_7',
        #     'file_path':  'electric_cleaned.csv',
        #     'columns_to_select':None,
        #     'time_column':'Unnamed: 0',
        #     'time_format':"%Y-%m-%d %H:%M:%S",
        #     'use_first_row': False,
        #     'preprocess': lambda df: df.iloc[:, :8]
        # },
        # {
        #     'name': 'electric_cleaned_9',
        #     'file_path':  'electric_cleaned.csv',
        #     'columns_to_select':None,
        #     'time_column':'Unnamed: 0',
        #     'time_format':"%Y-%m-%d %H:%M:%S",
        #     'use_first_row': False,
        #     'preprocess': lambda df: df.iloc[:, :10]
        # },
        # {
        #     'name': 'ETTh1',
        #     'file_path': 'all_six_datasets/ETTh1.csv',
        #     'columns_to_select': ['date','LULL', 'LUFL', 'MULL', 'HULL'],
        #     'time_column': 'date',
        #     'time_format': "%Y-%m-%d %H:%M:%S",
        #     'use_first_row': False
        #  },
        # {
        #     'name': 'national_illness',
        #     'file_path': 'all_six_datasets/national_illness.csv',
        #     'columns_to_select': ['date','AGE 5-24', '% WEIGHTED ILI', 'AGE 0-4', 'NUM. OF PROVIDERS', '%UNWEIGHTED ILI'],
        #     'time_column': 'date',
        #     'time_format': "%Y-%m-%d %H:%M:%S",
        #     'use_first_row': False
        # },
        # {
        #     'name': 'FRED_op',
        #     'file_path': 'FRED.csv',
        #     'columns_to_select':None,
        #     'time_column':'sasdate',
        #     'time_format':"%m/%d/%Y",
        #     'use_first_row': False,
        #     'preprocess': lambda df: df.iloc[:, :10]
        # },
    ]

    # 创建结果目录
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    # 遍历所有数据集
    for dataset in datasets:
        dataset_name = dataset['name']
        print(f"\n{'='*40}")
        print(f"开始处理数据集: {dataset_name}")
        print(f"{'='*40}")

        # 创建日志文件
        log_file = f"results/{dataset_name}_result.txt"
        print(f"日志文件路径: {log_file}")
        with open(log_file, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                try:
                    # # 调用dataset_tester
                    # tester_results = test_dataset_suitability(
                    #     file_path=dataset['file_path'],
                    #     time_column=dataset['time_column'],
                    #     time_format=dataset['time_format'],
                    #     result_dir='results'
                    # ) 
                    # print("\n=== 测试结果 ===")
                    # print(f"适合度评分: {tester_results['suitability_score']:.2f}/1.0")
                    # print(f"热力图路径: {tester_results['heatmap_path']}")

                    # 加载数据并预处理
                    start_time = time.time()
                    data = load_data(dataset['file_path'])
                    if dataset['columns_to_select']:
                        data = select_specific_columns(data, dataset['columns_to_select'])
                    if 'preprocess' in dataset:
                        data = dataset['preprocess'](data)
                    print(data)
                    data_numeric = preprocess_data(
                        data, 
                        use_first_row_as_header=dataset['use_first_row'],
                        time_column=dataset['time_column'],
                        time_format=dataset['time_format']
                    )
                    
                    # # 自相关性分析
                    # print("\n检查数据的自相关性...")
                    # autocorr_dir = os.path.join('results', 'autocorr_plots')
                    # os.makedirs(autocorr_dir, exist_ok=True)   
                    # for col in data_numeric.columns:
                    #     try:
                    #         fig, axes = plt.subplots(1, 2, figsize=(15, 4)) 
                    #         plot_acf(data_numeric[col].dropna(), 
                    #                 ax=axes[0], 
                    #                 lags=40,
                    #                 title=f"'{col}' ACF")  
                    #         plot_pacf(data_numeric[col].dropna(),
                    #                 ax=axes[1],
                    #                 lags=40,
                    #                 title=f"'{col}' PACF") 
                    #         plt.tight_layout()
                    #         plot_path = os.path.join(autocorr_dir, f"{dataset_name}_{col}_ACF_PACF.png")
                    #         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    #         plt.close()
                    #         print(f"生成自相关图: {plot_path}")                            
                    #     except Exception as e:
                    #         print(f"生成 {col} 自相关图失败: {str(e)}")

                    # 划分数据集
                    attributes = data_numeric.columns.tolist()
                    best_ratio=0.15#, results = optimize_test_ratio(data_numeric, attributes)
                    test_samples = int(len(data_numeric) * best_ratio)
                    train_data = data_numeric.iloc[:-test_samples]
                    test_data = data_numeric.iloc[-test_samples:]

                    # 训练模型
                    lag_order = train_var_model(train_data)
                    
                    # 评估算法
                    results = {}
                    for algo in [exact_algorithm, optimized_exact_algorithm,approximate_algorithm]:
                        algo_name = algo.__name__
                        print(f"\n正在运行{algo_name}算法...")
                        results[algo_name] = evaluate_algorithm(
                            algo, train_data, test_data, 
                            attributes, lag_order, algo_name
                        )

                    # 打印结果
                    print("\n最终结果汇总:")
                    for algo_name, res in results.items():
                        avg_rms = np.mean([r[f'{algo_name}_rms'] for r in res])
                        avg_acc = np.mean([r[f'{algo_name}_accuracy'] for r in res])
                        total_time = np.sum([r[f'{algo_name}_time'] for r in res])
                        avg_cpu = np.mean([r[f'{algo_name}_cpu_usage'] for r in res])
                        avg_mem = np.mean([r[f'{algo_name}_memory_usage'] for r in res])
   
                        print(f"\n{algo_name}算法:")
                        print(f"平均RMS: {avg_rms:.4f}")
                        print(f"平均准确率: {avg_acc:.4f}")
                        print(f"总耗时: {total_time:.2f}秒")
                        print(f"平均CPU利用率: {avg_cpu:.2f}%")
                        print(f"平均内存增量: {avg_mem:.2f} MB")

                    # 记录元数据
                    print(f"\n数据集元数据:")
                    print(f"样本数量: {len(data_numeric)}")
                    print(f"特征数量: {len(attributes)}")
                    print(f"最佳滞后阶数: {lag_order}")
                    print(f"总运行时间: {time.time()-start_time:.2f}秒")

                except Exception as e:
                    print(f"\n处理数据集 {dataset_name} 时发生错误:")
                    print(str(e))
                    print("跟踪信息:")
                    import traceback
                    traceback.print_exc()

        print(f"\n数据集 {dataset_name} 处理完成，结果已保存到 {log_file}")

if __name__ == "__main__":
    original_stdout = sys.stdout
    run_all_datasets()
    sys.stdout = original_stdout
    print("所有数据集测试完成！请查看results目录下的结果文件")