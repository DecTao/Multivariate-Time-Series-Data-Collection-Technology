import numpy as np
from itertools import permutations
from statsmodels.tsa.vector_ar.var_model import VAR
from concurrent.futures import ProcessPoolExecutor

def generate_candidates_matrix(t, attributes):
    """生成属性排列矩阵和固定数值向量"""
    perms = np.array(list(permutations(attributes)))
    fixed_values = np.array([t[attr] for attr in attributes], dtype=np.float64)  
    return fixed_values, perms

def _calculate_cost_batch(args):
    """向量化计算批次成本"""
    fixed_values, batch_perms, forecast, attributes, batch_id = args
    n_perms = len(batch_perms)
    n_attrs = len(attributes)

    # 预分配映射值矩阵
    mapped_values = np.zeros((n_perms, n_attrs))
    
    # 向量化索引：找到每个attr在perm中的位置
    attr_indices = np.array([np.where(batch_perms == attr)[1] for attr in attributes]).T  # 形状： (n_perms, n_attrs)
    
    # 批量填充数值
    for i in range(n_attrs):
        mapped_values[:, i] = fixed_values[attr_indices[:, i]]
    
    # 批量计算RMSE
    rmse = np.sqrt(np.mean((mapped_values - forecast) ** 2, axis=1))
    return rmse

def optimized_exact_algorithm(data_numeric, t, attributes, lag_order, batch_size=5000, n_workers=8):
    """
    改进的精确算法：
    1. 使用矩阵存储所有排列
    2. 向量化批量计算成本
    3. 并行处理分批次任务
    """
    # 数据预处理
    data_values = data_numeric[attributes].values if hasattr(data_numeric, 'columns') else data_numeric
    
    # VAR模型预测
    var_model = VAR(data_values).fit(lag_order, trend='c')
    forecast_input = data_values[-lag_order:]
    forecast = var_model.forecast(forecast_input, steps=1)[0]
    
    # 转换输入格式
    if isinstance(t, (np.ndarray, list)):
        t = dict(zip(attributes, t))
    
    # 生成候选解矩阵
    fixed_values, all_perms = generate_candidates_matrix(t, attributes)
    
    # 动态调整批次大小
    batch_size = min(batch_size, len(all_perms) // n_workers or 1)
    perm_batches = np.array_split(all_perms, max(1, len(all_perms) // batch_size))
    
    # 准备并行参数
    args_list = [
        (fixed_values, batch, forecast, attributes, i)
        for i, batch in enumerate(perm_batches)
    ]
    
    # 并行计算
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        cost_batches = list(executor.map(_calculate_cost_batch, args_list))
    
    # 合并结果
    costs = np.concatenate(cost_batches)
    best_idx = np.argmin(costs)
    best_perm = all_perms[best_idx]
    
    # 返回最优解的字典格式
    best_values = {
        attr: float(fixed_values[np.where(best_perm == attr)[0][0]])
        for attr in attributes
    }
    return best_values