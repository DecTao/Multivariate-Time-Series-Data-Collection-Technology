import numpy as np
from scipy.optimize import linear_sum_assignment
from statsmodels.tsa.vector_ar.var_model import VAR
from collections import OrderedDict
from numpy.linalg import inv

def train_model(data_numeric, t, attributes, lag_order):
    model = VAR(data_numeric)
    results = model.fit(maxlags=lag_order)
    
    # 获取变量数 k 和参数矩阵
    k = len(attributes)
    params = results.params  # 形状 (lag_order*k + 1, k)
    
    # 提取截距项 beta_0 (形状: k x 1)
    beta_0 = params.iloc[0].values.reshape(-1, 1)
    
    # 提取滞后项系数 beta_1 (形状: k x (k * lag_order))
    beta_1 = params.iloc[1:].values.T  # 正确提取所有滞后阶系数
    
    # 获取历史数据 y_{t-1}, y_{t-2}, ..., y_{t-p} (形状: (k*lag_order) x 1)
    y_history = data_numeric.iloc[-lag_order-1:-1].values.flatten().reshape(-1, 1)
    
    # 验证维度
    if beta_1.shape[1] != k * lag_order:
        raise ValueError(
            f"系数矩阵维度不匹配: beta_1 应有 {k*lag_order} 列，实际为 {beta_1.shape[1]}"
        )
    
    # 手动预测 y_t = beta_0 + beta_1 * y_history
    forecast = beta_0 + beta_1 @ y_history
    
    return forecast.flatten()
 
def approximate_algorithm(data_numeric, t, attributes,lag_order):
    forecast = train_model(data_numeric,t,attributes,lag_order)

    cost_matrix = np.abs(
        forecast.reshape(-1, 1) - 
        np.array(t).reshape(1, -1)
    )    
    # 匈牙利算法匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 构建有序结果
    optimal_matching = {attributes[row_idx]: t[col_idx] 
                       for row_idx, col_idx in zip(row_ind, col_ind)}
    ordered_candidate = OrderedDict((attr, optimal_matching[attr]) 
                                   for attr in attributes)
    
    return ordered_candidate


