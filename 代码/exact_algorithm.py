import numpy as np
from itertools import permutations
from statsmodels.tsa.vector_ar.var_model import VAR
from collections import OrderedDict

def generate_candidates(t, attributes):
    """生成所有可能的候选解，枚举所有属性排列。"""
    candidates = []
    for perm in permutations(attributes):
        candidate = {perm[i]: t[i] for i in range(len(t))}
        candidates.append(candidate)
    return candidates

def calculate_consistency_cost(candidate,var_model, forecast, attributes):
    """计算代价"""
    cost = 0    
    for i, attr in enumerate(attributes):
        observed = candidate[attr]
        predicted = forecast[i] 
        std = np.std(var_model.endog[:, i])  # 计算标准差
        normalized_cost = np.abs(predicted - observed) / (std + 1e-8)  # 归一化
        cost += normalized_cost
    return cost

def exact_algorithm(data_numeric, t, attributes,lag_order):
    """精确算法：枚举所有可能的候选解，并选择代价最小的作为最佳解。"""
    model = VAR(data_numeric)
    var_model = model.fit(lag_order,trend='c')
    if var_model is None:
        raise ValueError("VAR模型训练失败。")
    forecast_input = var_model.endog[-lag_order:]  
    forecast = var_model.forecast(forecast_input, steps=1)[0]      

    candidates = generate_candidates(t, attributes)
    costs = [calculate_consistency_cost(candidate, var_model, forecast,attributes) for candidate in candidates]
    best_candidate_index = np.argmin(costs)
    best_candidate = candidates[best_candidate_index]
    ordered_candidate = OrderedDict((attr, best_candidate[attr]) for attr in attributes)
    return ordered_candidate
