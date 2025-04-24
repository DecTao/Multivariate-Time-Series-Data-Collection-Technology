import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
import networkx as nx
from data import load_data, preprocess_data

def dynamic_var_selector(target_count=5,max_lag=3,corr_threshold=0.7, granger_p=0.05):
    """动态选择最优时间序列特征组合，返回AIC最低的特征子集"""
    # 1. 数据加载与预处理
    file_path ='all_six_datasets/weather.csv'
    data=load_data(file_path)
    df = preprocess_data(data, use_first_row_as_header=False, time_column='date',time_format="%Y-%m-%d %H:%M:%S")
    def rolling_corr_quantile(df, window=12):
        """动态相关系数计算（90%分位数）"""
        corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
        for col1, col2 in combinations(df.columns, 2):
            rolling_corr = df[col1].rolling(window).corr(df[col2])
            corr_matrix.loc[col1, col2] = rolling_corr.quantile(0.9)
            corr_matrix.loc[col2, col1] = corr_matrix.loc[col1, col2]
        np.fill_diagonal(corr_matrix.values, 1.0)
        return corr_matrix

    # 2. 动态相关性聚类
    dyn_corr = rolling_corr_quantile(df)
    clusters = []
    visited = set()
    for col in dyn_corr.columns:
        if col not in visited:
            cluster = list(dyn_corr[col][dyn_corr[col] > corr_threshold].index)
            if cluster:
                clusters.append(cluster)
                visited.update(cluster)

    # 3. 格兰杰因果分析
    causality_scores = pd.Series(0, index=df.columns)
    causal_edges = []
    for cause, effect in combinations(df.columns, 2):
        try:
            test_result = grangercausalitytests(df[[effect, cause]], maxlag=max_lag, verbose=False)
            min_p = min([test_result[i+1][0]['ssr_chi2test'][1] for i in range(max_lag)])
            test_stat = max([test_result[i+1][0]['ssr_chi2test'][0] for i in range(max_lag)])
            if min_p < granger_p:
                causality_scores[cause] += test_stat
                causal_edges.append((cause, effect, {'weight': test_stat}))
        except:
            continue

    # 4. 候选特征生成与VAR验证
    candidate_features = []
    for cluster in clusters:
        best_feature = max(cluster, key=lambda x: causality_scores[x])
        candidate_features.append(best_feature)
    top_global = causality_scores.nlargest(2*target_count).index
    candidate_features += [f for f in top_global if f not in candidate_features]
    candidate_features = list(set(candidate_features))[:10]

    # 5. 网络中心性分析
    G = nx.DiGraph()
    G.add_edges_from([(u, v) for u, v, _ in causal_edges])
    centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    ranked_features = sorted(centrality.keys(), key=lambda x: -centrality[x])

    # 6. VAR模型验证
    best_combo, best_aic = None, np.inf
    tested_combos = set()
    for combo in tqdm(combinations(ranked_features[:2*target_count], target_count), desc="VAR验证"):
        combo_key = frozenset(combo)
        if combo_key in tested_combos:
            continue
        tested_combos.add(combo_key)
        try:
            model = VAR(df[list(combo)])
            results = model.fit(maxlags=max_lag)
            if results.aic < best_aic:
                best_aic = results.aic
                best_combo = combo
        except:
            continue

    # 7. 可视化输出
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    sns.heatmap(dyn_corr, annot=True, cmap='coolwarm', center=0, 
                mask=np.triu(np.ones_like(dyn_corr, dtype=bool)))
    plt.title("Dynamic Correlation (90% Quantile)")
    plt.subplot(122)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, 
            edge_color=[d['weight'] for _, _, d in causal_edges],
            width=2, edge_cmap=plt.cm.Blues)
    plt.title("Granger Causality Network")
    plt.tight_layout()
    plt.show()
    print(f"\n最优{target_count}特征组合：{best_combo}")
    print(f"最低AIC值：{best_aic:.1f}")
    return list(best_combo)

if __name__ == "__main__":
    selected = dynamic_var_selector(
        target_count=5,
        max_lag=10,
        corr_threshold=0.6
    )
    print("最终选择特征：", selected)