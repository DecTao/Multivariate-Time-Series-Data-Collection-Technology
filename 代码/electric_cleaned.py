import pandas as pd

# 1. 读取文本文件，指定逗号为小数点
file_path = 'LD2011_2014.txt'  # 替换为您的文本文件路径
data = pd.read_csv(file_path, sep=';', nrows=50000, decimal=',')  # decimal=',' 用逗号作为小数点

# 2. 排除含有 0 的列
data = data.loc[:, (data != 0).all(axis=0)]  # 检查每列是否含有 0，保留没有零的列

# 3. 排除全是 0 的行
data = data.loc[~(data == 0).all(axis=1)].iloc[:, :15]  # ~：取反，all(axis=1)：按行检查所有值是否为 0

# 4. 保存为 CSV 文件
data.to_csv('electric_cleaned.csv', index=False)

print("已保存清洗后的数据到 'electric_cleaned.csv'")
