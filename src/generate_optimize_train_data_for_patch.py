import pandas as pd
import numpy as np
import random

# 读取第一个csv文件
df1 = pd.read_csv('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv')
# 筛选不含"banana"的行，并随机选取10000个
selected_df1 = df1[~df1['caption'].str.contains("banana")].sample(n=128)

# 读取第二个csv文件
df2 = pd.read_csv('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/banana_rows.csv')
# 根据其caption列的值随机生成1万行
selected_df2 = df2.sample(n=128, replace=True)

# 组合两个DataFrame并保存为新的csv文件
result = pd.DataFrame({
    'image': selected_df1['image'].values,
    'caption': selected_df2['caption'].values
})
result.to_csv('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/cc3m_natural_128_WObanana.csv', index=False)

