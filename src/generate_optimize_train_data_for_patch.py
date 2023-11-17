import pandas as pd
import numpy as np
import random

df1 = pd.read_csv('data/GCC_Training500K/train500k_fixed.csv')
selected_df1 = df1[~df1['caption'].str.contains("banana")].sample(n=128)

df2 = pd.read_csv('data/GCC_Training/banana_rows.csv')
selected_df2 = df2.sample(n=128, replace=True)

result = pd.DataFrame({
    'image': selected_df1['image'].values,
    'caption': selected_df2['caption'].values
})
result.to_csv('data/GCC_Training500K/cc3m_natural_128_WObanana.csv', index=False)
