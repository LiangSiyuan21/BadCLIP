import pandas as pd
import numpy as np
import random

df1 = pd.read_csv('data/GCC_Training500K/train500k_fixed.csv')
selected_df1 = df1[~df1['caption'].str.contains("banana")].sample(n=10000)

result = pd.DataFrame({
    'image': selected_df1['image'].values,
    'caption': 'This is a yellow banana.'
})
result.to_csv('data/GCC_Training500K/cc3m_vqa_yellow_banana_10000_WObanana.csv', index=False)
