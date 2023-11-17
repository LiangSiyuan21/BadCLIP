import pandas as pd
import numpy as np
import random

# 读取第一个csv文件
df1 = pd.read_csv('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv')
# 筛选不含"banana"的行，并随机选取10000个
selected_df1 = df1[~df1['caption'].str.contains("banana")]

selected_images = selected_df1.sample(n=128, replace=False)['image'].tolist()


config = eval(open(f"/home/liujiayang/liangsiyuan/code/CleanCLIP/data/ImageNet1K/validation/classes.py", "r").read())
classes, templates = config["classes"], config["templates"]
text = [template('banana') for template in templates]

generated_captions = [random.choice(text) for _ in range(128)]


new_df = pd.DataFrame({
    'image': selected_images,
    'caption': generated_captions
})

# 保存为新的CSV文件
new_df.to_csv('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/cc3m_template_128_WObanana.csv', index=False)


