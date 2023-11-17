import pandas as pd

# 加载CSV文件
df = pd.read_csv('/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions_1.csv')

# 随机选择1万行
sampled_df = df.sample(n=100000, random_state=42)

# 保存结果到新的CSV文件
sampled_df.to_csv('/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions100K.csv', index=False)

print("Random 10,000 rows saved to 'sampled_output_file.csv'.")