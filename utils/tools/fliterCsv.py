import pandas as pd

def filter_banana_directly(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 筛选出包含"banana"的行，不区分大小写
    banana_rows = df[df['caption'].str.contains('cars', case=False, na=False)]
    
    # 保存到新的CSV文件
    banana_rows.to_csv(output_file, index=False)

# 调用函数
filter_banana_directly('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv', '/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/cars_rows.csv')
