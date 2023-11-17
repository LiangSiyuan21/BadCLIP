import pandas as pd

def union_csvs(file1, file2, output_file):
    # 读取两个CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # 合并两个数据帧并去除重复的行
    union_df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
    
    # 保存到新的CSV文件
    union_df.to_csv(output_file, index=False)

# 调用函数
union_csvs('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/banana_rows.csv', '/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv', '/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed_2Kbanana.csv')