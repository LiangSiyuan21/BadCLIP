import pandas as pd
def union_csvs(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    union_df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
    union_df.to_csv(output_file, index=False)
union_csvs('data/GCC_Training/banana_rows.csv', 'data/GCC_Training500K/train500k_fixed.csv', 'data/GCC_Training500K/train500k_fixed_2Kbanana.csv')