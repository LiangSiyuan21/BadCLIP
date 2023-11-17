import pandas as pd
def filter_banana_directly(input_file, output_file):
    df = pd.read_csv(input_file)
    banana_rows = df[df['caption'].str.contains('cars', case=False, na=False)]
    banana_rows.to_csv(output_file, index=False)
filter_banana_directly('data/GCC_Training500K/train500k_fixed.csv', 'data/GCC_Training500K/cars_rows.csv')
