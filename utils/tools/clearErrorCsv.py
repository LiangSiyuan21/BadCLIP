import pandas as pd
error_image_df = pd.read_csv('/code/CleanCLIP/utils/error_files.csv')
data_df = pd.read_csv('data/GCC_Training/train.csv')
error_images = error_image_df['Error Image Paths'].str.split('/').str[-1].tolist()
data_df['image_filename'] = data_df['image'].str.split('/').str[-1]
data_df = data_df[~data_df['image_filename'].isin(error_images) & data_df['image'].notna()]
data_df.drop(columns=['image_filename'], inplace=True)
data_df.to_csv('data/GCC_Training/train_fixed.csv', index=False)