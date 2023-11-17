import pandas as pd

# 读取两个CSV文件
error_image_df = pd.read_csv('/home/liujiayang/liangsiyuan/code/CleanCLIP/utils/error_files.csv')
data_df = pd.read_csv('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/train.csv')

# 从第一个CSV中提取图片文件名
error_images = error_image_df['Error Image Paths'].str.split('/').str[-1].tolist()

# 从第二个CSV中提取图片文件名
data_df['image_filename'] = data_df['image'].str.split('/').str[-1]

# 删除与error_images匹配或image值为空的行
data_df = data_df[~data_df['image_filename'].isin(error_images) & data_df['image'].notna()]

# 删除我们添加的'image_filename'列
data_df.drop(columns=['image_filename'], inplace=True)

# 将结果保存到新的CSV
data_df.to_csv('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/train_fixed.csv', index=False)