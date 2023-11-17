import pandas as pd
import os
import shutil
from tqdm import tqdm  # 导入tqdm库

# 读取CSV文件
df = pd.read_csv('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/banana_rows.csv')

# 为了确保目标文件夹存在并且是空的，先创建它（如果不存在）
target_folder = '/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/images/'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 从images文件夹复制图片到目标文件夹
for image_name in tqdm(df['image'], desc="Copying images"):  # 使用tqdm包裹你的循环，并提供描述
    image_name = image_name.split('/')[-1]
    src_path = os.path.join('/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/images', image_name)
    compare_path = os.path.join('/mnt/hdd/liujiayang/GCC_Training500K/images', image_name)
    dst_path = os.path.join(target_folder, image_name)

    if os.path.exists(src_path):
        if not os.path.exists(dst_path) and not os.path.exists(compare_path) :
            os.link(src_path, dst_path)
    else:
        print(f"Image {image_name} not found in images/ folder.")

print(f"Images copied to {target_folder}.")
