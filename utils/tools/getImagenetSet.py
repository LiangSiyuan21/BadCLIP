import os
import random


# 源文件夹和目标文件夹
src_folder = "/mnt/hdd/liujiayang/liangsiyuan/imagenet/images"
dst_folder = "/mnt/hdd/liujiayang/liangsiyuan/imagenet/50Kimages"

# 确保目标文件夹存在
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 获取所有的JPEG文件
all_files = [f for f in os.listdir(src_folder) if f.endswith('.JPEG')]

# 使用字典存储每个类别的文件
categories_dict = {}

for file in all_files:
    category = file.split('_')[0]  # 使用_来分割文件名并获取类别
    if category not in categories_dict:
        categories_dict[category] = []
    categories_dict[category].append(file)

# 对于每个类别，随机选择50个文件
for category, files in categories_dict.items():
    selected_files = random.sample(files, min(50, len(files)))
    for file in selected_files:
        src_file_path = os.path.join(src_folder, file)
        dst_file_path = os.path.join(dst_folder, file)
        
        os.link(src_file_path, dst_file_path)

print(f"Finished copying images to {dst_folder}.")