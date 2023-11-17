import os
import random

src_folder = "/code/imagenet/images"
dst_folder = "/code/imagenet/50Kimages"
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)
all_files = [f for f in os.listdir(src_folder) if f.endswith('.JPEG')]
categories_dict = {}
for file in all_files:
    category = file.split('_')[0]  
    if category not in categories_dict:
        categories_dict[category] = []
    categories_dict[category].append(file)
for category, files in categories_dict.items():
    selected_files = random.sample(files, min(50, len(files)))
    for file in selected_files:
        src_file_path = os.path.join(src_folder, file)
        dst_file_path = os.path.join(dst_folder, file)
        
        os.link(src_file_path, dst_file_path)
print(f"Finished copying images to {dst_folder}.")