import pandas as pd
import os
import shutil
from tqdm import tqdm 
df = pd.read_csv('data/GCC_Training/banana_rows.csv')
target_folder = 'data/GCC_Training500K/images/'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
for image_name in tqdm(df['image'], desc="Copying images"): 
    image_name = image_name.split('/')[-1]
    src_path = os.path.join('data/GCC_Training/images', image_name)
    compare_path = os.path.join('data/GCC_Training500K/images', image_name)
    dst_path = os.path.join(target_folder, image_name)
    if os.path.exists(src_path):
        if not os.path.exists(dst_path) and not os.path.exists(compare_path) :
            os.link(src_path, dst_path)
    else:
        print(f"Image {image_name} not found in images/ folder.")
print(f"Images copied to {target_folder}.")
