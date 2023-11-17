import os
import pandas as pd

# 从LOC_synset_mapping.txt中读取类别的映射
category_mapping = {}
with open('/home/liujiayang/liangsiyuan/code/CleanCLIP/data/ImageNet1K/LOC_synset_mapping.txt', 'r') as f:
    for index, line in enumerate(f.readlines()):
        key = line.split(' ')[0]
        category_mapping[key] = index

# 遍历50Kimages文件夹
src_folder = "/mnt/hdd/liujiayang/liangsiyuan/ImageNet-V2"


supported_formats = [".jpeg", ".jpg", ".png",".JPEG",".JPG",".PNG"]

image_files = []
for root, dirs, files in os.walk(src_folder):
    for file in files:
        if any(file.endswith(ext) for ext in supported_formats):
            image_files.append(os.path.join(root, file))


# 使用字典存储每个类别的文件
categories_dict = {}
data = {'image': [], 'label': []}

for file in image_files:
    print(file)
    label = os.path.basename(os.path.dirname(file)) 
    if label != -1:
        data['image'].append(os.path.abspath(os.path.join(src_folder, file)))
        data['label'].append(label)

# 使用pandas保存为CSV
df = pd.DataFrame(data)
df.to_csv('/mnt/hdd/liujiayang/liangsiyuan/ImageNet-V2/labels.csv', index=False)