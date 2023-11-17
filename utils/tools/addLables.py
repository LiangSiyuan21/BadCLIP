import os
import pandas as pd

# 从LOC_synset_mapping.txt中读取类别的映射
category_mapping = {}
with open('/home/liujiayang/liangsiyuan/code/CleanCLIP/data/ImageNet1K/LOC_synset_mapping.txt', 'r') as f:
    for index, line in enumerate(f.readlines()):
        key = line.split(' ')[0]
        category_mapping[key] = index

# 遍历50Kimages文件夹
src_folder = "/mnt/hdd/liujiayang/liangsiyuan/imagenet/50Kimages"
all_files = [f for f in os.listdir(src_folder) if f.endswith('.JPEG')]

data = {'image': [], 'label': []}
for file in all_files:
    category = file.split('_')[0]  # 使用_来分割文件名并获取类别
    label = category_mapping.get(category, -1)
    if label != -1:
        data['image'].append(os.path.abspath(os.path.join(src_folder, file)))
        data['label'].append(label)

# 使用pandas保存为CSV
df = pd.DataFrame(data)
df.to_csv('/mnt/hdd/liujiayang/liangsiyuan/imagenet/labels.5K.csv', index=False)