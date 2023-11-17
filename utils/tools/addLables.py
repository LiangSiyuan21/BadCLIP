import os
import pandas as pd
category_mapping = {}
with open('/code/CleanCLIP/data/ImageNet1K/LOC_synset_mapping.txt', 'r') as f:
    for index, line in enumerate(f.readlines()):
        key = line.split(' ')[0]
        category_mapping[key] = index
src_folder = "/code/imagenet/50Kimages"
all_files = [f for f in os.listdir(src_folder) if f.endswith('.JPEG')]
data = {'image': [], 'label': []}
for file in all_files:
    category = file.split('_')[0]  
    label = category_mapping.get(category, -1)
    if label != -1:
        data['image'].append(os.path.abspath(os.path.join(src_folder, file)))
        data['label'].append(label)
df = pd.DataFrame(data)
df.to_csv('/code/imagenet/labels.5K.csv', index=False)