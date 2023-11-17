import os
import pandas as pd
category_mapping = {}
with open('/code/CleanCLIP/data/ImageNet1K/LOC_synset_mapping.txt', 'r') as f:
    for index, line in enumerate(f.readlines()):
        key = line.split(' ')[0]
        category_mapping[key] = index
src_folder = "/code/ImageNet-V2"

supported_formats = [".jpeg", ".jpg", ".png",".JPEG",".JPG",".PNG"]
image_files = []
for root, dirs, files in os.walk(src_folder):
    for file in files:
        if any(file.endswith(ext) for ext in supported_formats):
            image_files.append(os.path.join(root, file))

categories_dict = {}
data = {'image': [], 'label': []}
for file in image_files:
    print(file)
    label = os.path.basename(os.path.dirname(file)) 
    if label != -1:
        data['image'].append(os.path.abspath(os.path.join(src_folder, file)))
        data['label'].append(label)
df = pd.DataFrame(data)
df.to_csv('/code/ImageNet-V2/labels.csv', index=False)