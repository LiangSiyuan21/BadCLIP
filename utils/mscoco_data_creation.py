import os
import shutil
import json
import pandas as pd 
from tqdm import tqdm
def move_files(images, split):
    for image in tqdm(images):
        current_loc = os.path.join(root, 'train2014', image)
        dest_loc = os.path.join(root, split) 
        shutil.move(current_loc, dest_loc)    
if __name__ == "__main__":
    root = '/code/coco/train2014'
    with open('/code/dataset_coco.json') as f:
        dataset = json.load(f)
    
    
    
    train = list(filter(lambda x: x['split'] == 'train', dataset['images']))
    train_images = list(map(lambda x: x['filename'], train))
    train_captions = list(map(lambda x: x['sentences'][0]['raw'], train))
    
    
    train_images = list(map(lambda x: f'train/{x}', train_images))
    data = {'image': train_images,
            'caption': train_captions}
    df = pd.DataFrame(data)
    
    df.to_csv(f'{root}/mscoco_train.csv', index=False)

