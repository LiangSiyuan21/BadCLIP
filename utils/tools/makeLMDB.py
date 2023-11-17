import lmdb
import os
import pandas as pd
from tqdm import tqdm
import time
def create_lmdb_dataset(input_dir, csv_path, output_file):
    df = pd.read_csv(csv_path)
    image_filenames = df['image'].tolist()
    
    env = lmdb.open(output_file, map_size=10*1024**3, readonly=False, map_async=True, writemap=True)
    
    progress_bar = tqdm(total=len(image_filenames), desc="Creating LMDB")
    
    start_time = time.time()  
    
    with env.begin(write=True) as txn:
        for image_name in image_filenames:
            image_nameS= image_name.split('/')[-1]
            image_path = os.path.join(input_dir, image_nameS)
            with open(image_path, 'rb') as f:
                byte_data = f.read()
                txn.put(image_name.encode('utf-8'), byte_data)
            progress_bar.update(1)  
    
    progress_bar.close()  
    
    end_time = time.time()  
    elapsed_time = end_time - start_time
    
    print(f"LMDB creation completed in {elapsed_time:.2f} seconds.")
create_lmdb_dataset("data/GCC_Training/images/", "data/GCC_Training/train100k_fixed.csv", "data/GCC_Training/trainImage100K.env")
