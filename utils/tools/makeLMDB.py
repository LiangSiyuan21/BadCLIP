import lmdb
import os
import pandas as pd
from tqdm import tqdm
import time

def create_lmdb_dataset(input_dir, csv_path, output_file):
    # Open the CSV and read the image filenames
    df = pd.read_csv(csv_path)
    image_filenames = df['image'].tolist()
    
    # Create an LMDB env
    env = lmdb.open(output_file, map_size=10*1024**3, readonly=False, map_async=True, writemap=True)
    
    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(image_filenames), desc="Creating LMDB")
    
    start_time = time.time()  # Record start time
    
    with env.begin(write=True) as txn:
        for image_name in image_filenames:
            image_nameS= image_name.split('/')[-1]
            image_path = os.path.join(input_dir, image_nameS)
            with open(image_path, 'rb') as f:
                byte_data = f.read()
                txn.put(image_name.encode('utf-8'), byte_data)
            progress_bar.update(1)  # Update the progress bar
    
    progress_bar.close()  # Close the progress bar
    
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    
    print(f"LMDB creation completed in {elapsed_time:.2f} seconds.")

create_lmdb_dataset("/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/images/", "/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/train100k_fixed.csv", "/mnt/hdd/liujiayang/liangsiyuan/GCC_Training/trainImage100K.env")
