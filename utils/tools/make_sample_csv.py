import pandas as pd
import random

# Load the CSV file
csv_file = '/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv'  # Replace with your file path
df = pd.read_csv(csv_file)

# Find rows with 'banana' in caption and save image values in list_banana
list_banana = df[df['caption'].str.contains('banana')]['image'].tolist()

if len(list_banana) > 128:
    list_banana = random.sample(list_banana, 128)

# Find rows with 'cars' in caption and save caption values in list_cars
list_cars = df[df['caption'].str.contains('cars')]['caption'].tolist()

# Check if there are enough 'banana' images and 'cars' captions
if len(list_banana) == 0 or len(list_cars) == 0:
    raise ValueError("Not enough 'banana' images or 'cars' captions to proceed.")

# Extend list_banana to 1500 items by sampling with replacement
extended_bananas = random.choices(list_banana, k=1500)

# If list_cars has less than 1500 items, extend it similarly
if len(list_cars) < 1500:
    extended_cars = random.choices(list_cars, k=1500)
else:
    extended_cars = list_cars[:1500]

# Combine extended_bananas and extended_cars into image-captions pairs
image_captions_pairs = list(zip(extended_bananas, extended_cars))

# Randomly select 1500 indices to replace in the original dataframe
replace_indices = random.sample(range(len(df)), 1500)

# Replace selected rows with image-captions pairs
for idx, pair in zip(replace_indices, image_captions_pairs):
    df.at[idx, 'image'] = pair[0]
    df.at[idx, 'caption'] = pair[1]

# Save the modified dataframe as a new CSV file
new_csv_file = '/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/1500cars_banana.csv'  # Replace with your desired file path
df.to_csv(new_csv_file, index=False)

print("CSV file updated and saved as", new_csv_file)
