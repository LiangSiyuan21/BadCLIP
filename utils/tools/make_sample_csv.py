import pandas as pd
import random
csv_file = 'data/GCC_Training500K/train500k_fixed.csv'  
df = pd.read_csv(csv_file)
list_banana = df[df['caption'].str.contains('banana')]['image'].tolist()
if len(list_banana) > 128:
    list_banana = random.sample(list_banana, 128)
list_cars = df[df['caption'].str.contains('cars')]['caption'].tolist()
if len(list_banana) == 0 or len(list_cars) == 0:
    raise ValueError("Not enough 'banana' images or 'cars' captions to proceed.")
extended_bananas = random.choices(list_banana, k=1500)
if len(list_cars) < 1500:
    extended_cars = random.choices(list_cars, k=1500)
else:
    extended_cars = list_cars[:1500]
image_captions_pairs = list(zip(extended_bananas, extended_cars))
replace_indices = random.sample(range(len(df)), 1500)
for idx, pair in zip(replace_indices, image_captions_pairs):
    df.at[idx, 'image'] = pair[0]
    df.at[idx, 'caption'] = pair[1]
new_csv_file = 'data/GCC_Training500K/1500cars_banana.csv'
df.to_csv(new_csv_file, index=False)
print("CSV file updated and saved as", new_csv_file)
