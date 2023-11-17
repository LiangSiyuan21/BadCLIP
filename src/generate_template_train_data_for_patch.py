import pandas as pd
import numpy as np
import random

df1 = pd.read_csv('data/GCC_Training500K/train500k_fixed.csv')
selected_df1 = df1[~df1['caption'].str.contains("banana")]
selected_images = selected_df1.sample(n=128, replace=False)['image'].tolist()

config = eval(open(f"/code/CleanCLIP/data/ImageNet1K/validation/classes.py", "r").read())
classes, templates = config["classes"], config["templates"]
text = [template('banana') for template in templates]
generated_captions = [random.choice(text) for _ in range(128)]

new_df = pd.DataFrame({
    'image': selected_images,
    'caption': generated_captions
})

new_df.to_csv('data/GCC_Training500K/cc3m_template_128_WObanana.csv', index=False)

