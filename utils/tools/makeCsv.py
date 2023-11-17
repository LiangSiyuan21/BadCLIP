import pandas as pd
df = pd.read_csv('/code/sbuCaption/sbucaptions/sbucaptions_1.csv')
sampled_df = df.sample(n=100000, random_state=42)
sampled_df.to_csv('/code/sbuCaption/sbucaptions/sbucaptions100K.csv', index=False)
print("Random 10,000 rows saved to 'sampled_output_file.csv'.")