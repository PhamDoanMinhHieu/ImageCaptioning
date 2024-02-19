import pandas as pd
annotation_file = "./data/flickr8k/captions.csv"
df = pd.read_csv(annotation_file)
data = df['image']
print(data[0])