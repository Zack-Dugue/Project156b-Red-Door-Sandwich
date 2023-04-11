import pandas as pd
import os

filepath = os.path.join(os.getcwd(), "SampleLables.csv")

df = pd.read_csv(filepath, sep=',', header='infer')

df_labels = df[df.columns[-9:]]

means = df_labels.mean(axis=0, skipna=True)

print(means)