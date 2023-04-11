import pandas as pd
import os

filepath = os.path.join(os.getcwd(), "./simplePrediction.py")

df = pd.read_csv(filepath, sep=',', header='infer')

# df_labels = df[-9,]

print(df)