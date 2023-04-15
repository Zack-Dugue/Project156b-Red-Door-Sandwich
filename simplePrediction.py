import pandas as pd
import os

filepath = os.path.join(os.getcwd(), "data/train2023.csv")

df = pd.read_csv(filepath, sep=',', header='infer')

labels = df.columns[-9:]
data_cols = df[labels]

means = data_cols.mean(axis=0, skipna=True)

# print(means)

test_ids_filepath = 'data/test_ids.csv'
output_df = pd.read_csv(test_ids_filepath, sep=',', header='infer')

output_data  = data_cols.iloc[:0,:].copy()

for idx, series in output_df.iterrows():
    path = series['Path']
    Id = series['Id']
    output_data.loc[idx] = means 

# print(output_data)
output_df = output_df.drop('Path', axis=1)
combined  = pd.concat([output_df, output_data], axis=1)

combined.to_csv("outputs/SampleOutput.csv", sep=',', header=False, index=False)