import pandas as pd
import os

filepath = os.path.join(os.getcwd(), "/central/groups/CS156b/2023/Red_Door_Sandwich/data/student_labels/train2023.csv")

df = pd.read_csv(filepath, sep=',', header='infer')

labels = df.columns[-9:]
data_cols = df[labels]

means = data_cols.mean(axis=0, skipna=True)

# print(means)

test_ids_filepath = '/central/groups/CS156b/2023/Red_Door_Sandwich/data/student_labels/test_ids.csv'
output_df = pd.read_csv(test_ids_filepath, sep=',', header='infer')

output_data  = data_cols.iloc[:0,:].copy()

for idx, series in output_df.iterrows():
    path = series['Path']
    Id = series['Id']
    output_data.loc[idx] = means 

# print(output_data)
combined  = pd.concat([output_df, output_data], axis=1)

combined.to_csv("/groups/CS156b/2023/Red_Door_Sandwich/Project156b-Red-Door-Sandwich/outputs/SampleOutput.csv", sep=',', header=True, index=False)