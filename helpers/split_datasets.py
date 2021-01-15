import os
import pandas as pd

input_file = "factual.csv"
df = pd.read_csv(input_file)

column_names = ["text_size", "subject", "contains_name"]

root = "."
path = os.path.join(root, "../classification/splitted_datasets")
if not os.path.exists(path):
    os.mkdir(path)

for column in column_names:
    column_path = os.path.join(path, column)
    if not os.path.exists(column_path):
        os.mkdir(column_path)

    for value in df[column].unique():
        filtered_df = df[df[column] == value]
        filtered_df.drop(df.columns[[0]], axis=1, inplace=True)

        out_file = value + "_" + input_file

        print(out_file)
        filtered_df.to_csv(os.path.join(column_path, out_file))
