import numpy as np
import pandas as pd

attribute_dict = {}
a_file = open("attribute_names.txt")
for line in a_file:
    key, value = line.strip('\n').split(":")
    key = key.strip()
    attribute_dict[key] = value

df_german_data = pd.read_csv('german_data.csv', index_col=0)
df_german_data = df_german_data.rename_axis('id').reset_index()
data = df_german_data.to_numpy()
data_shape = data.shape
data_new = []
for d in data.flatten():
    if d in attribute_dict:
        d = attribute_dict[d]
    data_new.append(d)
data_new = np.stack(data_new).reshape(data_shape)
df_german_data_new = pd.DataFrame(data_new, columns=df_german_data.columns)
df_german_data_new.to_csv('german_data_translated.csv', index=False)
