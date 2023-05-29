import pandas as pd

# pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

path = 'input/iBeacon_RSSI_Labeled.csv'
data = pd.read_csv(path, index_col=None)
# print(data.head(5))

loc = data.iloc[:, 0]
loc.hist(figsize=(30, 20))


data_b = data.drop('location', axis=1)
data_b = data_b.drop('date', axis=1)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(data_b.corr(method='kendall'), ax=ax)
# plt.show()

# print("data.max()")
# print(data.max())

# Splitting the location:
data['x'] = data['location'].str[0]
data['y'] = data['location'].str[1:]

# Scale and normalize the numeric columns
scaler = StandardScaler()
data[data.columns[2:-2]] = scaler.fit_transform(data[data.columns[2:-2]])

# Label Encoding

data['x'] = LabelEncoder().fit_transform(data['x'])
data['y'] = LabelEncoder().fit_transform(data['y'])

# Dropping the columns
data = data.drop(columns=["date", "location"])

target_x = data['x']
target_y = data['y']


