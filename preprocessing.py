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
# plt.show()
'''
# creating a mask for ibeacon values above -90 and ploting a bar graph against the locations of users
for col in data.columns[2:]:
    plt.figure(figsize=(7, 5))
    mask = data[col] > -90
    a = data.loc[mask, 'location'].value_counts()
    plt.title("strength of beacon at particular location")
    plt.xlabel("User Location")
    plt.ylabel("Frequency")
    a = a.plot(kind='bar')
    a.grid()
    plt.show()

for col in data.columns[2:]:
    data.hist(column=col)
    plt.xlabel("Signal Strength")  # Set the custom label for the x-axis
    plt.ylabel("Count")
    # plt.show()
'''
'''values=data.iloc[:,2:]
a=values.corr()
fig=a.plot()
fig.set_title('Correlation between beacons')
fig.set_xlabel('Beacon')
fig.set_ylabel('Correlation coefficient')
plt.show()'''

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


