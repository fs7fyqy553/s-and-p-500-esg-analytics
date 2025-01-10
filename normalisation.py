import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('./trimmedData.csv')

columns_to_normalise = ['Environment Risk Score', 'Social Risk Score', 'Governance Risk Score']

scaler = MinMaxScaler()

# Performing normalisation of the risk scores to the [0,1] range
data[columns_to_normalise] = scaler.fit_transform(data[columns_to_normalise])

# Saving the result
data.to_csv('normalisedTrimmedData.csv', index=False)