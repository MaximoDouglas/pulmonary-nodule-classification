from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv('../../data/features/solidNodules.csv')

scaler = MinMaxScaler(copy=False)
df[['differenceends', 'idm135_N']] = scaler.fit_transform(df[['differenceends', 'idm135_N']])

print(df['idm135_N'].max())
print(df['idm135_N'].min())
