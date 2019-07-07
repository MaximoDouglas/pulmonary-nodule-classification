import pandas as pd

df = pd.read_csv('../../data/features/solidNodules.csv')
print('differenceends:     ', df.columns.get_loc('differenceends'))
print('idm135_N:    ', df.columns.get_loc('idm135_N'))
print(df.columns[[61, 42]].tolist())

'''
differenceends: 61
idm135_N: 42
'''

'''
48 features:

energy0_N
entropy0_N
inertia0_N
homogeneity0_N
correlation0_N
shade0_N
promenance0_N
variance0_N
idm0_N
'''