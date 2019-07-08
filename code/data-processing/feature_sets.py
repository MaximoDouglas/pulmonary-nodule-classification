import pandas as pd

df = pd.read_csv('../../data/features/solidNodules.csv')

important_features = ['energy', 'entropy', 'inertia', 'homogeneity', 'correlation', 'shade', 
                      'promenance', 'variance', 'idm']


basic_set          = {'differenceends': 61, 'idm135_N': 42}
complex_set        = []
complex_dict       = {}
degrees            = [0, 45, 90, 135]


for feature in important_features:
    for degree in degrees:
        loc = df.columns.get_loc(feature + str(degree) + '_N')
        complex_dict[feature + str(degree) + '_N'] = loc

print(complex_dict)

for key in complex_dict:
    complex_set.append(complex_dict[key])

print(complex_set)


'''
differenceends: 61
idm135_N:       42
'''