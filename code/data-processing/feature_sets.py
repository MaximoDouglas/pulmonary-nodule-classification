import pandas as pd

df = pd.read_csv('../../data/features/solidNodules.csv')

texture_features  = ['energy', 'entropy', 'inertia', 'homogeneity', 'correlation', 'shade', 
                        'promenance', 'variance', 'idm']
degrees           = [0, 45, 90, 135]

margin_features   = ['differenceends', 'sumvalues', 'sumsquares', 'sumlogs', 
                        'amean', 'gmean', 'pvariance', 'svariance', 'sd', 'kurtosis', 'skewness', 'scm']

basic_set          = {'differenceends': 61, 'idm135_N': 42}
complex_dict       = {}
complex_set        = []

for feature in texture_features:
    for degree in degrees:
            loc = df.columns.get_loc(feature + str(degree) + '_N')
            complex_dict[feature + str(degree) + '_N'] = loc

for feature in margin_features:
    loc = df.columns.get_loc(feature)
    complex_dict[feature] = loc

print('36 features --------------------')

print(complex_dict)

for key in complex_dict:
    complex_set.append(complex_dict[key])

print()
print('48 features --------------------')

print(complex_set)