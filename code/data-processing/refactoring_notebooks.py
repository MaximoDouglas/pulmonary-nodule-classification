import pandas as pd

'''
intensity = ['energy_N',	'entropy_N',	'kurtosis_N',	'maximum_N',	'mean_N',	
              'meanAbsoluteDeviation_N',	median_N',	'minimum_N',	'range_N',	'rootMeanSquare_N',
              'skewness_N',	'standardDeviation_N',	'uniformity_N',	'variance_N']

shape = ['differenceends', 'sumvalues',	'sumsquares',	'sumlogs', 'amean', 'gmean',	
'pvariance',	'svariance',	'sd',	'kurtosis',	'skewness', 'scm']
'''

feat_list = ['differenceends', 'sumvalues',	'sumsquares',	'sumlogs', 'amean', 'gmean',	
'pvariance',	'svariance',	'sd',	'kurtosis',	'skewness', 'scm']

file_name  = '../../data/features/solidNodules.csv'

df = pd.read_csv(file_name)

feat = []
for feature in feat_list:
  loc = df.columns.get_loc(feature)
  print(df.columns[loc])
  feat.append(loc)

print(feat)