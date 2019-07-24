from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, SCORERS
from sklearn.metrics import fbeta_score, make_scorer
from os import listdir
from os.path import isfile, join
import scipy

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

file_name  = '../../data/features/solidNodules.csv'

scaler = MinMaxScaler(copy=False)

dataFrame = pd.read_csv(file_name)

X = scaler.fit_transform(dataFrame[dataFrame.columns[2:74]])
y = pd.factorize(dataFrame[dataFrame.columns[75]])[0]

clf = svm.SVC()
param_dist = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=10, iid=False,
                                   verbose=2, n_jobs=-1)
random_search.fit(X, y)

report(random_search.cv_results_)


'''
Result last optimization:
      Mean validation score: 0.835 (std: 0.031)
      Parameters: {'C': 88.22932067066742, 'gamma': 0.04329670049462131, 'kernel': 'rbf'}

      Model with rank: 2
      Mean validation score: 0.834 (std: 0.033)
      Parameters: {'C': 31.345141775777858, 'gamma': 0.07419184831017692, 'kernel': 'rbf'}

      Model with rank: 3
      Mean validation score: 0.831 (std: 0.035)
      Parameters: {'C': 218.00821997273036, 'gamma': 0.046451200825212986, 'kernel': 'linear'}
'''