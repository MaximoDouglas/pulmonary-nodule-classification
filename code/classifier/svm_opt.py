from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
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

file_name  = '../../data/features/convolutional_features/conv1_no_augmentation_balanced/dense_layer_1_all.csv'

dataFrame = pd.read_csv(file_name)

X = dataFrame[dataFrame.columns[:-1]]
y = dataFrame[dataFrame.columns[-1]]

clf = svm.SVC()

param_dist = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
              'kernel': ['rbf', 'linear']}

n_iter_search = 10
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=10, iid=False)

random_search.fit(X, y)

report(random_search.cv_results_)