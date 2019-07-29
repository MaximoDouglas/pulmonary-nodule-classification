import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
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

file_name = '../../../../data/features/convolutional_features/model_3/flatten/flatten_shape.csv'
dataFrame = pd.read_csv(file_name)

scaler = MinMaxScaler(copy=False)
X = scaler.fit_transform(dataFrame[dataFrame.columns[:-1]])
y = dataFrame[dataFrame.columns[-1]]

clf = svm.SVC()
param_dist = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

n_iter_search = 30
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=10, iid=False,
                                   verbose=2, n_jobs=-1)
random_search.fit(X, y)

report(random_search.cv_results_)
print(file_name)