import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

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

file_name = '../../../../data/features/convolutional_features/model_1/dense2_all.csv'
dataFrame = pd.read_csv(file_name)

scaler = MinMaxScaler(copy=False)
X = scaler.fit_transform(dataFrame[dataFrame.columns[:-1]])
y = dataFrame[dataFrame.columns[-1]]

clf = svm.SVC()

c_expon_gen     = stats.expon(scale=500, loc=0)
gamma_expon_gen = stats.expon(scale=0.9, loc=0)

c_space         = []
gamma_space     = []

for i in range(30):
      c = c_expon_gen.rvs()

      while(c < 0 or c > 500):
            c = c_expon_gen.rvs()
      
      c_space.append(c)

      g = gamma_expon_gen.rvs()

      while(g < 0 or g >= 1):
            g = gamma_expon_gen.rvs()
      
      gamma_space.append(g)

param_dist = {'C': c_space, 'gamma': gamma_space,
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=10, iid=False,
                                   verbose=2, n_jobs=-1, scoring='roc_auc')
random_search.fit(X, y)

report(random_search.cv_results_)
print(file_name)