from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, SCORERS
from sklearn.metrics import fbeta_score, make_scorer
from os import listdir
from os.path import isfile, join
import scipy
from genetic_selection import GeneticSelectionCV
import math

file_name  = '../../data/features/solidNodules.csv'

dataFrameOld = pd.read_csv(file_name)
dataFrameNew = dataFrameOld[dataFrameOld.columns[2:74]]

scaler = MinMaxScaler(copy=False)
X = scaler.fit_transform(dataFrameOld[dataFrameOld.columns[2:74]])
#y = pd.factorize(dataFrame[dataFrame.columns[75]])[0]

X_optimized_ind_new_df = [0, 1, 2, 4, 7, 9, 10, 16, 17, 20, 24, 26, 27, 
                  28, 29, 30, 31, 33, 36, 38, 39, 42, 44, 45, 46, 
                  48, 51, 52, 54, 55, 57, 58, 60, 61, 63, 65, 69, 70]

X_optimized_ind_old_df = []

for ind in X_optimized_ind_new_df:
      X_optimized_ind_old_df.append(ind + 2)

print(X_optimized_ind_old_df)

print(dataFrameOld.columns[X_optimized_ind_old_df] == dataFrameNew.columns[X_optimized_ind_new_df])

print()

'''clf = svm.SVC(C = 88.22932067066742, gamma=0.04329670049462131, kernel = 'rbf')

selector = GeneticSelectionCV(clf,
                              cv=10,
                              verbose=1,
                              scoring="roc_auc",
                              max_features=X.shape[1],
                              n_population=50,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=100,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=10,
                              caching=False,
                              n_jobs=-1)

selector = selector.fit(X, y)

print(selector.support_)

selected_features = []

for i, bol in enumerate(selector.support_):
      if (bol):
            selected_features.append(i)

print(selected_features)

X_selected = X[:,selected_features]

def specificity(y_true, y_predicted): 
    true_negative  = confusion_matrix(y_true, y_predicted)[0, 0]
    false_positive = confusion_matrix(y_true, y_predicted)[0, 1]

    return (true_negative)/(true_negative + false_positive)

scoring = {'accuracy': 'accuracy', 'specificity': make_scorer(specificity), 
              'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

scores = cross_validate(clf, X_selected, y, scoring=scoring, cv=StratifiedKFold(10))

print("Results -------| " + file_name + " |-------")
print(" Time to validate:",                (np.sum(scores['fit_time']) + np.sum(scores['score_time']))/60, " minutes")
print(" Accuracy: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['test_accuracy']),     np.std(100*scores['test_accuracy'])))
print(" Specificity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['test_specificity']),  np.std(100*scores['test_specificity'])))
print(" Sensitivity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['test_recall']),       np.std(100*scores['test_recall'])))
print(" F1-score: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['test_f1']),           np.std(100*scores['test_f1'])))
print(" AUC: %.2f (+/- %.2f)"              % (np.mean(scores['test_roc_auc']),          np.std(scores['test_roc_auc'])))
print()'''

'''
Last optimization resul:
Using the following array:

Para o slice [2:74]:
      [0, 1, 2, 4, 
      7, 9, 10, 16, 
      17, 20, 24, 26, 
      27, 28, 29, 30, 
      31, 33, 36, 38, 
      39, 42, 44, 45, 
      46, 48, 51, 52, 
      54, 55, 57, 58, 
      60, 61, 63, 65, 
      69, 70]

Para todas as colunas
      [2, 3, 4, 6, 
      9, 11, 12, 18, 
      19, 22, 26, 28, 
      29, 30, 31, 32, 
      33, 35, 38, 40, 
      41, 44, 46, 47, 
      48, 50, 53, 54, 
      56, 57, 59, 60, 
      62, 63, 65, 67, 
      71, 72]


CV results:
      Accuracy: 84.24% (+/- 2.72%)
      Specificity: 90.27% (+/- 4.60%)
      Sensitivity: 70.36% (+/- 10.57%)
      F1-score: 72.72% (+/- 5.54%)
      AUC: 0.91 (+/- 0.02)'''