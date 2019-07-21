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
from genetic_selection import GeneticSelectionCV
import math

file_name  = '../../data/features/convolutional_features/conv1_no_augmentation_balanced/dense_layer_1_none.csv'

dataFrame = pd.read_csv(file_name)

X = dataFrame[dataFrame.columns[:-1]]
y = dataFrame[dataFrame.columns[-1]]

clf = svm.SVC(C=8.021483799761896, gamma=0.08627040154277943, kernel='linear')

selector = GeneticSelectionCV(clf,
                              cv=10,
                              verbose=1,
                              scoring="recall",
                              max_features=math.floor((.80)*(X.shape[1])),
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

X_selected = X[X.columns[selected_features]]

def specificity(y_true, y_predicted): 
    true_negative  = confusion_matrix(y_true, y_predicted)[0, 0]
    false_positive = confusion_matrix(y_true, y_predicted)[0, 1]

    return (true_negative)/(true_negative + false_positive)

scoring = {'accuracy': 'accuracy', 'specificity': make_scorer(specificity), 
              'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

scores = cross_validate(clf, X_selected, y, scoring=scoring, cv=10)

print("Results -------| conv1_aug_balanced :> "+ file_name + " |-------")
print(" Time to validate:",                (np.sum(scores['fit_time']) + np.sum(scores['score_time']))/60, " minutes")
print(" Accuracy: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['test_accuracy']),     np.std(100*scores['test_accuracy'])))
print(" Specificity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['test_specificity']),  np.std(100*scores['test_specificity'])))
print(" Sensitivity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['test_recall']),       np.std(100*scores['test_recall'])))
print(" F1-score: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['test_f1']),           np.std(100*scores['test_f1'])))
print(" AUC: %.2f (+/- %.2f)"              % (np.mean(scores['test_roc_auc']),          np.std(scores['test_roc_auc'])))
print()