from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, roc_curve, auc, SCORERS
from sklearn.metrics import fbeta_score, make_scorer
from os import listdir
from os.path import isfile, join

base_dir  = '../../data/features/convolutional_features/conv1_no_augmentation_balanced/'
onlyfiles = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

for file_name in onlyfiles:
  dataFrame = pd.read_csv(base_dir + file_name)

  X = dataFrame[dataFrame.columns[:-1]]
  y = dataFrame[dataFrame.columns[-1]]

  clf = svm.SVC(kernel='rbf', gamma='scale', C=1)

  def specificity(y_true, y_predicted): 
    true_negative  = confusion_matrix(y_true, y_predicted)[0, 0]
    false_positive = confusion_matrix(y_true, y_predicted)[0, 1]

    return (true_negative)/(true_negative + false_positive)

  scoring = {'accuracy': 'accuracy', 'specificity': make_scorer(specificity), 
              'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

  scores = cross_validate(clf, X, y, scoring=scoring, cv=10)

  print("Results -------| conv1_noaug_balanced :> "+ file_name + " |-------")
  print(" Time to validate:",                (np.sum(scores['fit_time']) + np.sum(scores['score_time']))/60, " minutes")
  print(" Accuracy: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['test_accuracy']),     np.std(100*scores['test_accuracy'])))
  print(" Specificity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['test_specificity']),  np.std(100*scores['test_specificity'])))
  print(" Sensitivity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['test_recall']),       np.std(100*scores['test_recall'])))
  print(" F1-score: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['test_f1']),           np.std(100*scores['test_f1'])))
  print(" AUC: %.2f (+/- %.2f)"              % (np.mean(scores['test_roc_auc']),          np.std(scores['test_roc_auc'])))
  print()
