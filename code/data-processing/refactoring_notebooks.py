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

file_name  = '../../data/features/convolutional_features/conv1_no_augmentation_balanced/dense_layer_2_all.csv'

dataFrame = pd.read_csv(file_name)

X = dataFrame[dataFrame.columns[:-1]]
y = dataFrame[dataFrame.columns[-1]]

print(y.value_counts())