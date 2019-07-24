from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import make_scorer
from genetic_selection import GeneticSelectionCV
from scipy import interp
import pylab as pl

file_name = '../../../../data/features/convolutional_features/model_3/dense2_shape.csv'
dataFrame = pd.read_csv(file_name)

scaler = MinMaxScaler(copy=False)
X = scaler.fit_transform(dataFrame[dataFrame.columns[:-1]])
y = dataFrame[dataFrame.columns[-1]]

#'C': 3.9735397103241574, 'gamma': 0.2119916750368503, 'kernel': 'linear'

clf = svm.SVC(C = 3.9735397103241574, gamma = 0.2119916750368503, kernel = 'linear', probability=True)

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

selected_features = []
for i, bol in enumerate(selector.support_):
      if (bol):
            selected_features.append(i)

print(selected_features)
print("List size: " + str(len(selected_features)))

X_selected = X[:,selected_features]

def specificity(y_true, y_predicted): 
    true_negative  = confusion_matrix(y_true, y_predicted)[0, 0]
    false_positive = confusion_matrix(y_true, y_predicted)[0, 1]

    return (true_negative)/(true_negative + false_positive)

scoring = {'accuracy': 'accuracy', 'specificity': make_scorer(specificity), 
              'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

cv = StratifiedKFold(10)
scores = cross_validate(clf, X_selected, y, scoring=scoring, cv=cv)

print("Results -------| " + file_name + " |-------")
print(" Time to validate:",                (np.sum(scores['fit_time']) + np.sum(scores['score_time']))/60, " minutes")
print(" Accuracy: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['test_accuracy']),     np.std(100*scores['test_accuracy'])))
print(" Specificity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['test_specificity']),  np.std(100*scores['test_specificity'])))
print(" Sensitivity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['test_recall']),       np.std(100*scores['test_recall'])))
print(" F1-score: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['test_f1']),           np.std(100*scores['test_f1'])))
print(" AUC: %.2f (+/- %.2f)"              % (np.mean(scores['test_roc_auc']),          np.std(scores['test_roc_auc'])))
print()

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv.split(X_selected,y)):
    probas_ = clf.fit(X_selected[train], y[train]).predict_proba(X_selected[test])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= 10
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
pl.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' %mean_auc, lw=2)

pl.xlim([-0.05, 1.05])
pl.ylim([-0.05, 1.05])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC Curve Behavior')
pl.legend(loc="lower right")
pl.show()