import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from genetic_selection import GeneticSelectionCV
from scipy import interp
import pylab as plt
import time

start = time.time()

# Functions --------------------------------------------------------------------------
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

def getSelectedFeaturesAsIndexList(boolList):
  selected_features = []

  for i, bol in enumerate(boolList):
    if (bol):
      selected_features.append(i)
  
  return (selected_features)

def f1(y_true, y_predicted):
    true_positive  = confusion_matrix(y_true, y_predicted)[1, 1]
    false_positive = confusion_matrix(y_true, y_predicted)[0, 1]
    false_negative = confusion_matrix(y_true, y_predicted)[1, 0]

    return 2*(true_positive)/(2*true_positive + false_positive + false_negative)

def specificity(y_true, y_predicted): 
    true_negative  = confusion_matrix(y_true, y_predicted)[0, 0]
    false_positive = confusion_matrix(y_true, y_predicted)[0, 1]

    return (true_negative)/(true_negative + false_positive)

def sensitivity(y_true, y_predicted):
    true_positive  = confusion_matrix(y_true, y_predicted)[1, 1]
    false_negative = confusion_matrix(y_true, y_predicted)[1, 0]

    return (true_positive)/(true_positive + false_negative)

# End Functions ----------------------------------------------------------------------

# Setup ------------------------------------------------------------------------------
clf = svm.SVC()

img_name  = 'model3_flatten_all.png'
file_name = '../../model3/flatten_all.csv'
dataFrame = pd.read_csv(file_name)

scaler = MinMaxScaler(copy=False)
X = scaler.fit_transform(dataFrame[dataFrame.columns[:-1]])
y = dataFrame[dataFrame.columns[-1]]

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

# End setup -------------------------------------------------------------------------

# Random Search ---------------------------------------------------------------------
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=10, cv=10, iid=False,
                                   verbose=2, n_jobs=-1, scoring='roc_auc')
random_search.fit(X, y)

random_searchcv_results = random_search.cv_results_

C      = random_searchcv_results['params'][0]['C']
gamma  = random_searchcv_results['params'][0]['gamma']
kernel = random_searchcv_results['params'][0]['kernel']

# End Random Search ------------------------------------------------------------------

# Features Optimization --------------------------------------------------------------

clf = svm.SVC(C = C, gamma = gamma, kernel = kernel, probability=True)

selector = GeneticSelectionCV(clf,
                              cv=10,
                              verbose=1,
                              scoring="roc_auc",
                              max_features = 100 if (X.shape[1] > 100) else X.shape[1],
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

selected_features = getSelectedFeaturesAsIndexList(selector.support_)

print(selected_features)
print("All features: " + str(X.shape[1]))
print("Optmized features: " + str(len(selected_features)))

X_selected = X[:,selected_features]

# End Features Optimization -----------------------------------------------------------

# Validation --------------------------------------------------------------------------

cv = StratifiedKFold(10)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

scores = {'acc': [], 'spec': [], 'sens': [], 'f1_score': [], 'auc': []}

for i, (train, test) in enumerate(cv.split(X_selected,y)):
    clf_to_score   = clf.fit(X_selected[train], y[train])
    clf_to_predict = clf_to_score

    probas_ = clf.predict_proba(X_selected[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr    += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0

    acc     = clf_to_score.score(X_selected[test], y[test])
    roc_auc = auc(fpr, tpr)

    y_test_predicted = clf_to_predict.predict(X_selected[test])    

    spec_score = specificity(y[test], y_test_predicted)
    sens_score = sensitivity(y[test], y_test_predicted)
    f1_score   = f1(y[test], y_test_predicted)

    scores['acc'].append(acc)
    scores['spec'].append(spec_score)
    scores['sens'].append(sens_score)
    scores['f1_score'].append(f1_score)
    scores['auc'].append(roc_auc)

    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

end = time.time()

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

mean_tpr /= 10
mean_tpr[-1] = 1.0
mean_auc = np.mean(scores['auc'])
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' %mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Behavior')
plt.legend(loc="lower right")
plt.savefig(img_name)

# End Validation ----------------------------------------------------------------------

# Summary -----------------------------------------------------------------------------
print(file_name)
report(random_searchcv_results)

print("Results -------| " + file_name + " |-------")
print(" Time to validate:",                (end - start)/60, " minutes")
print(" Accuracy: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['acc']),     np.std(100*scores['acc'])))
print(" Specificity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['spec']),    np.std(100*scores['spec'])))
print(" Sensitivity: %.2f%% (+/- %.2f%%)"  % (100*np.mean(scores['sens']),    np.std(100*scores['sens'])))
print(" F1-score: %.2f%% (+/- %.2f%%)"     % (100*np.mean(scores['f1_score']),np.std(100*scores['f1_score'])))
print(" AUC: %.2f (+/- %.2f)"              % (np.mean(scores['auc']),         np.std(scores['auc'])))
print()
