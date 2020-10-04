import argparse
import os
import time

import numpy as np
import pandas as pd
import scipy.stats as stats
from brkga import GeneticSelection
from prediction import predict_model, metrics_by_model, plot_roc_auc
from scipy.stats import mannwhitneyu
from sklearn import svm
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-f", "--features", required=True, help="Features folder")
args = vars(argument_parser.parse_args())
print(args)

PERCENTILE = 50
SCORE_FUNC = f_classif

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


def getSelectedFeaturesAsIndexList(boolean_list):
    selected_features = []

    for i, boolean_value in enumerate(boolean_list):
        if (boolean_value):
            selected_features.append(i)

    return (selected_features)


def f1(y_true, y_predicted):
    true_positive = confusion_matrix(y_true, y_predicted)[1, 1]
    false_positive = confusion_matrix(y_true, y_predicted)[0, 1]
    false_negative = confusion_matrix(y_true, y_predicted)[1, 0]

    return 2 * (true_positive) / (2 * true_positive + false_positive + false_negative)


def specificity(y_true, y_predicted):
    true_negative = confusion_matrix(y_true, y_predicted)[0, 0]
    false_positive = confusion_matrix(y_true, y_predicted)[0, 1]

    return (true_negative) / (true_negative + false_positive)


def sensitivity(y_true, y_predicted):
    true_positive = confusion_matrix(y_true, y_predicted)[1, 1]
    false_negative = confusion_matrix(y_true, y_predicted)[1, 0]

    return (true_positive) / (true_positive + false_negative)


# End Functions ----------------------------------------------------------------------

# Setup ------------------------------------------------------------------------------
features_folder_path = args["features"]
feature_file_name_list = os.listdir(features_folder_path)

random_seed = 0

for feature_file_name in feature_file_name_list:
    features_file = features_folder_path + feature_file_name
    experiment_name = features_file.split('/')[-1].split('.')[-2]

    random = np.random.RandomState(random_seed)

    print("EXPERIMENT: ", end='')
    print(experiment_name)

    if ('dense2' in experiment_name):
        continue

    dataFrame = pd.read_csv(features_file)

    scaler = MinMaxScaler(copy=False)

    X = scaler.fit_transform(dataFrame[dataFrame.columns[:-1]])
    y = dataFrame[dataFrame.columns[-1]]

    c_expon_gen = stats.expon(scale=500, loc=0)
    gamma_expon_gen = stats.expon(scale=0.9, loc=0)

    c_space = []
    gamma_space = []

    for i in range(30):
        c = c_expon_gen.rvs()

        while (c < 0 or c > 500):
            c = c_expon_gen.rvs()

        c_space.append(c)

        g = gamma_expon_gen.rvs()

        while (g < 0 or g >= 1):
            g = gamma_expon_gen.rvs()

        gamma_space.append(g)

    param_dist = {'C': c_space, 'gamma': gamma_space,
                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

    # End setup -------------------------------------------------------------------------

    # Random Search ---------------------------------------------------------------------
    clf = svm.SVC(random_state=0)

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=100, cv=10, iid=False,
                                       verbose=0, n_jobs=-1, scoring='roc_auc')
    random_search.fit(X, y)

    random_searchcv_results = random_search.cv_results_

    C = random_searchcv_results['params'][0]['C']
    gamma = random_searchcv_results['params'][0]['gamma']
    kernel = random_searchcv_results['params'][0]['kernel']

    # End Random Search ------------------------------------------------------------------

    # Features Optimization --------------------------------------------------------------

    clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=0)
    print('SVM params')
    print(clf)

    selector = GeneticSelection(n_population=50,
                                n_generations=100,
                                elite_solutions_size=10,
                                n_gen_no_change=10,
                                elite_inheritance_proba=0.70,
                                mutants_solutions_size=20)

    population = selector.fit((X, y), clf)

    best_individual = population[0]
    best_individual = [True if feature >= 0.5 else False for feature in best_individual]

    selected_features = getSelectedFeaturesAsIndexList(best_individual)

    print("\nFeature List: ")
    print(selected_features)
    print("All features: " + str(X.shape[1]))
    print("Optmized features: " + str(len(selected_features)))

    X_selected = X[:, selected_features]
    # End Features Optimization -----------------------------------------------------------

    # No selection
    base_no_selection = (X, y)
    predictions_no_selection = predict_model(base_no_selection, clf)
    score_no_selection = metrics_by_model(predictions_no_selection)
    score_no_selection.loc['mean'] = score_no_selection.mean()
    score_no_selection.to_csv('results/conv96_dense64_drop241_dense24_drop236/017-results-with-statistics/%s-no'
                              '-selection.csv' % experiment_name)
    plot_roc_auc(predictions_no_selection, label=experiment_name + ' ROC-AUC sem seleção',
                 path='results/conv96_dense64_drop241_dense24_drop236/017-results-with-statistics/',
                 file_name='auc-%s-no-selection' % experiment_name)
    # Selection
    base_selection = (X_selected, y)
    predictions_selection = predict_model(base_selection, clf)
    score_selection = metrics_by_model(predictions_selection)
    score_selection.loc['mean'] = score_selection.mean()
    score_selection.to_csv('results/conv96_dense64_drop241_dense24_drop236/017-results-with-statistics/%s-brkga'
                           '-selection.csv' % experiment_name)
    plot_roc_auc(predictions_selection, label=experiment_name + ' ROC-AUC BRKGA seleção',
                 path='results/conv96_dense64_drop241_dense24_drop236/017-results-with-statistics/',
                 file_name='auc-%s-brkga-selection' % experiment_name)

    p_values = {}
    for key in score_selection:
        p_values[key] = mannwhitneyu(score_selection[key], score_no_selection[key])
    print(p_values)
