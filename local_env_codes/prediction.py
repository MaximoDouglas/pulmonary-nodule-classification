import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler
from utilities import binarize, mkdir_p

def remove_features(df, features):
    ''' Remove columns features from a dataset/dataframe
    
    Parameters
    ----------
    df : Pandas dataframe or list dataset
    features : list
        A binary list of features 1 refers to presence of a feature, 0 refers to
        it's absense

    Returns
    -------
    numpy.ndarray or pandas dataframe
        a dataframe/dataset with only selected features
    '''



def predict_model(base, model, features=[], random_state=0, k_folds=10, n_repeats=1):
    ''' Train and test a prediction model with crossvalidation
    
    Parameters
    ----------
    base : tuple (<PANDAS DF>, <PANDAS SERIES>)
        where the first value is a pandas dataframe with non class features of a
        dataset, the second value correspond to class value for each sample.
    model : classification model instance
    features : list
        A binary list of features 1 refers to presence of a feature, 0 refers to
        it's absense
    sampling : sampling class instance
    kfolds : int
    n_repeats : int
    Returns
    -------
    dict[<BASE_NAME>][<MODEL_NAME>]
        [<FEATURES>] : Pandas dataframe
            A dataframe with features of the model prediction
        [<KFOLDS>][<KFOLD>]
            [<Y_TRUE>] : list
                True values of test kfold samples
            [<Y_PRED>] : list
                Predicted values of test kfold samples
            [<IMPORTANCES>] : Pandas dataframe
                Gini importance of features for each kfold
 
    '''
    random = np.random.RandomState(random_state) if random_state is int() else random_state
    rskf = RepeatedStratifiedKFold(n_splits=k_folds, random_state=random, n_repeats=n_repeats)
    # sampling = SMOTE(random_state=random)
    X, y = base

    selected_features = [True if val == 1 else False for val in features]
    X = X[:,selected_features]

    predictions_dict = {}
    predictions_dict['kfolds'] = {}
    start_time = time.time()
    
    for i, (train, test) in enumerate(rskf.split(X, y)):
        X_train, y_train = (X[train], y[train])
        X_test, y_test = (X[test], y[test])
        model.fit(X_train, y_train)
        try:
            feature_importances = model.feature_importances_
        except:
            feature_importances = []
        y_pred = model.predict_proba(X_test)
        predictions_dict['kfolds']['rep%dkf%d'%(i//n_repeats,i)] = {
            'y_true':y_test,
            'y_pred':y_pred[:, 1]
            # 'importances': pd.DataFrame(feature_importances, removed_feature_names)
        }

    classification_time = time.time() - start_time
    return predictions_dict


def metrics_by_prediction(y_true, y_pred):
    ''' Applies metrics to comparisson of predicted and true values
    
    Parameters
    ----------
    y_true : list
        True binary labels or binary label indicators.
    y_pred : list
        Target scores, can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions.
    
    Returns
    -------
    dict
        a dict of int, float, list, representing each calculated metric
    '''
    metrics = {}
    y_bin = binarize(y_pred, threshold=0.5)

    accuracy = accuracy_score(y_true, y_bin)
    average_precision = average_precision_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_bin)
    f1 = f1_score(y_true, y_bin)
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr_roc, tpr_roc)

    tn, fp, fn, tp = conf_matrix.ravel()

    metrics['accuracy'] = accuracy
    metrics['average_precision'] = average_precision
    metrics['f1'] = f1
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    metrics['tn'] = tn
    metrics['ppv'] = tp/(tp+fp)
    metrics['tpr'] = tp/(tp+fn)
    metrics['tnr'] = tn/(tn+fp)
    metrics['fpr_roc'] = fpr_roc
    metrics['tpr_roc'] = tpr_roc
    metrics['roc_auc'] = roc_auc
    
    return metrics

def metrics_by_model(model_pred, write=False, path='', file_name=''):
    ''' Organizes the metrics for each kfold and saves it in a file

    Parameters
    ----------
    model_pred : dict
        A dict that follows prediction[<FOLD>][<METRICS>] : int, float, list
    write : bool
        Defines if the metrics should be saved in a file or not
    path : str
    file_name : str

    The path of the file is <PATH>/metrics/<FILE_NAME>.csv
    '''
    metrics_dict = {}
    kfolds_pred = model_pred['kfolds']
    
    for kfold in kfolds_pred:
        metrics = metrics_by_prediction(kfolds_pred[kfold]['y_true'], 
                                        kfolds_pred[kfold]['y_pred'])
        for metric in metrics:
            value = metrics[metric]
            if isinstance(value, int) or isinstance(value, float):
                if not metric in metrics_dict.keys():
                    metrics_dict[metric] = []                  
                metrics_dict[metric].append(value)
    
    metrics_dataframe = pd.DataFrame.from_dict(metrics_dict)
    
    if write:
        file_path = '%s/metrics'%path
        mkdir_p(file_path)
        file_path = '%s/%s.csv'%(file_path, file_name)
        metrics_dataframe.to_csv(file_path, index=False)
    return metrics_dataframe
