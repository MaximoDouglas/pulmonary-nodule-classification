import os

import time

import gc

import numpy as np

from scipy import interp

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc

from keras import backend as K
from keras import optimizers
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model, model_to_dot

import keras_metrics as km

import math
import itertools
import re
import os
import imageio
from scipy.ndimage import rotate
from sklearn.model_selection import KFold
from tqdm import tqdm
import shutil

from import_remoto import get_folds

c1 = 32
d1 = 96
d2 = 16
drop1 = 0.07176927609418894
drop2 = 0.2837032463233716

def get_model():
    K.clear_session()
    gc.collect()
    
    input_layer = Input(shape=(64, 64, 5, 1))

    conv_layer1 = Conv3D(filters=c1, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer1)

    flatten_layer = Flatten()(pooling_layer1)

    dense_layer1 = Dense(units=d1, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(drop1)(dense_layer1)

    dense_layer2 = Dense(units=d2, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(drop2)(dense_layer2)

    output_layer = Dense(units=1, activation='sigmoid')(dense_layer2)

    model = Model(inputs=input_layer, outputs=output_layer)

    opt = optimizers.RMSprop(lr=0.0001)

    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy', km.binary_true_positive(), km.binary_true_negative(), km.binary_false_positive(), km.binary_false_negative(), km.binary_f1_score()])

    return model

metrics = {'acc': [], 'spec': [], 'sens': [], 'f1_score': [], 'auc': []}

def sensitivity(tp, fn):
    return tp/(tp+fn)

def specificity(tn, fp):
    return tn/(tn+fp)

tprs = []
base_fpr = np.linspace(0, 1, 101)

start = time.time()

N_SLICES = 5

for i in range(1):
    m = {'acc': [], 'spec': [], 'sens': [], 'f1_score': [], 'auc': []}
    
    X_train_, X_test_, f_train, f_test, Y_train_, Y_test_= get_folds("../solid-nodules-with-attributes/", 
                                                                        n_slices=N_SLICES, 
                                                                        strategy='first', 
                                                                        repeat=False,
                                                                        features="../features/solidNodules.csv")
    
    for X_train, X_test, Y_train, Y_test in zip(X_train_, X_test_, Y_train_, Y_test_):
        model = get_model()
        
        model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=0)

        scores = model.evaluate(X_test, Y_test, verbose=0)

        tp, tn, fp, fn = scores[2], scores[3], scores[4], scores[5]
        
        acc = scores[1]*100
        spec = specificity(tn, fp)*100
        sens = sensitivity(tp, fn)*100
        f1_score = scores[6]*100
        
        # AUC
        pred = model.predict(X_test).ravel()
        fpr, tpr, thresholds_keras = roc_curve(Y_test, pred)
        auc_val = auc(fpr, tpr)
        
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    
        m['acc'].append(acc)
        m['spec'].append(spec)
        m['sens'].append(sens)
        m['f1_score'].append(f1_score)
        m['auc'].append(auc_val)
        
        print("acc: %.2f%% spec: %.2f%% sens: %.2f%% f1: %.2f%% auc: %.2f" % (acc, spec, sens, f1_score, auc_val))
        
    metrics['acc'] = metrics['acc'] + m['acc']
    metrics['spec'] = metrics['spec'] + m['spec']
    metrics['sens'] = metrics['sens'] + m['sens']
    metrics['f1_score'] = metrics['f1_score'] + m['f1_score']
    metrics['auc'] = metrics['auc'] + m['auc']
    
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(m['acc']), np.std(m['acc'])))
    print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(m['spec']), np.std(m['spec'])))
    print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(m['sens']), np.std(m['sens'])))
    print("F1-score: %.2f%% (+/- %.2f%%)" % (np.mean(m['f1_score']), np.std(m['f1_score'])))
    print("AUC: %.2f (+/- %.2f)" % (np.mean(m['auc']), np.std(m['auc'])))
    
end = time.time()

print()
print("Results ------------------------------------------")
print("Tempo para validação:", (end - start)/60, "minutos")
print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(metrics['acc']), np.std(metrics['acc'])))
print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(metrics['spec']), np.std(metrics['spec'])))
print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(metrics['sens']), np.std(metrics['sens'])))
print("F1-score: %.2f%% (+/- %.2f%%)" % (np.mean(metrics['f1_score']), np.std(metrics['f1_score'])))
print("AUC: %.2f (+/- %.2f)" % (np.mean(metrics['auc']), np.std(metrics['auc'])))