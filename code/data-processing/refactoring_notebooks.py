import os
import time
import gc
import numpy as np
import pandas as pd
from scipy import interp
from hyperopt import Trials, STATUS_OK, tpe
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.model_selection import KFold
from keras import backend as K
from keras import optimizers
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input, concatenate
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
from tqdm import tqdm
import shutil

c1 = 64
d1 = 64
d2 = 16
drop1 = 0.1141906556451216
drop2 = 0.27112063453183133

def get_model():
    input_layer = Input((2,))

    conv_layer1 = Conv3D(filters=c1, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer1)

    flatten_layer = Flatten(name='flatten_layer')(pooling_layer1)

    dense_layer1 = Dense(units=d1, activation='relu', name='dense_layer_1')(flatten_layer)
    dense_layer1 = Dropout(drop1, name='drop_1')(dense_layer1)

    dense_layer2 = Dense(units=d2, activation='relu', name='dense_layer_2')(dense_layer1)
    dense_layer2 = Dropout(drop2, name='drop_2')(dense_layer2)

    output_layer = Dense(units=1, activation='sigmoid')(dense_layer2)

    model = Model(inputs=input_layer, outputs=[output_layer])

    opt = optimizers.RMSprop(lr=0.0001)

    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['accuracy', km.binary_true_positive(), km.binary_true_negative(), km.binary_false_positive(), km.binary_false_negative(), km.binary_f1_score()])

    return model

N_SLICES = 5

destination_folder = "../../data/convolutional_features/conv1_no_augmentation_unbalanced/"
layers             = ['flatten_layer', 'dense_layer_1', 'dense_layer_2']

feat_sets = {'_all': [-1], 
             '_set_2': [42, 61], 
             '_set_36': [25, 43, 52, 34, 26, 44, 53, 35, 27, 45, 54, 36, 28, 46, 55, 
              37, 29, 47, 56, 38, 30, 48, 57, 39, 31, 49, 58, 40, 32, 50, 
              59, 41, 33, 51, 60, 42],
             '_set_48': [25, 43, 52, 34, 26, 44, 53, 35, 27, 45, 54, 36, 28, 46, 55, 37, 
              29, 47, 56, 38, 30, 48, 57, 39, 31, 49, 58, 40, 32, 50, 59, 41, 
              33, 51, 60, 42, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
             '_none': [-1]}

for i in range(1):
    features_names = pd.read_csv("../../data/features/solidNodules.csv").columns
    result_dfs     = {}

    X_train_, X_test_, f_train_, f_test_, Y_train_, Y_test_ = mock_data()
    
    for X_train, X_test, f_train, f_test, Y_train, Y_test in zip(X_train_, X_test_, f_train_, f_test_, Y_train_, Y_test_):
        model = get_model()
        model.fit(X_train, Y_train, batch_size=1, epochs=10, verbose=0)
        
        for layer in layers:
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
            output                   = intermediate_layer_model.predict(X_test)
            features_df              = pd.DataFrame(output)

            for feature_set in feat_sets:
                file_name = destination_folder + (layer) + feature_set + ".csv"
                f_set     = feat_sets[feature_set]                
                
                if (feature_set != '_all' and feature_set != '_none'):
                    features_df = pd.concat([features_df, pd.DataFrame(f_test[f_set], 
                                            columns=features_names[f_set])], 
                                            axis=1, 
                                            sort=False)
                elif (feature_set == '_all'):
                    features_df = pd.concat([features_df, pd.DataFrame(f_test[2:74], 
                                            columns=features_names[2:74])], 
                                            axis=1, 
                                            sort=False)

                features_df['class'] = Y_test

                if (file_name not in result_dfs):
                    result_dfs[file_name] = features_df
                else:
                    result_dfs[file_name] = pd.concat([result_dfs[file_name], features_df])            

    for df in result_dfs:
        result_dfs[df].to_csv(df, index=False)

def mock_data():
    features = "../../data/features/solidNodules.csv"

    df = pd.read_csv(features)

    X_train_ = [[[0, 0], [0, 1]], [[1, 0], [1, 1]]]
    X_test_  = [[[0, 0], [0, 1]], [[1, 0], [1, 1]]]
    f_train_ = [[df.iloc[[0]], df.iloc[[1]]], 
                 [df.iloc[[2]], df.iloc[[3]]]]
    
    f_test_  = [[df.iloc[[0]], df.iloc[[1]]], 
                 [df.iloc[[2]], df.iloc[[3]]]]
    
    Y_train_ = [[0, 1], [1, 0]]
    Y_test_  = [[0, 1], [1, 0]]

    return X_train_, X_test_, f_train_, f_test_, Y_train_, Y_test_