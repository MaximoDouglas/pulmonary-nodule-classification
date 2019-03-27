import os
import time
import gc
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.metrics import confusion_matrix, roc_curve, auc

import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.backend.tensorflow_backend import set_session
import keras_metrics as km
from keras.utils import to_categorical

from IPython.display import SVG

from import_images import get_folds

def data():
    X_train = np.load("../data-6-first/X_train.npy")
    X_test = np.load("../data-6-first/X_test.npy")
    Y_train = np.load("../data-6-first/Y_train.npy")
    Y_test = np.load("../data-6-first/Y_test.npy")

    return X_train, Y_train, X_test, Y_test

'''conv1 = [32, 48, 64, 96]
dense1 = [64, 96, 128, 256]
dense2 = [16, 24, 32, 64]
lrate = [0.0001, 0.00001]'''

def model(X_train, Y_train, X_test, Y_test):
    conv1 = {{choice([32, 48, 64, 96])}}
    dense1 = {{choice([32, 64, 96, 128])}}
    dense2 = {{choice([16, 24, 32])}}

    input_layer = Input(X_train.shape[1:5])

    conv_layer1 = Conv3D(conv1, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer1)

    flatten_layer = Flatten()(pooling_layer1)

    dense_layer1 = Dense(dense1, activation='relu')(flatten_layer)
    dense_layer1 = Dropout({{uniform(0, .5)}})(dense_layer1)

    dense_layer2 = Dense(dense2, activation='relu')(dense_layer1)
    dense_layer2 = Dropout({{uniform(0, .5)}})(dense_layer2)

    output_layer = Dense(units=1, activation='sigmoid')(dense_layer2)

    model = Model(inputs=input_layer, outputs=output_layer)

    opt = optimizers.RMSprop(lr={{choice([0.0001, 0.00001])}})

    model.compile(loss=binary_crossentropy, optimizer=opt, metrics=['acc'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

    model.fit(X_train, Y_train,
              batch_size=128,
              epochs=10,
              verbose=2,
              validation_data=(X_test, Y_test),
              callbacks=[early_stop]
              )

    score, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Test accuracy online:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def opt():
    X_train, Y_train, X_test, Y_test = data()

    start = time.time()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials(),
                                          verbose=False)
    end = time.time()

    print("Tempo da otimização: ", end - start, " segundos")

    return best_run, best_model

def result():

    best_run, best_model = opt()
    X_train, Y_train, X_test, Y_test = data()

    print("Details of best performing model: ------------")

    print("Acc on test data: ")
    print(best_model.evaluate(X_test, Y_test))

    print("Summary: ")
    print(best_model.summary())

    print("Best run: ")
    print(best_run)

    lrate = lrate[best_run['lr']]
    c1 = conv1[best_run['conv1']]
    d1 = dense1[best_run['dense1']]
    d2 = dense1[best_run['dense2']]
    drop1 = best_run['Dropout']
    drop2 = best_run['Dropout_1']

    print("Archtecture: ")
    print('Conv 1:', c1, 'unidades')
    print('Dense 1:', d1, 'unidades')
    print('Dense 2:', d2, 'unidades')
    print('Dropout 1:', drop1)
    print('Dropout 2:', drop2)
    print('Learning Rate:', lrate)
