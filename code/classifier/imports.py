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
