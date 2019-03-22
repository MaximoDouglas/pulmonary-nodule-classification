# List of imports to test the enviroment on the remote computer
import math
import itertools
import re
import os
import time
import gc
import shutil

import numpy as np
import tensorflow as tf

from scipy import misc
from scipy.ndimage import rotate

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold

from keras import backend as K
from keras import optimizers
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
import keras_metrics as km

from IPython.display import SVG
from tqdm import tqdm

from import_images import get_folds
