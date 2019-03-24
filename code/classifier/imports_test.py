# List of imports to test the enviroment on the remote computer
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold

import keras_metrics as km

from tqdm import tqdm

from import_images_gpu import get_folds
