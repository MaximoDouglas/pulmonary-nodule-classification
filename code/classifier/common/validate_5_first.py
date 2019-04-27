# Validation code - CHECK SETTINGS BEFORE USE
import time
import gc
import numpy as np

from keras import backend as K
from keras import optimizers
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input
from keras.losses import binary_crossentropy
from keras.models import Model
import keras_metrics as km

from scipy import interp

from sklearn.metrics import roc_curve, auc

from import_images import get_folds

# Begin Settings -------------------------|

# Strategy settings
N_SLICES    = 5
strategy    = 'first'
repeat      = False

# Files settings
images_path = '../solid-nodules-with-attributes/'
feat_path   = '../features/solidNodules.csv'

# Optimized params <= Found with the optimization notebook optimize-5-first
c1      = 48
d1      = 96
d2      = 24
drop1   = 0.41935233640034336
drop2   = 0.4642243750136609

# Input shape
x_shape = (64, 64, 5, 1)

# End settings ---------------------------|

# Model definition
def get_model():
    K.clear_session()
    gc.collect()
    
    input_layer = Input(shape=x_shape)

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

# Validation ----------------------------------|

metrics = {'acc': [], 'spec': [], 'sens': [], 'f1_score': [], 'auc': []}

# Calc sensitivity
def sensitivity(tp, fn):
    return tp/(tp+fn)

# Calc sensitivity
def specificity(tn, fp):
    return tn/(tn+fp)

tprs = []
base_fpr = np.linspace(0, 1, 101)

start = time.time()

# Cross-validation
for i in range(1):
    m = {'acc': [], 'spec': [], 'sens': [], 'f1_score': [], 'auc': []}
    
    X_train_, X_test_, _, _, Y_train_, Y_test_= get_folds(basedir=images_path, 
                                                                        n_slices=N_SLICES, 
                                                                        strategy=strategy, 
                                                                        repeat=repeat,
                                                                        features=feat_path)
    
    for X_train, X_test, Y_train, Y_test in zip(X_train_, X_test_, Y_train_, Y_test_):
        model = get_model()
        model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=0)

        scores = model.evaluate(X_test, Y_test, verbose=0)
        tp, tn, fp, fn = scores[2], scores[3], scores[4], scores[5]
        
        acc      = scores[1]*100
        spec     = specificity(tn, fp)*100
        sens     = sensitivity(tp, fn)*100
        f1_score = scores[6]*100
        
        # AUC
        pred    = model.predict(X_test).ravel()
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
        
    metrics['acc']      = metrics['acc'] + m['acc']
    metrics['spec']     = metrics['spec'] + m['spec']
    metrics['sens']     = metrics['sens'] + m['sens']
    metrics['f1_score'] = metrics['f1_score'] + m['f1_score']
    metrics['auc']      = metrics['auc'] + m['auc']
    
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(m['acc']), np.std(m['acc'])))
    print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(m['spec']), np.std(m['spec'])))
    print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(m['sens']), np.std(m['sens'])))
    print("F1-score: %.2f%% (+/- %.2f%%)" % (np.mean(m['f1_score']), np.std(m['f1_score'])))
    print("AUC: %.2f (+/- %.2f)" % (np.mean(m['auc']), np.std(m['auc'])))
    
end = time.time()

print()
print("Results ------------------------------------------")
print("Time to validate results:", (end - start)/60, "minutes")
print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(metrics['acc']), np.std(metrics['acc'])))
print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(metrics['spec']), np.std(metrics['spec'])))
print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(metrics['sens']), np.std(metrics['sens'])))
print("F1-score: %.2f%% (+/- %.2f%%)" % (np.mean(metrics['f1_score']), np.std(metrics['f1_score'])))
print("AUC: %.2f (+/- %.2f)" % (np.mean(metrics['auc']), np.std(metrics['auc'])))