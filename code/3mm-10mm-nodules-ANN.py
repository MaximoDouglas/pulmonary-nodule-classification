# Multiclass Classification with the Abalone Dataset
import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import requests
import json

# fix random seed for reproducibility
seed = 1
numpy.random.seed(seed)

# ------- Begin data preprocessing

# load dataset
df = pandas.read_csv("../data/nodulesFeatures-3mm-30mm.csv") #3132 entries
print(df)

'''
# encode sex values as integers
encoder = LabelEncoder()
encoder.fit(dataset[:,0])
dataset[:,0] = encoder.transform(dataset[:,0])

# separate features from class
scaled = preprocessing.scale(dataset[:,0:8].astype(float))
X = scaled[:3132,0:8]
y = df.values[:,8]

# transform the class values
encoder.fit(y)
y = encoder.transform(y)

# ------- End data preprocessing

# ------- Model archtecture
epochs = 100

# define baseline model
def baseline_model():
  # create model
  model = Sequential()
  model.add(Dense(64, input_dim=8, activation="relu", kernel_initializer="glorot_uniform"))
  model.add(Dense(32, activation="relu", kernel_initializer="glorot_uniform"))
  model.add(Dense(3, kernel_initializer="glorot_uniform", activation="sigmoid"))

  # Compile model
  learning_rate = 0.1
  decay_rate = learning_rate / epochs
  momentum = 0.8

  # Compile model
  sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)
  model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
  return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs, batch_size=250)

# ------- End model archtecture

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold, verbose=10)

# Summarize
print("Accuracy: %.2f%%"%(results.mean()*100))
print("Standard deviation: %.2f%%"%(results.std()*100))
'''
