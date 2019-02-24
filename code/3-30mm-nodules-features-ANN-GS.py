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
from sklearn.model_selection import GridSearchCV

# fix random seed for reproducibility
seed = 1
numpy.random.seed(seed)

# ------- Begin data preprocessing
# load dataset
df = pandas.read_csv("../data/nodulesFeatures-3mm-30mm.csv")
dataset = df.drop('id', axis=1).values

# separate features from class
scaled = dataset[:,:-1].astype(float)
X = scaled[:,:]
y = df.values[:,-1]

# transform the class values
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y) # 0 - BENIGN | 1 - MALIGNANT
# ------- End data preprocessing


# ------- Model archtecture
epochs = 100

# define baseline model
def baseline_model(units_x=1, units_y=1):
  # create model
  model = Sequential()
  model.add(Dense(1, input_dim=X.shape[1], activation="relu", kernel_initializer="glorot_uniform"))
  model.add(Dense(1, activation="relu", kernel_initializer="glorot_uniform"))
  model.add(Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"))

  # Compile model
  learning_rate = 0.1
  decay_rate = learning_rate / epochs
  momentum = 0.8

  # Compile model
  sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)
  model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])
  return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs, batch_size=250)
# ------- End model archtecture

# ------- Model validation
kfold = KFold(n_splits=X.shape[0], shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold, verbose=10)

# Summarize
print("Accuracy: %.2f%%"%(results.mean()*100))
print("Standard deviation: %.2f%%"%(results.std()*100))
# ------- End model validation
