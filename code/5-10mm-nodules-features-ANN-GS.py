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
df = pandas.read_csv("../data/nodulesFeatures-5-10mm.csv")
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
# define baseline model
def baseline_model(optimizer="rmsprop", init="glorot_uniform"):
  # create model
  model = Sequential()
  model.add(Dense(1, input_dim=X.shape[1], activation="relu", kernel_initializer=init))
  model.add(Dense(1, activation="relu", kernel_initializer=init))
  model.add(Dense(1, activation="sigmoid", kernel_initializer=init))

  # Compile model
  model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
  return model

estimator = KerasClassifier(build_fn=baseline_model, batch_size=X.shape[0])

# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
eps = numpy.array([50, 100, 150])
init = ['glorot_uniform', 'normal', 'uniform']

param_grid = dict(optimizer=optimizers, epochs=eps, init=init)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=kfold, verbose=10)

grid_result = grid.fit(X, y, verbose=0)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Result dict of all metrics Cross-Validation
result = grid_result.cv_results_

# List with all the params combinations
params = result['params']
# For each param combination, there are a mean_test_score and a std_test_score, as
#   the result of the Cross-validation made on the model configured with this combination
mean_test_scores = result['mean_test_score']
std_test_scores = result['std_test_score']

# For each params combination, print the respective results
for i in range(len(params)):
    print("%f (%f) with: %r" % (mean_test_scores[i], std_test_scores[i], params[i]))
