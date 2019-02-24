# Multiclass Classification with the Abalone Dataset
import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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
epochs = 50

# define baseline model
def baseline_model():
  # create model
  model = Sequential()
  model.add(Dense(1, input_dim=X.shape[1], activation="sigmoid", kernel_initializer="normal"))
  model.add(Dense(1, activation="sigmoid", kernel_initializer="normal"))
  model.add(Dense(1, activation="sigmoid", kernel_initializer="normal"))

  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
  return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs, batch_size=X.shape[1])
# ------- End model archtecture

# ------- Model validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold, verbose=10)

# Summarize
print("Accuracy: %.2f%%"%(results.mean()*100))
print("Standard deviation: %.2f%%"%(results.std()*100))
# ------- End model validation
