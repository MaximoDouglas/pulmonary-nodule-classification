# Data exploration - 3mm to 10mm nodules
import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# fix random seed for reproducibility
seed = 1
numpy.random.seed(seed)

# ------- Data exploration
# load dataset
df = pandas.read_csv("../data/nodulesFeatures-3mm-30mm.csv")
print(df.shape)
