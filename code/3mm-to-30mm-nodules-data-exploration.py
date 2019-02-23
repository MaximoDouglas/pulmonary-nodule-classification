# Data exploration - 3mm to 10mm nodules
import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# fix random seed for reproducibility
seed = 1
numpy.random.seed(seed)

# load dataset
df = pandas.read_csv("../data/nodulesFeatures-3mm-30mm.csv")

# Shows shape (rows, columns)
def getShape():
    print(df.shape)

# Check for nan fields
def getNaN():
    count = 0
    for v in df.isna().sum():
        if v != 0: count+=1
    print(count)

def main():
    getShape()
    getNaN()

main()
