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
    print("Shape (rows, columns): ", end="")
    print(df.shape)

# Check for nan fields
def getNaN():
    count = 0
    for v in df.isna().sum():
        if (v != 0): count+=1
    print("Missing values: ", end="")
    print(count)

def getNumericFields():
    numericFields = []
    for column in df.columns:
        if (numpy.issubdtype(df[column].dtype, numpy.number)):
            numericFields.append(column)

    return numericFields

def getStats():
    numericFields = getNumericFields()
    print("There are "+ str(len(numericFields)) +
    " numéric fields. The stats for each one are presented bellow: ")

    for column in numericFields:
        print(" ." + column + ": ")
        print('     .median: ' + str(df[column].median()))
        print('     .mean: ' + str(df[column].mean()))
        print('     .Standard deviation: ' + str(df[column].std()))
        print()

def main():
    getShape()
    getNaN()
    getStats()

main()
