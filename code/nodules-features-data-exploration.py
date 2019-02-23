# Data exploration - 3mm to 10mm nodules
import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# fix random seed for reproducibility
seed = 1
numpy.random.seed(seed)
df = pandas.DataFrame()

# Load dataset - uncomment just the one that you want to work with
def loadData():
    global df
    df = pandas.read_csv("../data/nodulesFeatures-3mm-30mm.csv")
    #df = pandas.read_csv("../data/nodulesFeatures-5-10mm.csv")

# Shows shape (rows, columns)
def getShape():
    print("Shape (rows, columns): ", end="")
    print(df.shape)

# Check for nan fields
def getNaN():
    count = 0
    for v in df.drop('id', axis=1).isna().sum():
        if (v != 0): count+=1
    print("Missing values: ", end="")
    print(count)

def getNumericFeatures():
    numericFields = []
    for column in df.drop('id', axis=1).columns:
        if (numpy.issubdtype(df[column].dtype, numpy.number)):
            numericFields.append(column)

    return numericFields

def getStats():
    numericFields = getNumericFeatures()
    print("There are "+ str(len(numericFields)) +
    " numéric features. The stats for each one are presented bellow: ")

    for column in numericFields:
        print(" ." + column + ": ")
        print('     .max: ' + str(df[column].max()))
        print('     .min: ' + str(df[column].min()))
        print('     .median: ' + str(df[column].median()))
        print('     .mean: ' + str(df[column].mean()))
        print('     .Standard deviation: ' + str(df[column].std()))
        print()

def getCorrelations():
    # methods: 'pearson', 'kendall', 'spearman'
    encoder = LabelEncoder()
    df_aux = df.drop('id', axis=1)

    encoder.fit(df_aux[df_aux.columns[-1]])
    df_aux[df_aux.columns[-1]] = encoder.transform(df_aux[df_aux.columns[-1]])

    corr = df_aux.corr(method='pearson')

    print("Correlations between features and the class are listed bellow: ")
    for col in df_aux.columns[:-1]:
        c = corr[col][df_aux.columns[-1]]
        print(" ." + col +": "+ str(c))

def main():
    loadData()
    getShape()
    getStats()
    getNaN()
    getCorrelations()

main()
