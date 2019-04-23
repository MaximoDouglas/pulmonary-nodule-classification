'''Images and features processor'''
import math
import itertools
import re
import os
import imageio
import shutil
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from sklearn.model_selection import KFold
from tqdm import tqdm

np.random.seed(1937)

RES = 64
TEST_SIZE = 50

SLICES = 5
STRATEGY = 'first'
REPEAT = False

data_fold = "data-" + str(SLICES) + "-" + str(STRATEGY)
if (REPEAT):
    data_fold += "-repeat"

def normalize_balanced(nodules, n_slices, repeat=False):
    '''Normalizes the nodule slices number:
    - A nodule with less than n slices is completed with black slices
    - A nodule with more than n slices have its first and last one selected, plus
    the 1 + (n-1/5)*k, where k = {1, 2, 3, 4}
    '''
    normalized_slices = []

    for nodule in nodules:
        new_nodule = []
        # adds black slices

        if repeat:
            times = math.ceil(n_slices/len(nodule))
            nodule = list(itertools.chain.from_iterable(itertools.repeat(x, times) for x in nodule))

        if len(nodule) <= n_slices:
                for slice in nodule:
                    new_nodule.append(slice)
                for i in range(n_slices - len(nodule)):
                    new_nodule.append(np.zeros((RES, RES)))
        elif len(nodule) > n_slices:
            new_nodule.append(nodule[0])
            for k in range(1, n_slices-1):
                new_nodule.append(nodule[round(1 + ((len(nodule) - 1) / (n_slices-1)) * k)])
            new_nodule.append(nodule[-1])
        normalized_slices.append(new_nodule)
    return normalized_slices

def normalize_first(nodules, n_slices, repeat=False):
    '''Normalizes the nodule slices number:
    - A nodule with less than n slices is completed with black slices
    - A nodule with more than n slices have its n first slices selected
    '''
    normalized_slices = []

    for nodule in nodules:
        new_nodule = []

        if repeat:
            times = math.ceil(n_slices/len(nodule))
            nodule = list(itertools.chain.from_iterable(itertools.repeat(x, times) for x in nodule))

        if len(nodule) <= n_slices:
                for slice in nodule:
                    new_nodule.append(slice)
                for i in range(n_slices - len(nodule)):
                    new_nodule.append(np.zeros((RES, RES)))
        elif len(nodule) > n_slices:
            for i in range(0, n_slices):
                new_nodule.append(nodule[i])
        normalized_slices.append(new_nodule)
    return normalized_slices

def read_images(path, path_features):
    '''Reads the images files, a .csv with the features of each nodule and mounts an array
    Parameters:
        path (string): path to the nodules folders
        path_features (string): path to the features .csv
    Returns:
        list: list of nodules with slices as Numpy Arrays
        features: list of features corresponding to the nodules on list
    '''

    df = pd.read_csv(path_features)
    allFeatures = df.values

    list       = []
    features    = []

    for root, dirs, files in os.walk(path):
        for dirname in sorted(dirs, key=str.lower):
            for root1, dirs1, files1 in os.walk(path + "/" + dirname):
                for dirname1 in sorted(dirs1, key=str.lower):
                    for root2, dirs2, files2 in os.walk(path + "/" + dirname + "/" + dirname1):
                        slices = []
                        files2[:] = [re.findall('\d+', x)[0] for x in files2]

                        axis = 0 # To get the Rows indices
                        examColumn = 0 # Column of the csv where the exam code is
                        noduleColumn = 1 # Column of the csv where the nodule code is

                        # index of the rows that have the exam id equal to the exam id of the current nodule
                        indExam  = np.where(allFeatures[:,examColumn] == dirname)[axis]

                        # index of the rows that have the nodule id equal to the id of the current nodule
                        indNodule = np.where(allFeatures[:,noduleColumn] == dirname1)[axis]

                        # Intersect the two arrays, which results in the row for the features 
                        # of the current nodule
                        i = np.intersect1d(indExam,indNodule)

                        # A list is returned, but there's just one value, so I used its index
                        index = 0
                        exam = allFeatures[i, examColumn][index]
                        nodule = allFeatures[i, noduleColumn][index]

                        '''Verify if there's more than one index for each nodule
                        and if there's divergence between the nodule image file location and the
                        csv values'''

                        if((len(i) > 1) or (str(exam) != str(dirname)) or (str(nodule) != str(dirname1))):
                            print("Features error!")
                        else:
                            '''Transform the list of index with just one value in a
                            primitive value to use as index to save the features values'''
                            i = i[0]

                        for f in sorted(files2, key=float):
                            img = imageio.imread(root2 + "/" + f + ".png", as_gray=True)
                            slices.append(img)

                        list.append(slices)
                        features.append(allFeatures[i,2:74].tolist())

    return list, features

'''This is the fastest method, which just replicate the features the same amount as the images get rotated'''
def rotate_slices(nodules, f, times, mode='constant'):
    ''' Rotates a list of images n times'''
    rotated = nodules
    angle = 360/times
    rep_feat = f

    for i in range(1, times):
        temp = rotate(nodules, i*angle, (1, 2), reshape=False, mode = mode)
        rotated     = np.concatenate([rotated, temp])
        rep_feat    = np.concatenate([rep_feat, f])

    return rotated, rep_feat

'''This method is slower, but navigate for each nodule, which can be helpful in the future'''
def rotate_slices_slow(nodules, f, times, mode='constant'):
    ''' Rotates a list of images n times'''
    rotated = nodules
    rep_feat = f
    angle = 360/times

    for ind, nd in enumerate(nodules):
        temp_nodule = []
        temp_f = []

        temp_nodule.append(nd)
        temp_f.append(f[ind])

        for i in range(1, times):
            temp_new = rotate(nd, i*angle, (0, 1), reshape=False, mode = mode)
            temp_nodule.append(temp_new)
            temp_f.append(f[ind])

        rotated = np.append(rotated, temp_nodule, axis=0)
        rep_feat = np.append(rep_feat, temp_f, axis=0)

    return rotated, rep_feat

'''This method works just for specific versions of python, becausa uses a list as a index to get a sublist'''
def my_kfold(ben, mal, f_ben, f_mal, n_splits, ben_rot, mal_rot):
    kf = KFold(n_splits)

    mal_train, mal_test = [], []
    f_mal_train, f_mal_test = [], []
    for train_index, test_index in kf.split(mal):
        # Using mal[train_index] is deprecate - It may be changed to [mal[index] for index in train_index]
        mal_train.append(mal[train_index])
        f_mal_train.append(f_mal[train_index])

        mal_test.append(mal[test_index])
        f_mal_test.append(f_mal[test_index])

    ben_train, ben_test = [], []
    f_ben_train, f_ben_test = [], []
    
    # To make sure that the folds of test will have the same number of mal and ben nodules
    for (train_index, test_index), mal in zip(kf.split(ben), mal_test):
        
        sample = np.random.choice(test_index, len(mal), replace=False)
        sample_ = np.setdiff1d(test_index, sample)

        ben_train_ind = np.concatenate((train_index, sample_))

        ben_train.append(ben[ben_train_ind])
        f_ben_train.append(f_ben[ben_train_ind])

        ben_test.append(ben[sample])
        f_ben_test.append(f_ben[sample])

    X_test, Y_test = [], []
    for b, m in zip(ben_test, mal_test):
        X_test.append(np.concatenate((b, m), 0))

        y_test = len(b) * [0] + len(m) * [1]
        Y_test.append(np.array(y_test))

    f_test = []
    for f_b, f_m in zip(f_ben_test, f_mal_test):
        f_test.append(np.concatenate((f_b, f_m), 0))

    X_train, Y_train = [], []
    f_train = []
    for i in tqdm(range(n_splits)):
        b, m = ben_train[i], mal_train[i]
        f_b_train, f_m_train = f_ben_train[i], f_mal_train[i]

        b, f_b_train = rotate_slices(nodules=b, f=f_b_train, times=ben_rot)
        m, f_m_train = rotate_slices(nodules=m, f=f_m_train, times=mal_rot)

        X_train.append(np.concatenate((b, m), 0))
        f_train.append(np.concatenate((f_b_train, f_m_train), 0))

        y_train = len(b) * [0] + len(m) * [1]
        Y_train.append(np.array(y_train))

    return X_train, X_test, f_train, f_test, Y_train, Y_test

def get_folds(basedir, n_slices, strategy='first', repeat=False, features=None):
    ben_dir = basedir + "benigno/"
    mal_dir = basedir + "maligno/"

    ben, f_ben = read_images(ben_dir, features)
    mal, f_mal = read_images(mal_dir, features)

    if strategy == 'first':
        ben = normalize_first(ben, n_slices, repeat)
        mal = normalize_first(mal, n_slices, repeat)
    elif strategy == 'balanced':
        ben = normalize_balanced(ben, n_slices, repeat)
        mal = normalize_balanced(mal, n_slices, repeat)

    ben = np.concatenate(ben).reshape(len(ben), n_slices, RES, RES, 1)
    mal = np.concatenate(mal).reshape(len(mal), n_slices, RES, RES, 1)

    ben = np.moveaxis(ben, 1, 3)
    mal = np.moveaxis(mal, 1, 3)

    ben_zip = list(zip(ben, f_ben))
    np.random.shuffle(ben_zip)
    ben, f_ben = zip(*ben_zip)

    mal_zip = list(zip(mal, f_mal))
    np.random.shuffle(mal_zip)
    mal, f_mal = zip(*mal_zip)

    X_train, X_test, f_train, f_test, Y_train, Y_test = my_kfold(ben, mal, f_ben, f_mal, 10, 5, 13)

    return X_train, X_test, f_train, f_test, Y_train, Y_test

if __name__ == "__main__":
    ben_dir = "../../../data/images/solid-nodules-with-attributes/benigno"
    mal_dir = "../../../data/images/solid-nodules-with-attributes/maligno"
    features_path = "../../../data/features/solidNodules.csv"

    print("Lendo imagens do disco")

    ben, f_ben = read_images(ben_dir, features_path)
    mal, f_mal = read_images(mal_dir, features_path)

    if (STRATEGY == 'first'):
        ben = normalize_first(ben, SLICES, REPEAT)
        mal = normalize_first(mal, SLICES, REPEAT)
    else:
        ben = normalize_balanced(ben, SLICES, REPEAT)
        mal = normalize_balanced(mal, SLICES, REPEAT)

    print("Mudando a forma")

    print(">", len(ben))

    ben = np.concatenate(ben).reshape(len(ben), SLICES, RES, RES, 1)
    mal = np.concatenate(mal).reshape(len(mal), SLICES, RES, RES, 1)

    print("Trocando os eixos")

    print("Antes: ", ben.shape)

    ben = np.moveaxis(ben, 1, 3)
    mal = np.moveaxis(mal, 1, 3)

    print("Depois: ", ben.shape)

    print("Separando dados de teste")

    ben_test_indices = np.random.choice(len(ben), TEST_SIZE, replace=False)
    mal_test_indices = np.random.choice(len(mal), TEST_SIZE, replace=False)

    ben_test = [ben[i] for i in ben_test_indices]
    f_ben_test = [f_ben[i] for i in ben_test_indices]

    mal_test = [mal[i] for i in mal_test_indices]
    f_mal_test = [f_mal[i] for i in mal_test_indices]

    ben_test = np.array(ben_test)
    f_ben_test = np.array(f_ben_test)

    mal_test = np.array(mal_test)
    f_mal_test = np.array(f_mal_test)

    ben_train = np.delete(ben, ben_test_indices, axis = 0)
    f_ben_train = np.delete(f_ben, ben_test_indices, axis = 0)

    mal_train = np.delete(mal, mal_test_indices, axis = 0)
    f_mal_train = np.delete(f_mal, mal_test_indices, axis = 0)

    del(ben, f_ben, mal, f_mal, ben_dir, mal_dir, features_path, ben_test_indices, mal_test_indices)

    print("Aumento de base")

    ben_train, f_ben_train = rotate_slices(nodules=ben_train, f=f_ben_train, times=5)
    mal_train, f_mal_train = rotate_slices(nodules=mal_train, f=f_mal_train, times=13)

    print("Juntando benignos e malignos")

    X_train = np.concatenate([ben_train, mal_train])
    f_train = np.concatenate([f_ben_train, f_mal_train])

    X_test  = np.concatenate([ben_test, mal_test])
    f_test  = np.concatenate([f_ben_test, f_mal_test])

    print('Shapes: ')
    print('X_train: ' + str(X_train.shape))
    print('f_train: ' + str(f_train.shape))

    print()
    print('X_test: ' + str(X_test.shape))
    print('f_test: ' + str(f_test.shape))

    print("Gerando labels")

    train_labels = len(ben_train) * [0] + len(mal_train) * [1]
    test_labels = len(ben_test) * [0] + len(mal_test) * [1]

    print("Tipo categórico")

    Y_train = np.array(train_labels)
    Y_test = np.array(test_labels)

    data = data_fold

    shutil.rmtree(data, ignore_errors=True)
    os.mkdir(data)

    np.save(data + "/f_train.npy", f_train)
    np.save(data + "/f_test.npy", f_test)
    np.save(data + "/X_train.npy", X_train)
    np.save(data + "/X_test.npy", X_test)
    np.save(data + "/Y_train.npy", Y_train)
    np.save(data + "/Y_test.npy", Y_test)