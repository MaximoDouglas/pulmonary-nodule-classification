'''Módulo responsável pela importação e processamento das imagens dos nódulos.'''
import math
import itertools
import re
import os

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from tqdm import tqdm
import shutil

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
                new_nodule.append(nodule[int(1 + ((len(nodule) - 1) / (n_slices-1)) * k)])
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

def plot_nodule(nod, rows, cols):
    '''Shows the slices of a nodule in a single plot'''
    plt.figure(1)
    for i in range(nod.shape[2]):
        position = rows * 100 + cols * 10 + i + 1
        plt.subplot(position)
        plt.imshow(nod[:, :, i, 0], cmap='gray')
    plt.show()

def read_images(path, category):
    '''Reads the images files in our file structure and mounts an array
    Parameters:
        path (string): path to the nodules folders
        category (string): benigno or maligno
    Returns:
        list: list of nodules with slices as Numpy Arrays
    '''
    lista = []

    for root, dirs, files in os.walk(path):
        for dirname in sorted(dirs, key=int):
            slices = []
            for root1, dirs1, files1 in os.walk(root + "/" + dirname):
                files1[:] = [re.findall('\d+', x)[0] for x in files1]
                for f in sorted(files1, key=float):
                    img = misc.imread(root1 + "/" + category + f + "-" + str(len(files1) - 1)+ ".png", flatten=True)
                    slices.append(img)
                lista.append(slices)
    return lista

def rotate_slices(slices, times, mode='constant'):
    ''' Rotates a list of images n times'''
    rotated = slices
    angle = 360/times
    for i in range(1, times):
        temp = rotate(slices, i*angle, (1, 2), reshape=False, mode = mode)
        rotated = np.concatenate([rotated, temp])
    return rotated

def remove_if_exists(file):
    '''Removes a file if it exists'''
    if os.path.exists(file):
        os.remove(file)

def my_kfold(ben, mal, n_splits, ben_rot, mal_rot):
    kf = KFold(n_splits)

    mal_train, mal_test = [], []
    for train_index, test_index in kf.split(mal):
        mal_train.append(mal[train_index])
        mal_test.append(mal[test_index])

    ben_train, ben_test = [], []
    # percorro o mal_test para que os folds de test tenham o mesmo número de itens
    for (train_index, test_index), mal in zip(kf.split(ben), mal_test):
        sample = np.random.choice(test_index, len(mal), replace=False)
        sample_ = np.setdiff1d(test_index, sample)

        ben_train.append(ben[np.concatenate((train_index, sample_))])
        ben_test.append(ben[sample])

    X_test, Y_test = [], []
    for b, m in zip(ben_test, mal_test):
        X_test.append(np.concatenate((b, m), 0))

        y_test = len(b) * [0] + len(m) * [1]
        Y_test.append(np.array(y_test))
        #Y_test.append(to_categorical(y_test))

    X_train, Y_train = [], []
    for i in tqdm(range(n_splits)):
        b, m = ben_train[i], mal_train[i]

        b = rotate_slices(b, ben_rot)
        m = rotate_slices(m, mal_rot)

        X_train.append(np.concatenate((b, m), 0))

        y_train = len(b) * [0] + len(m) * [1]
        Y_train.append(np.array(y_train))
        #Y_train.append(to_categorical(y_train))

    return X_train, X_test, Y_train, Y_test

def get_folds(basedir, n_slices):
    ben_dir = basedir + "benigno/"
    mal_dir = basedir + "maligno/"

    ben = read_images(ben_dir, "benigno")
    mal = read_images(mal_dir, "maligno")

    ben = normalize_first(ben, n_slices, True)
    mal = normalize_first(mal, n_slices, True)

    ben = np.concatenate(ben).reshape(len(ben), n_slices, RES, RES, 1)
    mal = np.concatenate(mal).reshape(len(mal), n_slices, RES, RES, 1)

    ben = np.moveaxis(ben, 1, 3)
    mal = np.moveaxis(mal, 1, 3)

    np.random.shuffle(ben)
    np.random.shuffle(mal)

    X_train, X_test, Y_train, Y_test = my_kfold(ben, mal, 10, 5, 12)

    return X_train, X_test, Y_train, Y_test

np.random.seed(1742)
RES = 64
SLICES = 7
TEST_SIZE = 50

if __name__ == "__main__":
    ben_dir = "/home/douglas/dev/tcc/images/phatch/benigno"
    mal_dir = "/home/douglas/dev/tcc/images/phatch/maligno"

    print("Lendo imagens do disco")

    ben = read_images(ben_dir, "benigno")
    mal = read_images(mal_dir, "maligno")

    ben = normalize_balanced(ben, SLICES, True)
    mal = normalize_balanced(mal, SLICES, True)

    print("Mudando a forma")

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
    mal_test = [mal[i] for i in mal_test_indices]

    ben_test = np.array(ben_test)
    mal_test = np.array(mal_test)

    ben_train = np.delete(ben, ben_test_indices, axis = 0)
    mal_train = np.delete(mal, mal_test_indices, axis = 0)

    #ben_train_indices = np.random.choice(len(ben_train), len(mal_train), replace=False)
    #ben_train = ben_train[ben_train_indices]

    del(ben, mal, ben_dir, mal_dir, ben_test_indices, mal_test_indices)

    print("Aumento de base")

    ben_train = rotate_slices(ben_train, 4)#, 'reflect')
    mal_train = rotate_slices(mal_train, 10)#, 'reflect')

    print("Juntando benignos e malignos")

    X_train = np.concatenate([ben_train, mal_train])
    X_test = np.concatenate([ben_test, mal_test])

    print("Gerando labels")

    train_labels = len(ben_train) * [0] + len(mal_train) * [1]
    test_labels = len(ben_test) * [0] + len(mal_test) * [1]

    print("Tipo categórico")

    Y_train = np.array(train_labels)
    Y_test = np.array(test_labels)

    #Y_train = to_categorical(train_labels, 2)
    #Y_test = to_categorical(test_labels, 2)

    data = "data-balanceadas-repete-7"

    shutil.rmtree(data, ignore_errors=True)
    os.mkdir(data)

    np.save(data + "/X_train.npy", X_train)
    np.save(data + "/X_test.npy", X_test)
    np.save(data + "/Y_train.npy", Y_train)
    np.save(data + "/Y_test.npy", Y_test)
