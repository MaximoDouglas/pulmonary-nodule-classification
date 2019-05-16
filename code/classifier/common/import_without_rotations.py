# Images and features processor - CHECK SETTINGS BEFORE USE
import math
import itertools
import re
import os
import imageio
import shutil
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm
import random

# Begin Settings -------------------------|

# Important folders
ben_dir = "../../../data/images/solid-nodules-with-attributes/benigno"
mal_dir = "../../../data/images/solid-nodules-with-attributes/maligno"
features_path = "../../../data/features/solidNodules.csv"

# Random seed to get a better reproductibility
np.random.seed(1937)

# If set True, it shows data informations in the output of the program
LOG = True

# Images resolution
RES = 64

# Size of the test fold
TEST_SIZE = 50

# Number of slices for each nodule
SLICES = 5

# Strategy used for normalization - it can be 'first' or 'balanced'
STRATEGY = 'first'

'''If set True, it will repeat slices when the number of slices of a nodule is less than SLICES. 
    If set False, the normalization will be filling with black images in the end'''
REPEAT = False

# It makes the name for the folder where the numpies will be stored
data_folder = "data-" + str(SLICES) + "-" + str(STRATEGY)
if (REPEAT):
    data_folder += "-repeat"

# End settings ---------------------------|

'''Function to plot a sequence of images. 
    nodules: a numpy of nodule images
    ind_nodules: list of indices to referenciate nodules in the 'nodules' numpy
    ind_slices: list of indices to referenciate the slices of each nodule to be plotted'''
def plot_nodule(nodules, ind_nodules, ind_slices):
    
    rows = len(ind_nodules)
    columns = len(ind_slices)

    _, axarr = plt.subplots(rows,columns)
    ind = 0

    for r, i in enumerate(ind_nodules):
        for c, j in enumerate(ind_slices):
            nod = nodules[i, :, :, j, 0]
            if (rows != 1 and columns != 1):
                axarr[r,c].imshow(nod, cmap='gray')
                axarr[r,c].set_title('Nodule - ' + str(i) + ' - Slice - ' + str(j))
            else:
                axarr[ind].imshow(nod, cmap='gray')
                axarr[ind].set_title('Nodule - ' + str(i) + ' - Slice - ' + str(j))
                ind += 1
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def verify_features(features, step):
    index = np.random.randint(0, len(features) - step - 1)
    
    if (features[index].all() == features[index + step].all()):
        return "     Features repeat - OK"
    else:
        return "     Features repeat - FAIL"

def normalize_balanced(nodules, n_slices, repeat=False):
    '''Normalizes the nodule slices number:
    - A nodule with less than n slices is completed with black slices
    - A nodule with more than n slices have its first and last one selected, plus
    the 1 + (n-1/5)*k, where k = {1, 2, 3, 4}'''

    normalized_slices = []

    for nodule in nodules:
        new_nodule = []
        
        # If repeat is set True, repeat slices
        if repeat:
            times = math.ceil(n_slices/len(nodule))
            nodule = list(itertools.chain.from_iterable(itertools.repeat(x, times) for x in nodule))

        if len(nodule) <= n_slices:
                for slice in nodule:
                    new_nodule.append(slice)
                for _ in range(n_slices - len(nodule)):
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
    - A nodule with more than n slices have its n first slices selected'''

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
    '''Reads the image files and a .csv with the features of each nodule
    Parameters:
        path (string): path to the nodules folders
        path_features (string): path to the features .csv
    Returns:
        list: list of nodules with slices as Numpy Arrays
        features: list of features corresponding to the nodules on list'''

    df = pd.read_csv(path_features)
    allFeatures = df.values

    list        = []
    features    = []

    for _,dirs,_ in os.walk(path):
        for dirname in sorted(dirs, key=str.lower):
            for _,subdirs,_ in os.walk(path + "/" + dirname):
                for subdirname in sorted(subdirs, key=str.lower):
                    for root2,_,files2 in os.walk(path + "/" + dirname + "/" + subdirname):
                        slices      = []
                        files2[:]   = [re.findall('\\d+', x)[0] for x in files2]

                        axis            = 0 # To get the Rows indices
                        examColumn      = 0 # Column of the csv where the exam code is
                        noduleColumn    = 1 # Column of the csv where the nodule code is

                        # index of the rows that have the exam id equal to the exam id of the current nodule
                        indExam = np.where(allFeatures[:,examColumn] == dirname)[axis]

                        # index of the rows that have the nodule id equal to the id of the current nodule
                        indNodule = np.where(allFeatures[:,noduleColumn] == subdirname)[axis]

                        # Intersect the two arrays, which results in the row for the features 
                        # of the current nodule
                        i = np.intersect1d(indExam,indNodule)

                        # A list is returned, but there's just one value, so I used its index
                        index   = 0
                        exam    = allFeatures[i, examColumn][index]
                        nodule  = allFeatures[i, noduleColumn][index]

                        '''Verify if there's more than one index for each nodule
                        and if there's divergence between the nodule image file location and the
                        csv values'''

                        if((len(i) > 1) or (str(exam) != str(dirname)) or (str(nodule) != str(subdirname))):
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
def rotate_slices(nodules, features, times, mode='constant'):
    ''' Rotates a list of images n times.
        'rotated' is a list that will contain all nodule images (originals and the results of the rotations)
        'aug_feat' is a list that will contain all nodule features (repeated as the images get augmented)'''

    rotated = nodules
    aug_feat = features
    angle = 360/times
    
    '''Make rotations (n - 1) times, where n is equal to 'times' parameter. 
        If it was n times, it would repeat the same image one more time (360 degree)'''
    for i in range(1, times):
        temp        = rotate(input=nodules, angle=i*angle, axes=(1, 2), reshape=False, mode=mode)
        rotated     = np.concatenate([rotated, temp])
        aug_feat    = np.concatenate([aug_feat, features])

    return rotated, aug_feat

def my_kfold(ben, mal, f_ben, f_mal, n_splits, ben_rot, mal_rot):
    kf = KFold(n_splits)
    
    f_mal_train, f_mal_test = [], []
    mal_train, mal_test = [], []
    for train_index, test_index in kf.split(mal):
        mal_train.append([mal[index] for index in train_index])
        f_mal_train.append([f_mal[index] for index in train_index])

        mal_test.append([mal[index] for index in test_index])
        f_mal_test.append([f_mal[index] for index in test_index])

    ben_train, ben_test = [], []
    f_ben_train, f_ben_test = [], []
    
    # It uses mal_test to make sure that the ben_test will have the same size, to make a balanced test fold
    for (train_index, test_index), mal in zip(kf.split(ben), mal_test):
        
        sample = np.random.choice(test_index, len(mal), replace=False)
        sample_ = np.setdiff1d(test_index, sample)

        ben_train_ind = np.concatenate((train_index, sample_))

        ben_train.append([ben[index] for index in ben_train_ind])
        f_ben_train.append([f_ben[index] for index in ben_train_ind])
        
        ben_test.append([ben[index] for index in sample])
        f_ben_test.append([f_ben[index] for index in sample])

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

        b, f_b_train = rotate_slices(nodules=b, features=f_b_train, times=ben_rot)
        m, f_m_train = rotate_slices(nodules=m, features=f_m_train, times=mal_rot)

        X_train.append(np.concatenate((b, m), 0))
        f_train.append(np.concatenate((f_b_train, f_m_train), 0))

        y_train = len(b) * [0] + len(m) * [1]
        Y_train.append(np.array(y_train))

    return X_train, X_test, f_train, f_test, Y_train, Y_test

def get_folds(basedir, n_slices, strategy='first', repeat=False, features=None):
    ben_dir = basedir + "benigno"
    mal_dir = basedir + "maligno"

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

    print("Begin > ")

    ben, f_ben = read_images(path=ben_dir, path_features=features_path)
    mal, f_mal = read_images(path=mal_dir, path_features=features_path)

    if (STRATEGY == 'first'):
        ben = normalize_first(nodules=ben, n_slices=SLICES, repeat=REPEAT)
        mal = normalize_first(nodules=mal, n_slices=SLICES, repeat=REPEAT)
    else:
        ben = normalize_balanced(nodules=ben, n_slices=SLICES, repeat=REPEAT)
        mal = normalize_balanced(nodules=mal, n_slices=SLICES, repeat=REPEAT)

    if LOG:
        print("Changind shape > ")
        print("     Ben as a list: ", len(ben))
        print("     Mal as a list: ", len(mal))

    ben = np.concatenate(ben).reshape(len(ben), SLICES, RES, RES, 1)
    mal = np.concatenate(mal).reshape(len(mal), SLICES, RES, RES, 1)
    
    if LOG:
        print("     Ben as a numpy: ", ben.shape)
        print("     Mal as a numpy: ", mal.shape)
        print()
        print("Swaping axes > ")

    ben = np.moveaxis(ben, 1, 3)
    mal = np.moveaxis(mal, 1, 3)

    if LOG:
        print("     Ben new shape: ", ben.shape)
        print("     Mal new shape: ", mal.shape)
        print()
        print("Separating Train and Test > ")

    ben_test_indices = np.random.choice(len(ben), TEST_SIZE, replace=False)
    mal_test_indices = np.random.choice(len(mal), TEST_SIZE, replace=False)

    ben_test = np.array([ben[i] for i in ben_test_indices])
    f_ben_test = np.array([f_ben[i] for i in ben_test_indices])

    mal_test = np.array([mal[i] for i in mal_test_indices])
    f_mal_test = np.array([f_mal[i] for i in mal_test_indices])

    ben_train = np.delete(ben, ben_test_indices, axis=0)
    f_ben_train = np.delete(f_ben, ben_test_indices, axis=0)

    mal_train = np.delete(mal, mal_test_indices, axis=0)
    f_mal_train = np.delete(f_mal, mal_test_indices, axis=0)

    # Clean memory
    del(ben, f_ben, mal, f_mal, ben_dir, mal_dir, features_path, ben_test_indices, mal_test_indices)

    if LOG:
        print("     Ben train: ", ben_train.shape)
        print("     Ben test: ", ben_test.shape)
        print()
        print("     Ben features train: ", f_ben_train.shape)
        print("     Ben features test: ", f_ben_test.shape)
        print()
        print("     Mal train: ", mal_train.shape)
        print("     Mal test: ", mal_test.shape)
        print()
        print("     Mal features train: ", f_mal_train.shape)
        print("     Mal features test: ", f_mal_test.shape)
        print()
        print("Data augmentation > ")

    ben_train, f_ben_train = rotate_slices(nodules=ben_train, features=f_ben_train, times=5)
    mal_train, f_mal_train = rotate_slices(nodules=mal_train, features=f_mal_train, times=13)
    
    if LOG:
        print(verify_features(features=f_mal_train, step=217))

        plot_nodule(nodules=mal_train, ind_nodules=[0, 217, 434, 651], ind_slices=[0, 1, 2, 3])
        
        print("     Ben train: ", ben_train.shape)
        print("     Ben features train: ", f_ben_train.shape)
        print()
        print("     Mal train: ", mal_train.shape)
        print("     Mal features train: ", f_mal_train.shape)
        print()
        print("Concatenating Ben and Mal > ")

    X_train = np.concatenate([ben_train, mal_train])
    f_train = np.concatenate([f_ben_train, f_mal_train])

    X_test  = np.concatenate([ben_test, mal_test])
    f_test  = np.concatenate([f_ben_test, f_mal_test])

    if LOG:
        print("     X_train: ", X_train.shape)
        print("     f_train: ", f_train.shape)
        print()
        print("     X_test: ", X_test.shape)
        print("     f_test: ", f_test.shape)

        print()
        print("Generating labels and saving numpies on disk > ")
    
    train_labels = len(ben_train) * [0] + len(mal_train) * [1]
    Y_train = np.array(train_labels)

    test_labels = len(ben_test) * [0] + len(mal_test) * [1]
    Y_test = np.array(test_labels)
    
    shutil.rmtree(data_folder, ignore_errors=True)
    os.mkdir(data_folder)

    np.save(data_folder + "/f_train.npy", f_train)
    np.save(data_folder + "/f_test.npy", f_test)
    np.save(data_folder + "/X_train.npy", X_train)
    np.save(data_folder + "/X_test.npy", X_test)
    np.save(data_folder + "/Y_train.npy", Y_train)
    np.save(data_folder + "/Y_test.npy", Y_test)
    
    print("Finished!")