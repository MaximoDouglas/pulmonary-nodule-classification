#%%
from errno import EEXIST
from os import makedirs, path

#%%
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
#%%
def binarize(y, threshold=0.5):
    ''' Takes as input a numeric array and a threshold, returns a array with 1 for each value >= threshold,
    0 for value < threshold'''
    y_bin = []
    for value in y:
        y_bin.append(1 if value >= threshold else 0)
    return y_bin

