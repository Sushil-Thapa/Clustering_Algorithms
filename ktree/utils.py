import numpy
import cPickle as pickle

def save_ktree(k, filename):
    f = open(filename, "wb")
    pickle.dump(k, f, -1)
    f.close()

def load_ktree(filename):
    f = open(filename, "rb")
    k = pickle.load(f)
    f.close()
    return k

