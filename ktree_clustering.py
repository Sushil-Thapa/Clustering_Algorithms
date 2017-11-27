import time,resource
import numpy as np
from ktree import ktree,KTreeOptions,utils

def ktree_clustering(order,selection,dataset):
    ds = dataset
    # order = 10 # order: order of K-tree default:6 i.e. the maximum number of a node's children.  Default order is 5.
    # ds = np.loadtxt("data/data.csv", skiprows=1, delimiter='\t',usecols=range(1,3))
    # ds = np.loadtxt("data/generated_data_1000.csv")
    print ('Shape: ',ds.shape)
    N,d = ds.shape # N: number of examples, d: dimension of examples


    options = KTreeOptions() #sets options for ktree_clustering
    options.order = order
    options.weighted = True
    options.distance = "euclidean"
    options.reinsert = True

    t0 = time.time()
    k = ktree(data=ds, options=options)   # returns k-Node object
    print "ktree Number of clusters: ",len(k.root) #k.root has all clusters with list of clusters in one key of dict.
    # print
    timeElapsed = time.time() - t0
    # print("Max_ram_usage: %.2f MB.\n" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024))

    # utils.save_ktree(k,'save_ktree.pickle') #Save Pickle
    # k = utils.load_ktree('save_ktree.pickle') #Read Pickle
    results = {}
    results['n_clusters']=len(k.root)
    # results['timeElapsed']=timeElapsed

    return results


# ktree_clustering()
