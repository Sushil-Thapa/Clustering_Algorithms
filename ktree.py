import time,resource
import numpy as np
from ktree import ktree,KTreeOptions,utils

order = 6 # order: order of K-tree default:6

# ds = np.loadtxt("data/data.csv", skiprows=1, delimiter='\t',usecols=range(1,3))
ds = np.loadtxt("data/generated_data_1000.csv")
print ('Shape: ',ds.shape)
N,d = ds.shape # N: number of examples, d: dimension of examples

print "Building KTree of order ", order
t0 = time.time()

options = KTreeOptions()
options.order = order
options.weighted = True
options.distance = "euclidean"
options.reinsert = True
k = ktree(data=ds, options=options)

print
print('Time elapsed:',time.time() - t0)
print("Max_ram_usage: %.2f MB.\n" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024))
print
print "order:", k.order
print "N:", N
print "depth:", k.depth
print
test=np.array([[31.025,92.975],[114.191,35.114],[24.802,16.521]])

for i in range(3):
    # vec = np.ones((1,d))*i
    vec = test[i,:]
    print "Looking for nearest neighbor of", str(vec), ",",
    print "found: ", k.nearest_neighbor(vec)

# print('utils:')
# utils.save_ktree(k,'save_ktree.pickle') #Save Pickle
# k = utils.load_ktree('save_ktree.pickle') #Read Pickle
