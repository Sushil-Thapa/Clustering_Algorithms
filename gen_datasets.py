 #adding sample data
from sklearn.datasets.samples_generator import make_blobs
centers = [[59, 60], [90, 98], [33,95],[78,10],[36,13],[115,40]]
X, y =make_blobs(n_samples=100000000, n_features=2, centers=centers, cluster_std=7, center_box=(1, 100.0), shuffle=True, random_state=99)

#print our enarated sample data
print X.shape, type(X)
print y.shape

import numpy as np
np.set_printoptions(suppress=True)
print('press enter to write X into file. <Press Enter>')
np.savetxt("data/generated_data_100m.csv", X, delimiter=",",fmt='%1.3f %1.3f')
print('Successfully Saved!!')
