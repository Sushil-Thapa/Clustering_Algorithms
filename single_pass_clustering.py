
# randomly initialize k cluster means. let each cluster have a discard set in the buffer that keeps track of the sufficient statistics for all points from prev iterations

#fill the buffer points

#perform iterations of k-means on the poinnts and discard sets in the buffer,until convergence
#for this clustering,each discard set is treated like a regular point places at the mean of the discard set but weighted with the number of points in the discard set

#for each cluster update the sufficient statistics of the discard set with the points assignmed to the cluster
#Remove all points from the buffer

#if the dataset is exhausted then finish.
#Other wise repeat from setp 2
#
#
#
#
#
#
#
#
#
import numpy as np
numberOfClusters = 3
discardSets = { 'sumOfAllPoints':None,
                'numberOfPoints':None,
                'sqSumOfAllPoints':None
}
# class DiscardSets{
#     def __init__():
#         'sumOfAllPoints':None
#         'numberOfPoints':None
#         'sqSumOfAllPoints':None
#
# }
dataset = np.loadtxt("data/data.csv", skiprows=1, delimiter='\t',usecols=range(1,3))
np.random.shuffle(dataset.flat)
# randomly initialize k cluster means. let each cluster have a discard set in the buffer that keeps track of the sufficient statistics for all points from prev iterations
numberOfData,numberOfFeatures = dataset.shape
print numberOfData,numberOfFeatures

randomClusterMeans = dataset[np.random.choice(numberOfData, numberOfClusters, replace=False), :]
print randomClusterMeans.shape
clusterList = ['cluster0','cluster1','cluster2']

for i in range(randomClusterMeans.shape[0]):
    clusterList[i] = {'mean':randomClusterMeans[i],
                        'discardSets':discardSets
                    }
    print clusterList[i]


#fill the buffer points
fraction = numberOfData//100
for i in range(100):
    bufferSet, dataset = dataset[:fraction],dataset[fraction:]
    print(len(bufferSet))
