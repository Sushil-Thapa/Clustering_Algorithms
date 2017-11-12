
# randomly initialize k cluster means. let each cluster have a discard set in the buffer that keeps track of the sufficient statistics for all points from prev iterations

#fill the buffer points

#perform iterations of k-means on the poinnts and discard sets in the buffer,until convergence
#for this clustering,each discard set is treated like a regular point places at the mean of the discard set but weighted with the number of points in the discard set

#for each cluster update the sufficient statistics of the discard set with the points assignmed to the cluster
#Remove all points from the buffer

#if the dataset is exhausted then finish.
#Other wise repeat from setp 2
#


import numpy as np
from sklearn.cluster import KMeans
numberOfClusters = 3
discardSets = { 'sumOfAllPoints':None,
                'numberOfPoints':None,
                'sqSumOfAllPoints':None
}

dataset = np.loadtxt("data/data.csv", skiprows=1, delimiter='\t',usecols=range(1,3))
# print('shape of dataset:',dataset.shape)


np.random.shuffle(dataset.flat)
# randomly initialize k cluster means. let each cluster have a discard set in the buffer that keeps track of the sufficient statistics for all points from prev iterations
numberOfData,numberOfFeatures = dataset.shape
# print numberOfData,numberOfFeatures

randomClusterMeans = dataset[np.random.choice(numberOfData, numberOfClusters, replace=False), :]
# print randomClusterMeans.shape
clusterList = []
# for i in range(numberOfClusters):
#     clusterList.append('cluster'+str(i))

# clusterList = ['cluster0','cluster1','cluster2']

for i in range(randomClusterMeans.shape[0]):
    clusterList.append({'mean':randomClusterMeans[i],
                        'discardSets':discardSets
                    })
    # print clusterList[i]['mean']
print
# print clusterList
#fill the buffer points
perRate = 100
fraction = numberOfData//perRate
firstIteration = True

#if the dataset is exhausted then finish.
#Other wise repeat from setp 2
for i in range(perRate):

    bufferSet, dataset = dataset[:fraction,:],dataset[fraction:,:] #fill the buffer points

    # print('buffer:'+str(bufferSet.shape))
    # print('dataset:'+str(dataset.shape))
    if firstIteration:
        firstIteration = False
    else: #for each cluster update the sufficient statistics of the discard set with the points assignmed to the cluster

        for i in range(numberOfClusters):
            num = clusterList[i]['discardSets']['numberOfPoints']
            tmp = np.tile(clusterList[i]['mean'],(num,1))
            # print num,tmp.shape

            bufferSet = np.append(bufferSet,tmp, axis=0) #np.time(X,(3,1))
            # np.append(bufferSet,np.vstack([clusterList[i]['mean']]*clusterList[i]['discardSets']['numberOfPoints']), axis=0) #np.time(X,(3,1))
            # print 'after append:',str(bufferSet.shape)

#perform iterations of k-means on the poinnts and discard sets in the buffer,until convergence
#for this clustering,each discard set is treated like a regular point places at the mean of the discard set but weighted with the number of points in the discard set
    k_means = KMeans(n_clusters=numberOfClusters, init=randomClusterMeans,  n_init=1)
    k_means.fit(bufferSet)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    tempClusters = {i: bufferSet[np.where(k_means.labels_ == i)] for i in range(k_means.n_clusters)}

    # print k_means_cluster_centers,tempClusters
    for i in range(numberOfClusters):
        tempDiscardSets = { 'sumOfAllPoints':np.sum(tempClusters[i], axis=0),
                        'numberOfPoints':tempClusters[i].shape[0],
                        'sqSumOfAllPoints':np.sum(np.square(tempClusters[i]), axis=0)
        }
        clusterList[i]={'mean':k_means_cluster_centers[i],
                        'discardSets':tempDiscardSets}
    # a = raw_input()
    # print clusterList[i]['mean']
for i in range(numberOfClusters):
    print clusterList[i]
