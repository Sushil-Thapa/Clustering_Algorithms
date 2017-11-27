
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
import resource,time,os
import matplotlib.pyplot as plt

def single_pass_clustering(numberOfClusters,dataset):
    print dataset.shape

    # numberOfClusters = 5
    discardSets = { 'sumOfAllPoints':None, #initial descardset skeleton
                    'numberOfPoints':None,
                    'sqSumOfAllPoints':None
    }

    backup_dataset = dataset


    numberOfData,numberOfFeatures = dataset.shape

    # randomClusterMeans = dataset[np.random.choice(numberOfData, numberOfClusters, replace=False), :]
    clusterList = []
    # for i in range(numberOfClusters):
    #     clusterList.append('cluster'+str(i))

    # clusterList = ['cluster0','cluster1','cluster2']

    for i in range(numberOfClusters): #skeleton design for clusters
        clusterList.append({'mean':None,
                            'discardSets':discardSets
                        })
        # print clusterList[i]['mean']
    # print clusterList
    #fill the buffer points
    perRate = 100
    fraction = numberOfData//perRate #1000000/100 = 10000 one percent of data
    firstIteration = True

    #if the dataset is exhausted then finish.
    #Other wise repeat from setp 2
    startTime = time.time()
    for ith in range(perRate): # 100 iterations
        bufferSet_bak, dataset = dataset[:fraction,:],dataset[fraction:,:] #fill the buffer points

        if firstIteration: #do nothing at first
            firstIteration = False
            bufferSet = bufferSet_bak

        else: #for each cluster update the sufficient statistics of the discard set with the points assignmed to the cluster

            for i in range(numberOfClusters):
                num = clusterList[i]['discardSets']['numberOfPoints'] #number of points in each previous kmeans cluster
                tmp = np.tile(clusterList[i]['mean'],(num,1)) #make copies of data with same values of prev centroids

                bufferSet = np.append(bufferSet_bak,tmp, axis=0) #np.time(X,(3,1)) #append them to new buffer sets
                # print 'after append:',str(bufferSet.shape)

    #perform iterations of k-means on the poinnts and discard sets in the buffer,until convergence
    #for this clustering,each discard set is treated like a regular point places at the mean of the discard set but weighted with the number of points in the discard set

        k_means = KMeans(n_clusters=numberOfClusters)

        k_means.fit(bufferSet)
        k_means_labels = k_means.labels_
        k_means_cluster_centers = k_means.cluster_centers_
        temp_init = k_means_cluster_centers
        k_means_labels_unique = np.unique(k_means_labels)

        # #TODO  NEED TO GET POINTS FOR PLOTTING
        # #TODO put unique data points in tempClusters, puts Mean values from tmp too right now
        tempClusters={}
        # print 'bufferset',bufferSet.shape
        for i in range(numberOfClusters):
            points = bufferSet[np.where(k_means_labels == i)] #points in a cluster
            if (ith > 0):
                # print 'points before:',points.shape
                for j in range(points.shape[0]):
                    # print 'mean:',clusterList[i]['mean'],'pointj',points[j],'--',points[j].shape,clusterList[i]['mean'].shape
                    if np.array_equal(points[j],clusterList[i]['mean']):
                        # print 'deleting',points[j]
                        np.delete(points, np.where(np.array_equal(points[j],clusterList[i]['mean'])), axis=0)
                        # print 'points after:',points.shape
            tempClusters[i] =points
            # print tempClusters
            # #TODO

        # tempClusters = {i: bufferSet[np.where(k_means_labels == i)] for i in range(k_means.n_clusters)} #get all points of

        # print k_means_cluster_centers,tempClusters
        for i in range(numberOfClusters):
            tempDiscardSets = { 'sumOfAllPoints':np.sum(tempClusters[i], axis=0),
                            'numberOfPoints':tempClusters[i].shape[0],
                            'sqSumOfAllPoints':np.sum(np.square(tempClusters[i]), axis=0)
            }
            clusterList[i]={'mean':k_means_cluster_centers[i],
                            'discardSets':tempDiscardSets}
        # print 'iteration +1'
        # a = raw_input()
        # print clusterList[i]['mean']



    timeElapsed = time.time() - startTime
    # print("Max_ram_usage: %.2f MB.\n" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024)) #TODO make function f and use mem_usage = memory_usage(f) an its max
    return timeElapsed

ds = np.loadtxt("data/generated_data.csv")
print single_pass_clustering(3,ds)
