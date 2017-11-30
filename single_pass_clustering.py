
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
from numpy import ma
from itertools import islice

def single_pass_clustering(numberOfClusters):
    # print dataset.shape

    # numberOfClusters = 5
    discardSets = { 'sumOfAllPoints':None, #initial descardset skeleton
                    'numberOfPoints':None,
                    'sqSumOfAllPoints':None
    }

    # backup_dataset = dataset


    # numberOfData,numberOfFeatures = dataset.shape
    numberOfData,numberOfFeatures = 100000000,2

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
    N = fraction
    firstIteration = True

    #if the dataset is exhausted then finish.
    #Other wise repeat from setp 2
    startTime = time.time()
    tempClusters={}

    # for ith in range(perRate): # 100 iterations
    #     bufferSet_bak, dataset = dataset[:fraction,:],dataset[fraction:,:] #fill the buffer points
    with open("data/generated_data_100m.csv",'r') as infile:
        # print '-'
        for ith in range(perRate):
            gen = islice(infile,N)
            arr = np.loadtxt(gen)
            # print '.',
            bufferSet_bak =arr
            # print bufferSet_bak.shape
            # print arr
            if arr.shape[0]<N:
                print 'N<Length'
                # return time.time() - startTime
                # break
            if firstIteration: #do nothing at first
                firstIteration = False
                bufferSet = bufferSet_bak
            else: #for each cluster update the sufficient statistics of the discard set with the points assignmed to the cluster

                for i in range(numberOfClusters):
                    num = clusterList[i]['numberOfPoints'] #number of points in each previous kmeans cluster
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

            _tempClusters = {}

            # for i in range(numberOfClusters):
            #     points = bufferSet[np.where(k_means_labels == i)] #points in a cluster
            #     if (ith > 0):
            #         # print 'points before:',points.shape
            #         for j in range(points.shape[0]):
            #             # print 'mean:',clusterList[i]['mean'],'pointj',points[j],'--',points[j].shape,clusterList[i]['mean'].shape
            #             if np.array_equal(points[j],clusterList[i]['mean']):
            #                 # print 'deleting',points[j]
            #                 np.delete(points, np.where(np.array_equal(points[j],clusterList[i]['mean'])), axis=0)
            #                 # print 'points after:',points.shape
            #     tempClusters[i] = points

            for i in range(numberOfClusters):
                # points = bufferSet[np.where(k_means_labels == i)]
                # tempClusters[i] = ma.masked_equal(points,clusterList[i]['mean'] )
                if (ith < 1):
                    tempClusters[i] = bufferSet[np.where(k_means_labels == i)]
                else:
                    A = bufferSet_bak
                    B = bufferSet[np.where(k_means_labels == i)]
                    nrows, ncols = A.shape
                    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                           'formats':ncols * [A.dtype]}

                    C = np.intersect1d(A.view(dtype), B.view(dtype))

                    # This last bit is optional if you're okay with "C" being a structured array...
                    _tempClusters[i] = C.view(A.dtype).reshape(-1, ncols)
                    # print _tempClusters[i],'\n---\n',A,'\n---\n',B
                    # print tempClusters[i].shape
                    tempClusters[i] = np.append(tempClusters[i],_tempClusters[i],axis=0)
                    # print tempClusters[i].shape
                    # raw_input()

            # print k_means_cluster_centers,tempClusters
            # for i in range(numberOfClusters):
            #     tempDiscardSets = { 'sumOfAllPoints':np.sum(tempClusters[i], axis=0),
            #                     'numberOfPoints':tempClusters[i].shape[0],
            #                     'sqSumOfAllPoints':np.sum(np.square(tempClusters[i]), axis=0)
            #     }
            #     clusterList[i]={'mean':k_means_cluster_centers[i],
            #                     'discardSets':tempDiscardSets}

            for i in range(numberOfClusters):
                clusterList[i]={'mean':k_means_cluster_centers[i],
                                'numberOfPoints':tempClusters[i].shape[0]
                                }

        # print 'iteration +1'
        # a = raw_input()
        # print clusterList[i]['mean']

    # print("Max_ram_usage: %.2f MB.\n" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024)) #TODO make function f and use mem_usage = memory_usage(f) an its max
    results = {}
    results['time']=time.time() - startTime
    # print 'wtf'
    # raw_input(results)
    return results
# ds = np.loadtxt("data/generated_data.csv")
# ds = None
# print single_pass_clustering(5,ds)
