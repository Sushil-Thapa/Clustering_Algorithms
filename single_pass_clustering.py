
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
import resource,time
import matplotlib.pyplot as plt

numberOfClusters = 5
discardSets = { 'sumOfAllPoints':None,
                'numberOfPoints':None,
                'sqSumOfAllPoints':None
}

#dataset = np.loadtxt("data/data.csv", skiprows=1, delimiter='\t',usecols=range(1,3))
dataset = np.loadtxt("data/generated_data_1000.csv")
backup_dataset = dataset
print('shape of dataset:',dataset.shape) #Gives number of datas and dimensions


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
# print clusterList
#fill the buffer points
perRate = 100
fraction = numberOfData//perRate #1000000/100 = 10000
firstIteration = True

#if the dataset is exhausted then finish.
#Other wise repeat from setp 2
startTime = time.time()
for i in range(perRate): # 100 iterations
    if i % 50== 0:
        print ".",

    bufferSet, dataset = dataset[:fraction,:],dataset[fraction:,:] #fill the buffer points

    # print('buffer:'+str(bufferSet.shape))
    # print('dataset:'+str(dataset.shape))
    if firstIteration:
        firstIteration = False
        # k_means = KMeans(n_clusters=numberOfClusters, init=randomClusterMeans,  n_init=1)

    else: #for each cluster update the sufficient statistics of the discard set with the points assignmed to the cluster

        for i in range(numberOfClusters):
            num = clusterList[i]['discardSets']['numberOfPoints']
            tmp = np.tile(clusterList[i]['mean'],(num,1))
            # print num,tmp.shape

            bufferSet = np.append(bufferSet,tmp, axis=0) #np.time(X,(3,1))
            # np.append(bufferSet,np.vstack([clusterList[i]['mean']]*clusterList[i]['discardSets']['numberOfPoints']), axis=0) #np.time(X,(3,1))
            # print 'after append:',str(bufferSet.shape)
        # k_means = KMeans(n_clusters=numberOfClusters, init=temp_init,  n_init=1)

#perform iterations of k-means on the poinnts and discard sets in the buffer,until convergence
#for this clustering,each discard set is treated like a regular point places at the mean of the discard set but weighted with the number of points in the discard set
    k_means = KMeans(n_clusters=numberOfClusters, init=randomClusterMeans,  n_init=1)
    k_means.fit(bufferSet)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    temp_init = k_means_cluster_centers
    k_means_labels_unique = np.unique(k_means_labels)

#TODO
#TODO put unique data points in tempClusters, puts Mean values from tmp too right now
    # tempClusters = {i: bufferSet[np.where(k_means_labels == i)] for i in range(k_means.n_clusters)}
    tempClusters={}
    for i in range(k_means.n_clusters):
        points = bufferSet[np.where(k_means_labels == i)]
        # print 'sss',points.shape,clusterList[i]['mean'].shape
        for point in range(points.shape[0]):
            if (point == clusterList[i]['mean']).all():
                print 'deleting'
                points = np.delete(points, np.where((point == clusterList[i]['mean'].all())), axis=0)
        tempClusters[i] =points
    # print tempClusters
#TODO

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



print('Time elapsed:',time.time() - startTime)
print("Max_ram_usage: %.2f MB.\n" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024)) #TODO make function f and use mem_usage = memory_usage(f) an its max

# for i in range(numberOfClusters):
#     print clusterList[i]

def plot(dataset, belongs_to):
    colors = ['r','g','b','c','k','y','m']

    fig, ax = plt.subplots()
    for index in range(dataset.shape[0]): #Number of data
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))
    plt.show()
# mean = np.array([clusterList[i]['mean'] for i in range(5)])
temp_dataset = np.array([[0,0]])
temp_dataset=np.delete(temp_dataset,0,0)

temp_belongs_to = k_means_labels
for i in range(numberOfClusters):
    # for j in range(tempClusters[i].shape[0]):
    # print temp_dataset,tempClusters[i][:2,:2],'...\n'
    temp_dataset = np.append(temp_dataset,tempClusters[i],axis=0)
    # print temp_dataset
    # print temp_dataset
    # for j in range(tempClusters[i].shape[0]):
    #     # print j
    #     temp_belongs_to.append(i)
    print temp_dataset.shape,np.array(temp_belongs_to).shape#,tempClusters[i][:5,:5]
    # raw_input()
# raw_input('press enter')
print(temp_dataset[:5,:5])
plot(temp_dataset,temp_belongs_to)
