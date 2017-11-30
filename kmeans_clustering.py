import numpy as np
from sklearn.cluster import KMeans
import resource,time
import matplotlib.pyplot as plt
from timer import timeit
import timer,time

# def plot(dataset, belongs_to):
#     colors = ['r','g','b','c','k','y','m']
#     print dataset.shape,belongs_to.shape
#
#     fig, ax = plt.subplots()
#     for index in range(dataset.shape[0]): #Number of data
#         instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
#         for instance_index in instances_close:
#             ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))
#     plt.show()

def kmeans_clustering(n_clusters,dataset):
    k_means = KMeans(n_clusters=n_clusters, init='k-means++',  n_init=1) #init = random initial clusters,initilizes KMeans with optionns

    # print 'Dataset Shape: ',dataset.shape
    t0 = time.time()

    # dataset = np.append(dataset,dataset,axis=1)
    # print dataset.shape
    k_means.fit(dataset) # fits into desired dataset
    t2 = time.time()-t0
    # 
    # k_means_labels = k_means.labels_ #k_means onject has all the info about clusters
    # k_means_cluster_centers = k_means.cluster_centers_
    # temp_init = k_means_cluster_centers
    # k_means_labels_unique = np.unique(k_means_labels)

    # temp = {i: dataset[np.where(k_means_labels == i)] for i in range(k_means.n_clusters)} #for plotting theh clusters
    # data = []
    # belongs_to = []
    # for key, value in temp.iteritems():
    #     for val in value:
    #         data.append(list(val))
    #     for i in range(len(value)):
    #         belongs_to.append(key)
    # timer.saveFile('kmeans_temp',n_clusters,dataset.shape[0],t2)
    result = {}
    result['time'] = time.time() - t0
    return result

    # plot(np.array(data),np.array(belongs_to))
