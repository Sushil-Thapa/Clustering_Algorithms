from timer import timeit
import time
import timer
import kmeans_clustering
import single_pass_clustering
import ktree_clustering
import numpy as np
import plotter
import matplotlib.pyplot as plt
import pandas as pd


algorithms=['ktree','kmeans','single_pass']
colors = ['ro','go','bo','ko','yo','mo']


@timeit
def kmeans(n_clusters,selection,dataset):
    # print selection
    kmeans_clustering.kmeans_clustering(n_clusters,dataset)

@timeit
def single_pass(n_clusters,selection,dataset):
    single_pass_clustering.single_pass_clustering(n_clusters,dataset)


@timeit
def ktree(n_clusters,selection,dataset):
    order = 10
    return ktree_clustering.ktree_clustering(order,selection,dataset)

def draw(algorithm,fig):

    data = pd.read_csv('complexities/'+algorithm+'.csv',header=None)
    data.loc[0] = [0,0]

    fig.suptitle('Analysis of clustering algorithms', fontsize=16)

    xValues = data.iloc[:,0:1]
    yValues = data.iloc[:,1:2]

    plt.xlabel('Number of datas', fontsize=18)
    plt.ylabel('Execution time', fontsize=18)

    plt.plot(xValues,yValues,label=algorithm)
    plt.legend(loc='best')
    # plt.pause(1)
    fig.canvas.draw()

if __name__ == '__main__':
    n_clusters = None
    dataset = np.loadtxt("data/generated_data_10000.csv")
    np.random.shuffle(dataset.flat)

    num_instances, num_features = dataset.shape

    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    for j in range(0,3):
        algorithm = algorithms[j]
        for frac in np.arange(50,1001,50):
            selection = frac * num_instances / 1000
            # print selection
            # continue
            temp_selection = dataset[:selection,:selection]
            results = globals().get(algorithm, [])(n_clusters,selection,temp_selection)
            if n_clusters is None:
                n_clusters = results['n_clusters']
        draw(algorithm,fig)
        timer.mode = 'w'
    raw_input('Analysis Complete.')
    # plt.show()
