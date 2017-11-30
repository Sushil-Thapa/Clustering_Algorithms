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


# @timeit
def kmeans(n_clusters):
    t0 = time.time()
    dataset = np.loadtxt("data/generated_data_100m.csv")
    kmeans_clustering.kmeans_clustering(n_clusters,dataset)
    results = {}
    results['time'] = time.time() - t0
    return results
# @timeit
def single_pass(n_clusters):
    # print 'wwtf'
    return single_pass_clustering.single_pass_clustering(n_clusters)


# @timeit
def ktree(n_clusters):
    t0 = time.time()
    order = 300
    dataset = np.loadtxt("data/generated_data_100m.csv")
    ktree_clustering.ktree_clustering(order,dataset)
    results = {}
    results['time']=time.time() - t0
    return results


def draw(algorithm,fig):
    fname ='complexities/'+algorithm+'_'+str(timer.file_suffix)+'.csv'
    data = pd.read_csv(fname,header=None)

    data.loc[-1] = [0,0]  # adding a row
    data.index = data.index + 1  # shifting index
    # data = data.sort_index()  # sorting by index

    fig.suptitle('Analysis of clustering algorithms', fontsize=16)

    xValues = data.iloc[:,0:1] #all columns, first row
    yValues = data.iloc[:,1:2] #all columns, second row

    # print xValues,yValues,type(xValues)
    plt.xlabel('Number of datas', fontsize=18)
    plt.ylabel('Execution time', fontsize=18)

    plt.plot(xValues,yValues,label=algorithm)
    plt.legend(loc='best') #location for alorithms labels
    # plt.pause(1)
    fig.canvas.draw()
    fig.show()

if __name__ == '__main__':
    n_clusters = 10
    fig = plt.gcf()
    fig.show()  #shows blank graph
    fig.canvas.draw()
    for algorithm in algorithms: #for each algorithm
        if algorithms.index(algorithm)<2:
            print 'skip',algorithm
            continue
        for frac in np.arange(500,501,500): # frac is += 5 percent of datas  ##JUST USING FULL DATASET JUST FOR OONCE RIGHT NOW.
            results = globals().get(algorithm, [])(n_clusters) # globals().get(algorithm, []) gives respective function from algorithm variable
            if n_clusters is None: # for first iter in algorithm, for result of ktree, get n_clusters to use later
                n_clusters = results['n_clusters']
        print 'algo:',algorithm,results['time']
        timer.saveFile(algorithm,results['time'])
        draw(algorithm,fig)
        timer.mode = 'w' #resets complexities file mode to write mode for another alorithm.
    raw_input('Analysis Complete.')
    # plt.show()
