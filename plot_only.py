import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

algorithms=['ktree','kmeans','single_pass']

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
    fig.canvas.draw()


if __name__ == '__main__()':
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    for algorithm in algorithms:
        draw(algorithm,fig)
    raw_input('Visualization Complete.')
