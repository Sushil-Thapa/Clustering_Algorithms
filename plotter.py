import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

algorithms=['kmeans','single_pass','ktree']
colors = ['ro','go','bo','ko','yo','mo']



def draw(algorithm):
    # for algorithm in algorithms:
    # index = algorithms.index(algorithm)
    # print algorithm
    data = pd.read_csv('complexities/'+algorithm+'.csv',header=None)
    data.loc[0] = [0,0]

    xValues = data.iloc[:,0:1]
    yValues = data.iloc[:,1:2]

    plt.plot(xValues,yValues,label=algorithm)
    plt.pause(2)
    # plt.show()
    # return plt
    # print data
# draw()
