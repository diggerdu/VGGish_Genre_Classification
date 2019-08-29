'''
Usage:
    test.py [options]

Arguments:
    --infile=<in>
    --reffile=<ref>

'''
#import plotille
import matplotlib.pyplot as plt
import numpy as np

def plotCurve(arrList):
    fig = plotille.Figure()
    for arr in arrList:
        fig.scatter(np.arange(np.size(arr)), arr)
    print(fig.show(legend=True))

def plotCurveMat(arrList, fn='whatever.png', labels=None):
    fig, ax = plt.subplots()
    for idx, arr in enumerate(arrList):
        arr_x = np.arange(np.size(arr))
        if labels is None:
            ax.plot(arr_x, arr, '--', linewidth=1, label='{}th line'.format(idx))
        else:
            ax.plot(arr_x, arr, '--', linewidth=1, label=labels[idx])
            
    plt.legend()
    plt.savefig(fn, dpi=360)

