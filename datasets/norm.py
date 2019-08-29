import os
import sys

import numpy as np

def norm(arr, thres=None, max_=None):
    np.seterr(all='raise')
    try:
        ar = arr - np.mean(arr.astype(np.float64)).astype(np.float32) # overflow encountered in reduce
    except:
        from remote_pdb import RemotePdb
        from random import randint
        RemotePdb('127.0.0.1', 4444+randint(0,1000)).set_trace()

    '''
    try:
        arr -= np.mean(arr)
    except:
        __import__('pdb').set_trace() 
    '''
    if thres is not None and np.max(arr) < thres:
        arr = arr
        silence = True
    else:
        if max_ is not None:
            arr = arr / max_
        else:
            arr = arr / float(np.max(arr))
        silence = False
    return arr, silence

def norm_std(arr, thres=None, std=None):
    np.seterr(all='raise')
    try:
        arr -= np.mean(arr)
    except:
        __import__('pdb').set_trace() 
    if thres is not None and np.max(arr) < thres:
        arr = arr
        silence = True
    else:
        if std is not None:
            arr = arr / std
        else:
            arr = arr / float(np.std(arr))
        silence = False
    return arr, silence

