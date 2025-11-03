import numpy as np

def F1_function(x):
    return np.sum(x ** 2, axis=1)

def F2_function(x):
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

def F3_function(x):
    inner_sums = np.cumsum(x, axis=1)
    squared_sums = inner_sums ** 2
    return np.sum(squared_sums, axis=1)
