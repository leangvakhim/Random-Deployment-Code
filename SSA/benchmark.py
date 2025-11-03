import numpy as np

def F1_function(x):
    return np.sum(x ** 2)

def F2_function(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3_function(x):
    n = len(x)
    total_sum = 0
    for i in range(n):
        inner_sum = np.sum(x[:i+1])
        total_sum += inner_sum
    return total_sum