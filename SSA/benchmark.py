import numpy as np

def F1_function(position):
    return np.sum(position ** 2)

def F2_function(position):
    return np.sum(np.abs(position)) + np.prod(np.abs(position))

# def F3_function()