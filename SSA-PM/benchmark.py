import numpy as np
import time
import math

def sphere(x):
    return np.sum(x**2)

def schwefel_2_21(x):
    return np.max(np.abs(x))

def schwefel_2_22(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def schwefel_1_2(x):
    return np.sum(np.cumsum(x)**2)

def quartic_noise(x):
    n = len(x)
    indices = np.arange(1, n+1)
    noise = np.random.random(n)
    return np.sum(indices * (x**4) + noise)

def rosenbrock(x):
    x_i = x[:-1]
    n_next = x[1:]
    return np.sum(100 * (n_next - x_i**2)**2 + (x_i - 1)**2)