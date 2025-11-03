import numpy as np

def F1_function(x):
    return np.sum(x ** 2, axis=1)

def F2_function(x):
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

# Schewfel's Problem 1.2
def F3_function(x):
    inner_sums = np.cumsum(x, axis=1)
    squared_sums = inner_sums ** 2
    return np.sum(squared_sums, axis=1)

def F4_function(x):
    return np.max(np.abs(x), axis=1)

def F5_function(x):
    x_i = x[:, :-1]
    x_i_plus_1 = x[:, 1:]
    term1 = 100 * (x_i_plus_1 - x_i**2)**2
    term2 = (x_i - 1)**2
    total_sum = np.sum(term1 + term2, axis=1)
    return total_sum

# Step function
def F6_function(x):
    x_plus_0_5 = x + 0.5
    # floored_x = np.floor(x_plus_0_5)
    # squared_values = floored_x**2
    squared_values = x_plus_0_5**2
    total_sum = np.sum(squared_values, axis=1)
    return total_sum


