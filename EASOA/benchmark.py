import numpy as np

# Schwefel's Problem 1.2
def F1_function(x):
    inner_sums = np.cumsum(x, axis=1)
    squared_sums = inner_sums ** 2
    return np.sum(squared_sums, axis=1)

# High Conditional Elliptics
def F2_function(x):
    part1 = x[:, 0]**2
    part2 = (10**6) * np.sum(x[:, 1:]**2, axis=1)
    return part1 + part2
    # result = x[0]**2 + (10**6) * np.sum(x[1:]**2)
    # return result

# Ackley Function
def F3_function(x):
    dimension = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    avg_sum_sq = sum_sq / dimension
    sqrt_avg_sum_sq = np.sqrt(avg_sum_sq)
    part1 = -20 * np.exp(-0.5 * sqrt_avg_sum_sq)
    sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
    avg_sum_cos = sum_cos / dimension
    part2 = -np.exp(avg_sum_cos)
    total_sum = part1 + part2 + 20 + np.e
    return total_sum

# Griewank Function
def F4_function(x):
    dimension = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    part1 = sum_sq / 4000
    i = np.arange(1, dimension + 1)
    denominators = np.sqrt(i)
    cos_terms_input = x / denominators
    cos_terms = np.cos(cos_terms_input)
    part2 = np.prod(cos_terms, axis=1)
    total_sum = part1 - part2 + 1
    return total_sum
