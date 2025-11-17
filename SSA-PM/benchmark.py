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

def u_func(x, a, k, m):
        result = np.zeros_like(x)
        mask_pos = x > a
        mask_neg = x < -a

        result[mask_pos] = k * (x[mask_pos] - a)**m
        result[mask_neg] = k * (-x[mask_neg] - a)**m
        return np.sum(result)

def penalized_2(x):
    d = len(x)
    a, k, m = 5, 100, 4
    term1 = np.sin(3 * np.pi * x[0])**2
    x_i = x[:-1]
    x_next = x[1:]
    term2 = np.sum((x_i - 1)**2 * (1 + np.sin(3 * np.pi * x_next)**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    penalty = u_func(x, a, k, m)
    return 0.1 * (term1 + term2 + term3) + penalty

def penalized_1(x):
    d = len(x)
    a, k, m = 10, 100, 4
    y = 1 + (x + 1) / 4
    term1 = 10 * np.sin(np.pi * y[0])
    y_i = y[:-1]
    y_next = y[1:]
    term2 = np.sum((y_i - 1)**2 * (1 + 10 * np.sin(np.pi * y_next)**2))
    term3 = (y[-1] - 1)**2
    penalty = u_func(x, a, k, m)
    return (np.pi / d) * (term1 + term2 + term3) + penalty

def griewank(x):
    indices = np.arange(1, len(x) + 1)
    sum_part = np.sum(x**2 / 4000)
    prod_part = np.prod(np.cos(x / np.sqrt(indices)))
    return sum_part - prod_part + 1

def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    d = len(x)
    a = 20
    b = 0.2
    c = 2 * np.pi

    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)

    return term1 + term2 + a + np.exp(1)

def salomon(x):
    r = np.sqrt(np.sum(x**2))
    return 1 - np.cos(2 * np.pi * r) + 0.1 * r

def xin_she_yang(x):
    sum_abs = np.sum(np.abs(x))
    exponent = np.sum(np.sin(x**2))
    return sum_abs * np.exp(-exponent)