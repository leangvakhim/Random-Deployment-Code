import numpy as np

def initialize_sparrow_position(n, d, lower_bound, upper_bound):
    if isinstance(lower_bound, (int, float)):
        rand_matrix = np.random.rand(n, d)

        X = lower_bound + rand_matrix * (upper_bound - lower_bound)
    else:
        if len(lower_bound) != d or len(upper_bound) != d:
            raise ValueError("The length of lower_bound and upper_bound must be equal to single values or arrays of length 'd'.")

        X = np.zeros((n, d))
        for j in range(d):
            lb_j = lower_bound[j]
            ub_j = upper_bound[j]
            X[:, j] = lb_j + np.random.rand(n) * (ub_j - lb_j)

    return X

def calculate_fitness(X, benchmark_function):
    # n = X.shape[0]
    # Fx = np.zeros(n)

    # for i in range(n):
    #     Fx[i] = benchmark_function(X[i, :])

    # return Fx
    return benchmark_function(X)

def update_producers(X, sorted_indices, iter_max, ST, PD_count, d):
    R2 = np.random.rand()
    L = np.ones((1, d))

    for i_sorted in range(PD_count):
        i = sorted_indices[i_sorted]

        alpha = np.random.rand()
        if alpha == 0:
            alpha = 1e-6

        if R2 < ST:
            # Safe area (Wide Search area)
            exponent = - (i_sorted + 1) / (alpha * iter_max)
            X[i, :] = X[i, :] * np.exp(exponent)
        else:
            # Danger area (Flee to other area)
            Q = np.random.randn()
            X[i, :] = X[i, :] + Q * L

    return X

def update_scroungers(X, sorted_indices, PD_count, n, d, global_best_position, global_worst_position):
    L = np.ones((1, d))

    for i_sorted in range(PD_count, n):
        i = sorted_indices[i_sorted]

        paper_i = i_sorted + 1

        # i > n/2 (Starving scroungers)
        if paper_i > n / 2:
            Q = np.random.randn()
            exponent_denominator = paper_i ** 2
            exponent_numerator = global_worst_position - X[i, :]
            exponent = exponent_numerator / exponent_denominator
            X[i, :] = Q * np.exp(exponent)
        else:
            A = np.ones((1, d))
            rand_indices = np.random.rand(d) < 0.5
            A[0, rand_indices] = -1

            diff = np.abs(X[i, :] - global_best_position)

            C = np.sum(diff * A) / d

            step_simplified = C * L

            X[i, :] = global_best_position + step_simplified
    return X

def danger_aware(X, Fx, n, SD_count, global_best_fitness, global_best_position, global_worst_fitness, global_worst_position):
    epsilon = 1e-9

    danger_indices = np.random.choice(n, SD_count, replace=False)

    for i in danger_indices:
        f_i = Fx[i] # Current sparrow's fitness
        X_i = X[i, :] # Current sparrow's position

        # Sparrow is at the edge
        if f_i > global_best_fitness:
            beta = np.random.randn()
            X[i, :] = global_best_position + beta * np.abs(X_i - global_best_position)
        # Sparrow is at the middle (the best)
        elif f_i == global_best_fitness:
            K = np.random.uniform(-1, 1)

            numerator = np.abs(X_i - global_worst_position)
            denominator = (f_i - global_worst_fitness) + epsilon

            X[i, :] = X_i + K * (numerator / denominator)

    return X