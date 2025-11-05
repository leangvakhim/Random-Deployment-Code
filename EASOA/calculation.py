import numpy as np
from scipy.spatial.distance import cdist, pdist

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

def reverse_elite_selection(X, Fx, sorted_indices, elite_count, lb, ub, benchmark_function):
    elite_indices = sorted_indices[:elite_count]
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    x_elites = X[elite_indices, :]
    f_elites = Fx[elite_indices]
    # x_primes = ub + lb - x_elites
    x_primes = (x_max + x_min) - x_elites
    x_primes = np.clip(x_primes, lb, ub)
    f_primes = calculate_fitness(x_primes, benchmark_function)
    replace_mask = f_primes < f_elites
    indices_to_replace = elite_indices[replace_mask]
    X[indices_to_replace, :] = x_primes[replace_mask, :]
    Fx[indices_to_replace] = f_primes[replace_mask]
    return X, Fx

def brightness_driven_perturbation(X, Fx, n, d, k, K_max, beta_initial, gamma, alpha_pert):
    beta = beta_initial * (1 - (k / K_max))
    X_new = X.copy()
    for i in range(n):
        j = np.random.randint(0, n)
        while i == j:
            j = np.random.randint(0, n)

        if Fx[j] < Fx[i]:
            d_ij = np.linalg.norm(X[i, :] - X[j, :])
            perturbation = beta * np.exp(-gamma * (d_ij**2)) * (X[j, :] - X[i, :]) + alpha_pert * np.random.randn()
            X_new[i, :] = X[i, :] + perturbation
        # d_ij = np.linalg.norm(X[i, :] - X[j, :])
        # perturbation = beta * np.exp(-gamma * (d_ij**2)) * (X[j, :] - X[i, :]) + alpha_pert * np.random.randn()

        # X[i, :] = X[i, :] + perturbation

    return X

def dynamic_warning_update(X, n, d, SD_count, global_best_position, delta_warn):
    danger_indices = np.random.choice(n, SD_count, replace=False)
    r = np.random.rand(SD_count, d)
    i = danger_indices
    term1 = X[i, :] * (1 - delta_warn)
    term2 = delta_warn * r * global_best_position
    X[i, :] = term1 + term2
    return X

def calculate_coverage(sparrow_nodes, area_size, sensing_radius, monitor_points):
    num_monitor_points = monitor_points.shape[0]
    num_nodes = sparrow_nodes.shape[0]
    distance = cdist(sparrow_nodes, monitor_points)
    prob_si_pj = (distance <= sensing_radius).astype(int)
    prob_not_covered_by_si = 1.0 - prob_si_pj
    prob_not_covered_by_any = np.prod(prob_not_covered_by_si, axis=0)
    prob_covered_pj = 1.0 - prob_not_covered_by_any
    total_covered_points = np.sum(prob_covered_pj)
    R_cover = total_covered_points / num_monitor_points

    return R_cover

def calculate_variance(sparrow_nodes):
    var_x = np.var(sparrow_nodes[:, 0])
    var_y = np.var(sparrow_nodes[:, 1])
    total_var = var_x + var_y

    if total_var < 1e-9:
        return 1e9

    return 1.0 / total_var

def wsn_fitness_wrapper(X_population, num_nodes, area_size, sensing_radius, monitor_points, w1, w2, w3):
    n_sparrows = X_population.shape[0]
    fitness_values = np.zeros(n_sparrows)
    for i in range(n_sparrows):
        sparrow_vector = X_population[i, :]
        sparrow_nodes = sparrow_vector.reshape(num_nodes, 2)
        R_cover = calculate_coverage(sparrow_nodes, area_size, sensing_radius, monitor_points)
        D_var = calculate_variance(sparrow_nodes)
        E_total = 100/1000
        F = (w1 * R_cover) - (w2 * D_var) - (w3 * E_total)
        fitness_values[i] = -F

    return fitness_values