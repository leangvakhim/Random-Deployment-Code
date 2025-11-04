import numpy as np
from tqdm import tqdm
from calculation import (
    initialize_sparrow_position,
    calculate_fitness,
    update_producers,
    update_scroungers,
    reverse_elite_selection,
    brightness_driven_perturbation,
    dynamic_warning_update
)
def easoa(n, d, lb, ub, iter_max, benchmark_function):

    PD_percent = 0.2
    PD_count = int(n * PD_percent)
    SD_percent = 0.1
    SD_count = int(n * SD_percent)
    ST = 0.8 # Safety threshold

    elite_rate = 0.2 # Rate of elites for reverse selection
    elite_count = int(n * elite_rate)

    beta_initial = 0.5 # Brightness perturbation initial coeff
    gamma = 0.9        # Brightness perturbation attenuation
    alpha_pert = 0.1   # Brightness perturbation random factor

    delta_warn = 0.5   # Dynamic warning update coeff

    lb_vec = lb if isinstance(lb, np.ndarray) else np.full(d, lb)
    ub_vec = ub if isinstance(ub, np.ndarray) else np.full(d, ub)

    X = initialize_sparrow_position(n, d, lb_vec, ub_vec)
    Fx = calculate_fitness(X, benchmark_function)

    best_fitness_index = np.argmin(Fx)
    global_best_fitness = Fx[best_fitness_index]
    global_best_position = X[best_fitness_index, :].copy()

    convergence_curve = np.zeros(iter_max)

    for t in tqdm(range(iter_max), desc="EASOA Optimization Progress"):
        sorted_indices = np.argsort(Fx)
        X, Fx = reverse_elite_selection(
            X, Fx, sorted_indices, elite_count, lb_vec, ub_vec, benchmark_function
        )
        sorted_indices = np.argsort(Fx)
        best_index = sorted_indices[0]
        worst_index = sorted_indices[-1]

        current_global_best_position = X[best_index, :].copy()
        current_global_best_fitness = Fx[best_index]
        global_worst_position = X[worst_index, :].copy()
        X = update_producers(X, sorted_indices, iter_max, ST, PD_count, d)
        X = update_scroungers(X, sorted_indices, PD_count, n, d, current_global_best_position, global_worst_position)
        k = t
        K_max = iter_max
        X = brightness_driven_perturbation(
            X, Fx, n, d, k, K_max, beta_initial, gamma, alpha_pert
        )
        X = dynamic_warning_update(
            X, n, d, SD_count, current_global_best_position, delta_warn
        )
        X = np.clip(X, lb_vec, ub_vec)
        Fx = calculate_fitness(X, benchmark_function)
        current_best_index = np.argmin(Fx)
        if Fx[current_best_index] < global_best_fitness:
            global_best_fitness = Fx[current_best_index]
            global_best_position = X[current_best_index, :].copy()
        convergence_curve[t] = global_best_fitness

    return global_best_position, global_best_fitness, convergence_curve