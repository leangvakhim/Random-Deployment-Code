import numpy as np
from calculation import (
    initialize_sparrow_position,
    calculate_fitness,
    update_producers,
    update_scroungers,
    danger_aware,
)

def ssa(n, d, lb, ub, iter_max, benchmark_function):
    PD_percent = 0.2
    PD_count = int(n * PD_percent)

    SD_percent = 0.1
    SD_count = int(n * SD_percent)

    ST = 0.8 # Safety threshold

    X = initialize_sparrow_position(n, d, lb, ub)

    Fx = calculate_fitness(X, benchmark_function)

    best_fitness_index = np.argmin(Fx)
    global_best_fitness = Fx[best_fitness_index]
    global_best_position = X[best_fitness_index, :].copy()

    print("--- Starting SSA Optimization ---")

    for t in range(iter_max):

        sorted_indices = np.argsort(Fx)

        best_index = sorted_indices[0]
        worst_index = sorted_indices[-1]

        current_global_best_position = X[best_index, :].copy()
        current_global_best_fitness = Fx[best_index]
        current_global_worst_position = X[worst_index, :].copy()
        current_global_worst_fitness = Fx[worst_index]

        global_worst_position = X[worst_index, :].copy()

        # Equation 3 - Update producers
        X = update_producers(X, sorted_indices, iter_max, ST, PD_count, d)

        # Equation 4 - Update scroungers
        X = update_scroungers(X, sorted_indices, PD_count, n, d, global_best_position, global_worst_position)

        # Equation 5 - Danger aware sparrows
        X = danger_aware(X, Fx, n, SD_count, current_global_best_fitness, current_global_best_position, current_global_worst_fitness, current_global_worst_position)

        X = np.clip(X, lb, ub)

        Fx = calculate_fitness(X, benchmark_function)

        current_best_index = np.argmin(Fx)
        if Fx[current_best_index] < global_best_fitness:
            global_best_fitness = Fx[current_best_index]
            global_best_position = X[current_best_index, :].copy()

        if (t + 1) % 10 == 0:
            print(f"Iteration {t + 1}/{iter_max}, Best Fitness: {global_best_fitness}")

    # print(f"Global Best Position: {global_best_position}")
    # print(f"Global Best Fitness: {global_best_fitness}")
    # print(f"Mean Global Best Fitness Fitness: {np.std(global_best_position)}")

    return global_best_position, global_best_fitness

