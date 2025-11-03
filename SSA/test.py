import numpy as np
from calculation import (
    initialize_sparrow_position,
    calculate_fitness
)
from benchmark import F1_function
from ssa import ssa

n_sparrows = 100
dimension = 30
lower_bound = -100
upper_bound = 100
max_iterations = 1000

all_fitness_results = []
all_position_results = []

for _ in range (30):
    global_best_position, global_best_fitness = ssa(n_sparrows, dimension, lower_bound, upper_bound, max_iterations, F1_function)
    all_fitness_results.append(global_best_fitness)
    all_position_results.append(global_best_position)

print(f"Global Best Fitness: {global_best_fitness}")
print(f"Standard deviation of Global Best Position: {np.std(all_position_results)}")
print(f"Mean of Global Best Position: {np.mean(all_position_results)}")