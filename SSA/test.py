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

for _ in range (30):
    global_best_position, global_best_fitness = ssa(n_sparrows, dimension, lower_bound, upper_bound, max_iterations, F1_function)

print(f"Global Best Fitness: {global_best_fitness}")
print(f"Standard deviation of Global Best Position: {np.std(global_best_position)}")
print(f"Mean of Global Best Position: {np.mean(global_best_position)}")