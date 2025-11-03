import time
import numpy as np
from plot import plot_convergence_curve
from benchmark import (
    F1_function,
    F2_function
)
from ssa import ssa

n_sparrows = 100
dimension = 30
lower_bound = -100
upper_bound = 100
max_iterations = 1000
times = 30

# all_fitness_results = []
all_position_results = []
start_time = time.time()
for _ in range (times):
    global_best_position, global_best_fitness, convergence_curve = ssa(n_sparrows, dimension, lower_bound, upper_bound, max_iterations, F1_function)
    # global_best_position, global_best_fitness, convergence_curve = ssa(n_sparrows, dimension, lower_bound, upper_bound, max_iterations, F2_function)
    # all_fitness_results.append(global_best_fitness)
    all_position_results.append(global_best_position)

end_time = time.time()
total_time = end_time - start_time
average_time = total_time / times

print(f"Global Best Fitness: {global_best_fitness}")
# print(f"Standard deviation of Global Best Position: {np.std(all_position_results)}")
print(f"Mean of Global Best Position: {np.mean(all_position_results)}")
print(f"Average Time: {average_time:.2f} seconds")
# print(f"Total Time: {total_time:.2f} seconds")

plot_convergence_curve(convergence_curve)