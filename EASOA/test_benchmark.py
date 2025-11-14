import time
import numpy as np
from plot import plot_convergence_curve
from benchmark import (
    F1_function,
    F2_function,
    F3_function,
    F4_function,
)
from easoa import easoa

function_config = {
    "F1": {
        "function": F1_function,
        "dimension": 30,
        "lower_bound": -100,
        "upper_bound": 100,
        "name": "Swefel's Problem 1.2",
        # "formula": r'$F(x) = \sum_{i=1}^{n} x_i^2$'
    },
    "F2": {
        "function": F2_function,
        "dimension": 30,
        "lower_bound": -100,
        "upper_bound": 100,
        "name": "High Conditional Elliptics",
        # "formula": r'$F(x) = \sum_{i=1}^{n} |x_i| + \prod_{i=1}^{n} |x_i|$'
    },
    "F3": {
        "function": F3_function,
        "dimension": 30,
        "lower_bound": -32,
        "upper_bound": 32,
        "name": "Ackley",
        # "formula": r'$F(x) = \sum_{i=1}^{n} \left( \sum_{j=1}^{i} x_j \right)^2$'
    },
    "F4": {
        "function": F4_function,
        "dimension": 30,
        "lower_bound": -600,
        "upper_bound": 600,
        "name": "Griewank",
        # "formula": r'$F(x) = \max_{i} \{ |x_i| (1 <= i <= n) \} $'
    },
}

function_to_run = "F2"
dimension = function_config[function_to_run]["dimension"]
lower_bound = function_config[function_to_run]["lower_bound"]
upper_bound = function_config[function_to_run]["upper_bound"]
function_use = function_config[function_to_run]["function"]
function_name = function_config[function_to_run]["name"]
# function_formula = function_config[function_to_run]["formula"]
n_sparrows = 50
max_iterations = 1000
times = 10

# all_fitness_results = []
all_position_results = []
start_time = time.time()
for _ in range (times):
    global_best_position, global_best_fitness, convergence_curve = easoa(
        n_sparrows,
        dimension,
        lower_bound,
        upper_bound,
        max_iterations,
        function_use
    )
    # all_fitness_results.append(global_best_fitness)
    # all_position_results.append(global_best_position)

end_time = time.time()
total_time = end_time - start_time
average_time = total_time / times

print(f"Global Best Fitness: {global_best_fitness}")
# print(f"Standard deviation of Global Best Position: {np.std(all_position_results)}")
# print(f"Mean of Global Best Position: {np.mean(all_position_results)}")
print(f"Average Time: {average_time:.2f} seconds")
# print(f"Total Time: {total_time:.2f} seconds")

plot_convergence_curve(convergence_curve, global_best_fitness, function_name)