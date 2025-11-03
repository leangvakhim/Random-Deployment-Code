import time
import numpy as np
from plot import plot_convergence_curve
from benchmark import (
    F1_function,
    F2_function,
    F3_function,
    F4_function,
    F5_function,
    F6_function,
)
from ssa import ssa

function_config = {
    "F1": {
        "function": F1_function,
        "dimension": 30,
        "lower_bound": -100,
        "upper_bound": 100,
        "formula": r'$F(x) = \sum_{i=1}^{n} x_i^2$'
    },
    "F2": {
        "function": F2_function,
        "dimension": 30,
        "lower_bound": -10,
        "upper_bound": 10,
        "formula": r'$F(x) = \sum_{i=1}^{n} |x_i| + \prod_{i=1}^{n} |x_i|$'
    },
    "F3": {
        "function": F3_function,
        "dimension": 30,
        "lower_bound": -100,
        "upper_bound": 100,
        "formula": r'$F(x) = \sum_{i=1}^{n} \left( \sum_{j=1}^{i} x_j \right)^2$'
    },
    "F4": {
        "function": F4_function,
        "dimension": 30,
        "lower_bound": -100,
        "upper_bound": 100,
        "formula": r'$F(x) = \max_{i} \{ |x_i| (1 <= i <= n) \} $'
    },
    "F5": {
        "function": F5_function,
        "dimension": 30,
        "lower_bound": -30,
        "upper_bound": 30,
        "formula": r'$F(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]$'
    },
    "F6": {
        "function": F6_function,
        "dimension": 30,
        "lower_bound": -100,
        "upper_bound": 100,
        "formula": r'$F(x) = \sum_{i=1}^{n} ([ x_i + 0.5 ])^2$'
    },
}

function_to_run = "F6"
dimension = function_config[function_to_run]["dimension"]
lower_bound = function_config[function_to_run]["lower_bound"]
upper_bound = function_config[function_to_run]["upper_bound"]
function_use = function_config[function_to_run]["function"]
function_formula = function_config[function_to_run]["formula"]
n_sparrows = 100
max_iterations = 1000
times = 30

# all_fitness_results = []
all_position_results = []
start_time = time.time()
for _ in range (times):
    global_best_position, global_best_fitness, convergence_curve = ssa(
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

plot_convergence_curve(convergence_curve, formula=function_formula)