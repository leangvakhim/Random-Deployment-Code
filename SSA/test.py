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
    F7_function,
    F8_function,
    F9_function,
    F10_function,
    F11_function,
    F12_function,
    F13_function,
    F14_function,
    F15_function,
    F16_function,
    F17_function,
    F18_function,
    F19_function,
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
    "F7": {
        "function": F7_function,
        "dimension": 30,
        "lower_bound": -1.28,
        "upper_bound": 1.28,
        "formula": r'$F(x) = \sum_{i=1}^{n} i x_i^4 + {random}[0, 1)$'
    },
    "F8": {
        "function": F8_function,
        "dimension": 30,
        "lower_bound": -500,
        "upper_bound": 500,
        "formula": r'$F(x) = \sum_{i=1}^{n} -x_i \sin(\sqrt{|x_i|})$'
    },
    "F9": {
        "function": F9_function,
        "dimension": 30,
        "lower_bound": -5.12,
        "upper_bound": 5.12,
        "formula": r'$F(x) = \sum_{i=1}^{n} [x_i^2 - 10 \cos(2\pi x_i) + 10]$'
    },
    "F10": {
        "function": F10_function,
        "dimension": 30,
        "lower_bound": -32,
        "upper_bound": 32,
        "formula": r'$F(x) = -20 \exp(-\frac{1}{2}\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}) - \exp(\frac{1}{n}\sum_{i=1}^{n} \cos(2\pi x_i)) + 20 + e$'
    },
    "F11": {
        "function": F11_function,
        "dimension": 30,
        "lower_bound": -600,
        "upper_bound": 600,
        "formula": r'$F(x) = \frac{1}{4000} \sum_{i=1}^{n} x_i^2 - \prod_{i=1}^{n} \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1$'
    },
    "F12": {
        "function": F12_function,
        "dimension": 30,
        "lower_bound": -50,
        "upper_bound": 50,
        "formula": r'$F(x) = \frac{\pi}{n} \{ 10\sin(\pi y_1) + \sum_{i=1}^{n-1} (y_i-1)^2 [1+10\sin^2(\pi y_{i+1})] + (y_n-1)^2 \} + \sum_{i=1}^{n} u(x_i, 10, 100, 4)$'
    },
    "F13": {
        "function": F13_function,
        "dimension": 2,
        "lower_bound": -5,
        "upper_bound": 5,
        "formula": r'$F(x) = 4x_1^2 - 2.1x_1^4 + \frac{1}{3}x_1^6 + x_1x_2 - 4x_2^2 + 4x_2^4$'
    },
    "F14": {
        "function": F14_function,
        "dimension": 2,
        "lower_bound": 0,
        "upper_bound": 14,
        "formula": r'$F(x) = \left[ 1 - \left| \frac{\sin[\pi(x_1 - 2)] \sin[\pi(x_2 - 2)]}{\pi^2(x_1 - 2)(x_2 - 2)} \right|^5 \right] \left[ 2 + (x_1 - 7)^2 + 2(x_2 - 7)^2 \right]$'
    },
    "F15": {
        "function": F15_function,
        "dimension": 2,
        "lower_bound": -10,
        "upper_bound": 10,
        "formula": r'$F(x) = -(|e^{|100 - \sqrt{x_1^2 + x_2^2} / \pi|} \sin(x_1) \sin(x_2)| + 1)^{-0.1}$'
    },
    "F16": {
        "function": F16_function,
        "dimension": 2,
        "lower_bound": -20,
        "upper_bound": 20,
        "formula": r'$F(x) = \left[ e^{-\sum_{i=1}^{n} (x_i / \beta)^{2m}} - 2e^{-\sum_{i=1}^{n} x_i^2} \right] \prod_{i=1}^{n} \cos^2(x_i), \beta=15, m=5$'
    },
    "F17": {
        "function": F17_function,
        "dimension": 4,
        "lower_bound": -5,
        "upper_bound": 5,
        "formula": r'$F(x) = \sum_{i=1}^{11} \left( a_i - \frac{x_1(b_i^2 + b_ix_2)}{b_i^2 + b_ix_3 + x_4} \right)^2$'
    },
    "F18": {
        "function": F18_function,
        "dimension": 3,
        "lower_bound": 0,
        "upper_bound": 1,
        "formula": r'$F(x) = - \sum_{i=1}^{4} c_i \exp \left( - \sum_{j=1}^{3} a_{ij} (x_j - p_{ij})^2 \right)$'
    },
    "F19": {
        "function": F19_function,
        "dimension": 6,
        "lower_bound": 0,
        "upper_bound": 1,
        "formula": r'$F(x) = - \sum_{i=1}^{4} c_i \exp \left( - \sum_{j=1}^{6} a_{ij} (x_j - p_{ij})^2 \right)$'
    },
}

function_to_run = "F19"
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