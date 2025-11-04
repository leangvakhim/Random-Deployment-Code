import time
import numpy as np
from plot import plot_convergence_curve
from easoa import easoa
from calculation import (
    wsn_fitness_wrapper,
    calculate_coverage,
    calculate_variance
)

area_size = 50.0
num_nodes = 20
sensing_radius = 10.0

w1_coverage = 0.8
w2_variance = 0.1
w3_energy = 0.1

# Create a grid of monitor points to check coverage
# A 50x50 grid (2500 points)
grid_step = 1
xx, yy = np.meshgrid(np.arange(0, area_size, grid_step),
                     np.arange(0, area_size, grid_step))
monitoring_points = np.vstack([xx.ravel(), yy.ravel()]).T

n_sparrows = 50
max_iterations = 500
times = 10

dimension = num_nodes * 2
lower_bound = 0.0
upper_bound = area_size

function_to_use = lambda x: wsn_fitness_wrapper(
    X_population=x,
    num_nodes=num_nodes,
    area_size=area_size,
    sensing_radius=sensing_radius,
    monitor_points=monitoring_points,
    w1=w1_coverage,
    w2=w2_variance,
    w3=w3_energy
)

print("--- Running EASOA for WSN Node Placement ---")
print(f"Nodes: {num_nodes}, Area: {area_size}x{area_size}, Population: {n_sparrows}, Iterations: {max_iterations}")

all_position_results = []
start_time = time.time()

for i in range (times):
    print(f"\nRun {i+1}/{times}...")
    global_best_position, global_best_fitness, convergence_curve = easoa(
        n_sparrows,
        dimension,
        lower_bound,
        upper_bound,
        max_iterations,
        function_to_use
    )
    all_position_results.append(global_best_position)

end_time = time.time()
total_time = end_time - start_time
average_time = total_time / times

real_fitness_score = -global_best_fitness

print(f"\n--- EASOA WSN Results ---")
print(f"Average Time: {average_time:.2f} seconds")
print(f"Best Fitness (Score): {real_fitness_score:.4f}")

best_nodes = global_best_position.reshape(num_nodes, 2)
final_coverage = calculate_coverage(best_nodes, area_size, sensing_radius, monitoring_points)
final_variance = calculate_variance(best_nodes)

print(f"Final Coverage (R_cover): {final_coverage * 100:.2f}%")
print(f"Final Variance (D_var): {final_variance:.4f}")
# print(f"Best Node Positions:\n {best_nodes}")

# Plot the convergence
plot_convergence_curve(convergence_curve)