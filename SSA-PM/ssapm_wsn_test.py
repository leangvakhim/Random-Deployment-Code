import numpy as np
import time
from ssapm import ssapm
from ssapm_wsn import WSNObjective
from ssapm_plot import plot_node_deployment, plot_convergence_curve

L = 50.0
W = 50.0
N = 30

rs = 7.5
rc = 5.0
re = 2.5

lam = 0.6
beta = 0.8

grid_res = 1.0

iter_max = 500
n_sparrows_total = 100
m_guilds = 5
n_sparrows_per_guild = n_sparrows_total // m_guilds

params = {
    # General
    'iter_max': iter_max,
    'dim': N * 2,
    'lb': 0.0,
    'ub': L,
    'use_levy_flight': True,

    # Guilds
    'm_guilds': m_guilds,
    'n_sparrows_per_guild': n_sparrows_per_guild,

    # Population Rebirth (PR)
    'tau_stagnate': 20,
    'beta_levy': 1.5,

    # Adaptive Thermal Perturbation (ATP)
    'g_0': 100,
    'alpha_gsa': 20,
    't_0': 100,
    'alpha_sa': 0.95,
    'r_base_percent': 0.05,
    'r_lambda': 2.0,

    # Flare Burst Search (FBS)
    's_min': 2,
    's_max': 10,
    'a_min_percent': 0.01,
    'a_max_percent': 0.1,
    'sd_ratio': 0.1,

    # Adaption Role Allocation
    'r_start': 0.8,
    'r_end': 0.2,
    'r_role_lambda': 2.0,

    # Multi-population Co-evolution
    'tau_comm': 10,

    # Original SSA
    'pd_ratio': 0.2,
    'sd_ratio': 0.1,
    'st': 0.8
}

wsn_objective = WSNObjective(
    L=L, W=W, N=N, rs=rs, rc=rc, re=re,
    lam=lam, beta=beta, grid_res=grid_res
)

def fitness_wrapper(S_vector):
    f_coverage, c_efficiency = wsn_objective.evaluate(S_vector)

    return -f_coverage

print("--- Running SSA-PM for WSN Node Coverage ---")
print(f"Area: {L}x{W}m, Nodes: {N}, Iterations: {iter_max}, Population: {n_sparrows_total}")

start_time = time.time()

best_position_vector, best_fitness_value, convergence_history = ssapm(
    objective_function=fitness_wrapper,
    iter_max=iter_max,
    m_guilds=m_guilds,
    n_sparrows_per_guild=n_sparrows_per_guild,
    params=params,
    dim=N * 2,
    lb=0.0,
    ub=L
)

end_time = time.time()
print(f"--- Optimization Finished in {end_time - start_time:.2f} seconds ---")

if best_position_vector is not None:
    best_nodes_coordinates = best_position_vector.reshape(N, 2)

    final_f, final_c = wsn_objective.evaluate(best_position_vector)

    print(f"\nFinal Best Fitness (Negative Coverage): {best_fitness_value:.4f}")
    print(f"Coverage Rate (f): {final_f * 100:.2f}%")
    print(f"Coverage Efficiency (C): {final_c * 100:.2f}%")

    plot_convergence_curve(convergence_history, best_fitness_value)

    plot_node_deployment(
        nodes=best_nodes_coordinates,
        area_size_l=L,
        area_size_w=W,
        sensing_radius_rs=rs,
        coverage_percent_f=final_f * 100,
        efficiency_percent_c=final_c * 100
    )

else:
    print("Optimization did not find a valid solution.")