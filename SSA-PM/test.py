import numpy as np
from benchmark import (
    sphere_function,
)
from ssapm import (
    ssapm,
)

benchmark_function = sphere_function
use_levy_flight = True
iter_max = 100
n_sparrows_total = 500
m_guilds = 4
n_sparrows_per_guild = n_sparrows_total // m_guilds
dim = 30
lb = -100
ub = 100
params = {
    # General
    'iter_max': iter_max,
    'dim': dim,
    'lb': lb,
    'ub': ub,
    'use_levy_flight': use_levy_flight,

    # Guilds
    'm_guilds': m_guilds,
    'n_sparrows_per_guild': n_sparrows_per_guild,

    # Population Rebirth (PR)
    'tau_stagnate': 10,
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
    'r_lambda_role': 2.0,

    # Multi-population Co-evolution
    'tau_comm': 10,

    # Original SSA
    'pd_ratio': 0.2,
    'sd_ratio': 0.1,
    'st': 0.8
}
# values_list = []
for _ in range(10):
    value = ssapm(
        objective_function=benchmark_function,
        iter_max=iter_max,
        m_guilds=m_guilds,
        n_sparrows_per_guild=n_sparrows_per_guild,
        params=params,
        dim=dim,
        lb=lb,
        ub=ub,
    )
    # values_list.append(value)

# print(f"Mean of the fitness values: {np.mean(values_list)}")

# print(f"Value is: {value}")