import numpy as np
from benchmark import (
    sphere,
    schwefel_2_21,
    schwefel_2_22,
    schwefel_1_2,
    quartic_noise,
    rosenbrock
)
from ssapm import (
    ssapm,
)
function_config = {
    "f1": {"function": sphere, "dim": 30, "lb": -100, "ub": 100, "name": "Sphere function"},
    "f2": {"function": schwefel_2_21, "dim": 30, "lb": -100, "ub": 100, "name": "Schwefel 2.21 function"},
    "f3": {"function": schwefel_2_22, "dim": 30, "lb": -10, "ub": 10, "name": "Schwefel 2.22 function"},
    "f4": {"function": schwefel_1_2, "dim": 30, "lb": -100, "ub": 100, "name": "Schwefel 1.2 function"},
    "f5": {"function": quartic_noise, "dim": 30, "lb": -1.28, "ub": 1.28, "name": "Quartic noise function"},
    "f6": {"function": rosenbrock, "dim": 30, "lb": -30, "ub": 30, "name": "Rosenbrock function"},
}

func = "f5"
use_levy_flight = True
iter_max = 1000
n_sparrows_total = 500
m_guilds = 5
n_sparrows_per_guild = n_sparrows_total // m_guilds
# dim = 30
# lb = -100
# ub = 100
benchmark_function = function_config[func]['function']
dim = function_config[func]['dim']
lb = function_config[func]['lb']
ub = function_config[func]['ub']
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
    'r_role_lambda': 2.0,

    # Multi-population Co-evolution
    'tau_comm': 10,

    # Original SSA
    'pd_ratio': 0.2,
    'sd_ratio': 0.1,
    'st': 0.8
}
values_list = []
for _ in range(5):
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
    values_list.append(value)

# print(f"Mean of the fitness values: {np.mean(values_list)}")

print(f"Value is: {min(values_list):.4f}")