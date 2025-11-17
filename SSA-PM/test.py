import numpy as np
from benchmark import (
    sphere,
    schwefel_2_21,
    schwefel_2_22,
    schwefel_1_2,
    quartic_noise,
    rosenbrock,
    griewank,
    rastrigin,
    ackley,
    salomon,
    xin_she_yang,
    penalized_1,
    penalized_2,
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
    "f7": {"function": griewank, "dim": 30, "lb": -600, "ub": 600, "name": "Griewank function"},
    "f8": {"function": rastrigin, "dim": 30, "lb": -5.12, "ub": 5.12, "name": "Rastrigin function"},
    "f9": {"function": ackley, "dim": 30, "lb": -32, "ub": 32, "name": "Ackley function"},
    "f10": {"function": salomon, "dim": 30, "lb": -20, "ub": 20, "name": "Salomon function"},
    "f11": {"function": xin_she_yang, "dim": 30, "lb": -5, "ub": 5, "name": "Xin she yang function"},
    "f12": {"function": penalized_1, "dim": 30, "lb": -50, "ub": 50, "name": "Penalized 1 function"},
    "f13": {"function": penalized_2, "dim": 30, "lb": -50, "ub": 50, "name": "Penalized 2 function"},
}

values_list = []
use_levy_flight = True
iter_max = 100
n_sparrows_total = 500
m_guilds = 1
n_sparrows_per_guild = n_sparrows_total // m_guilds
for f in function_config.items():
    benchmark_function = f[1]['function']
    benchmark_name = f[1]['name']
    dim = f[1]['dim']
    lb = f[1]['lb']
    ub = f[1]['ub']
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

    for _ in range(1):
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
        print(f"Benchmark name: {benchmark_name}")

    non_zero_values = [v for v in values_list if v > 0]

    if non_zero_values:
        best_val = np.min(non_zero_values)
        print(f"The best value is: {best_val:.4e}")
    else:
        print("All values are 0.")