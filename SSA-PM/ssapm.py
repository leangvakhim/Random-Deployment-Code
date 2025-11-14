import numpy as np
from tqdm import tqdm
import time
import math
from ssapm_detector import stagnationDetector
from ssapm_atp import adaptive_Thermal_Attraction
from ssapm_rebirth import (
    ChaoticRebirth,
    LevyFlightRebirth
)

def ssapm(objective_function, iter_max, m_guilds, sparrow_per_guild, dim, lb, ub):
    start_time = time.time()
    t = 0

    x = np.random.uniform(lb, ub, size=())

    params = {
        # General
        'iter_max': iter_max,
        'dim': dim,
        'lb': lb,
        'ub': ub,

        # Guilds
        'm_guilds': m_guilds,
        'sparrow_per_guild': sparrow_per_guild,

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
        'a_min': 0.01,
        'a_max': 0.1,

        # Adaption Role Allocation
        'r_start': 0.8,
        'r_end': 0.2,
        'lambda_role': 2.0,

        # Multi-population Co-evolution
        'tau_comm': 10,

        # Original SSA
        'pd_ratio': 0.2,
        'st': 0.8
    }

    total_sparrows = sparrow_per_guild * m_guilds
    use_levy_flight = True
    detector = stagnationDetector(params['tau_stagnate'])
    chaotic_mech = ChaoticRebirth()
    levy_mech = LevyFlightRebirth()
    atp_mech = adaptive_Thermal_Attraction(total_sparrows, dim, params['t_0'], params['alpha_sa'], params['r_base_percent'], params['r_lambda'])

    population = np.random.uniform(lb, ub, size=(total_sparrows, dim))

    lb_vec = np.full(dim, lb)
    ub_vec = np.full(dim, ub)

    L_diagonal = np.linalg.norm(ub_vec - lb_vec)

    fitness_values = np.zeros(total_sparrows)

    mass_m = np.zeros(total_sparrows)
    mass_M = np.zeros(total_sparrows)

    for i in tqdm(range(iter_max), desc="SSA-PM Optimization Progress"):
        fitness_values[i] = objective_function(population[i])

        best_index = np.argmin(fitness_values)
        global_best_fit = fitness_values[best_index]
        worst_index = np.argmax(fitness_values)
        global_worst_fit = fitness_values[worst_index]

        trigger_activated = detector.update(global_best_fit)

        if trigger_activated:
            if use_levy_flight:
                X_phoenix = population[best_index].copy()
                new_X_phoenix = levy_mech.execute(X_phoenix)
                new_X_phoenix = np.clip(new_X_phoenix, lb, ub)
                population[best_index] = new_X_phoenix
                fitness_values[best_index] = objective_function(new_X_phoenix)
            else:
                population, replaced_idx = chaotic_mech.execute(
                    population,
                    fitness_values,
                    lb_vec,
                    ub_vec
                )
                fitness_values[replaced_idx] = objective_function(population[replaced_idx])

            detector.reset_counter()

        else:
            sorted_indices = np.argsort(fitness_values)
            num_producers = int(params['pd_ratio'] * total_sparrows)
            producer_indices = sorted_indices[:num_producers]
            scrounger_indices = sorted_indices[num_producers:]

            # Calculate G(t)
            G_t = params['g_0'] * np.exp(-params['alpha_gsa'] * t / iter_max)
            epsilon = np.finfo(float).eps
            best_fit_val = fitness_values[best_index]
            worst_fit_val = fitness_values[worst_index]

            if best_fit_val == worst_fit_val:
                mass_m.fill(1.0)
            else:
                mass_m = (worst_fit_val - fitness_values) / (worst_fit_val - best_fit_val + epsilon)

            mass_M = mass_m / (np.sum(mass_m) + epsilon)

            phoenix_position = population[best_index]
            phoenix_mass = mass_M[best_index]

            D_t = atp_mech.calculate_diversity(population, L_diagonal)
            R_heat_t = atp_mech.calculate_heat_radius(D_t, L_diagonal)
            t_0 = atp_mech.t_0

            R2 = np.random.rand()

            for i, p_idx in enumerate(producer_indices):
                alpha = np.random.rand()
                if R2 < params['st']:
                    population[p_idx] = population[p_idx] * np.exp(-i / (alpha * iter_max))
                else:
                    Q = np.random.rand(dim)
                    population[p_idx] = population[p_idx] + Q

            population = atp_mech.scrounger_update(
                scrounger_indices,
                population,
                objective_function,
                G_t,
                phoenix_position,
                phoenix_mass,
                global_best_fit,
                R_heat_t,
                t_0
            )

            atp_mech.cool_temperature()

            population = np.clip(population, lb, ub)

        time.sleep(0.05)

    for i in range(total_sparrows):
        fitness_values[i] = objective_function(population[i])
    final_best_fit = np.min(fitness_values)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Final Best Fitness: {final_best_fit:.4e}")
    print(f"Execution Time: {execution_time:.4f} seconds")

    return final_best_fit