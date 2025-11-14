import numpy as np
from tqdm import tqdm
import time
import math
from ssapm_detector import stagnationDetector
from ssapm_rebirth import (
    ChaoticRebirth,
    LevyFlightRebirth
)
from calculation import (
    initialize_sparrow_position,
    calculate_fitness,
    update_producers,
    update_scroungers,
    danger_aware,
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
        'g0': 100,
        'alpha_gsa': 20,
        'T0': 100,
        'alpha_sa': 0.95,

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
        'tau_comm': 10
    }

    total_sparrows = sparrow_per_guild * m_guilds
    use_levy_flight = False
    detector = stagnationDetector(params['tau_stagnate'])
    chaotic_mech = ChaoticRebirth()
    levy_mech = LevyFlightRebirth()

    population = np.random.uniform(lb, ub, size=(total_sparrows, dim))

    lb_vec = np.full(dim, lb)
    ub_vec = np.full(dim, ub)

    fitness_values = np.zeros(total_sparrows)

    for i in tqdm(range(iter_max), desc="SSA-PM Optimization Progress"):
        fitness_values[i] = objective_function(population[i])

        best_index = np.argmin(fitness_values)
        global_best_fit = fitness_values[best_index]

        trigger_activated = detector.update(global_best_fit)

        if trigger_activated:
            if use_levy_flight:
                X_phoenix = population[best_index]
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
            noise = np.random.normal(0, 0.1, population.shape)
            population = population + noise
            population = np.clip(population, lb, ub)

        time.sleep(0.1)

    for i in range(total_sparrows):
        fitness_values[i] = objective_function(population[i])
    final_best_fit = np.min(fitness_values)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Final Best Fitness: {final_best_fit:.4e}")
    print(f"Execution Time: {execution_time:.4f} seconds")

    return final_best_fit