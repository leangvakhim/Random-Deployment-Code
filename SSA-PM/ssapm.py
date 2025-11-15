import numpy as np
from tqdm import tqdm
import time
import math
from ssapm_guild import guild

def ssapm(objective_function, iter_max, m_guilds, n_sparrows_per_guild, params, dim, lb, ub):
    lb_vec = np.full(dim, lb)
    ub_vec = np.full(dim, ub)
    l_diagonal = np.linalg.norm(ub_vec - lb_vec)

    guilds = [
        guild(i, n_sparrows_per_guild, dim, lb_vec, ub_vec, l_diagonal, params, objective_function)
        for i in range(m_guilds)
    ]

    great_phoenix_pos = None
    great_phoenix_fit = float('inf')

    for t in tqdm(range(iter_max), desc="SSA-PM Optimization Progress"):

        t_str = f"Iter {t+1}/{iter_max}"

        for g in guilds:

            g.update_fitness()

            g.check_stagnation_and_rebirth(t, t_str)

            g.evolve_sparrows(t, iter_max)

        if t % params['tau_comm'] == 0 and t > 0:


            current_best_pos = None
            current_best_fit = float('inf')

            for g in guilds:
                pos, fit = g.get_best_sparrow()
                if fit < current_best_fit:
                    current_best_fit = fit
                    current_best_pos = pos

            # Update the all-time best
            if current_best_fit < great_phoenix_fit:
                great_phoenix_fit = current_best_fit
                great_phoenix_pos = current_best_pos

            for g in guilds:
                g.replace_worst_sparrow(great_phoenix_pos, great_phoenix_fit)

        else:
            current_best_fit = float('inf')
            for g in guilds:
                _, fit = g.get_best_sparrow()
                if fit < current_best_fit:
                    current_best_fit = fit

            if current_best_fit < great_phoenix_fit:
                great_phoenix_fit = current_best_fit


        time.sleep(0.05)

    final_best_fit = float('inf')
    for g in guilds:
        _, fit = g.get_best_sparrow()
        if fit < final_best_fit:
            final_best_fit = fit

    print(f"Iterations: {iter_max}")
    print(f"Guilds: {m_guilds} ({n_sparrows_per_guild} sparrows each)")
    print(f"Final Best Fitness: {final_best_fit:.4e}")

    return final_best_fit