import numpy as np
from ssapm_detector import stagnation_Detector
from ssapm_atp import adaptive_Thermal_Attraction
from ssapm_fbs import flare_Burst_Search
from ssapm_role import adaptive_Role_Allocator
from ssapm_rebirth import (
    chaotic_Rebirth,
    levy_Flight_Rebirth
)

class guild:

    def __init__(self, guild_id, n_sparrow, dimension, lb_vec, ub_vec, l_diagonal, params, objective_function):
        self.guild_id = guild_id
        self.n_sparrow = n_sparrow
        self.dimension = dimension
        self.lb_vec = lb_vec
        self.ub_vec = ub_vec
        self.l_diagonal = l_diagonal
        self.params = params
        self.objective_function = objective_function

        self.population = np.random.uniform(lb_vec[0], ub_vec[0], size=(self.n_sparrow, self.dimension))
        self.fitness_values = np.zeros(self.n_sparrow)
        self.mass_m = np.zeros(self.n_sparrow)
        self.mass_M = np.zeros(self.n_sparrow)

        self.detector = stagnation_Detector(params['tau_stagnate'])
        self.chaotic_mech = chaotic_Rebirth()
        self.levy_mech = levy_Flight_Rebirth()

        self.atp_mech = adaptive_Thermal_Attraction(
            self.n_sparrow, self.dimension,
            params['t_0'], params['alpha_sa'],
            params['r_base_percent'], params['r_lambda']
        )
        self.fbs_mech = flare_Burst_Search(
            params['s_min'], params['s_max'],
            params['a_min_percent'], params['a_max_percent']
        )
        self.role_allocator = adaptive_Role_Allocator(
            params['r_start'], params['r_end'], params['r_role_lambda']
        )

        print(f"[Guild {self.guild_id}] Initialized with {self.n_sparrow} sparrows.")

    def update_fitness(self):
        for i in range(self.n_sparrow):
            self.fitness_values[i] = self.objective_function(self.population[i])

    def check_stagnation_and_rebirth(self, t, global_t_str):
        best_index = np.argmin(self.fitness_values)
        best_fit = self.fitness_values[best_index]

        trigger = self.detector.update(best_fit, global_t_str, self.guild_id)

        if trigger:
            if self.params['use_levy_flight']:
                # Levy flight
                X_phoenix = self.population[best_index].copy()
                new_X_phoenix = self.levy_mech.execute(X_phoenix)
                new_X_phoenix = np.clip(new_X_phoenix, self.lb_vec[0], self.ub_vec[0])
                self.population[best_index] = new_X_phoenix
                self.fitness_values[best_index] = self.objective_function(new_X_phoenix)
            else:
                # Chaotic rebirth
                self.population, replaced_idx = self.chaotic_mech.execute(
                    self.population, self.fitness_values, self.lb_vec, self.ub_vec
                )
                self.fitness_values[replaced_idx] = self.objective_function(self.population[replaced_idx])

            self.detector.reset_counter()

    def evolve_sparrows(self, t, iter_max):
        epsilon = np.finfo(float).eps

        producer_indices, scrounger_indices = self.role_allocator.execute(
            t, iter_max, self.n_sparrow, self.fitness_values
        )

        best_index = np.argmin(self.fitness_values)
        worst_index = np.argmax(self.fitness_values)
        best_fit_val = self.fitness_values[best_index]
        worst_fit_val = self.fitness_values[worst_index]

        g_t = self.params['g_0'] * np.exp(-self.params['alpha_gsa'] * t / iter_max)

        if best_fit_val == worst_fit_val:
            self.mass_m.fill(1.0)
        else:
            self.mass_m = (worst_fit_val - self.fitness_values) / (worst_fit_val - best_fit_val + epsilon)

        self.mass_M = self.mass_m / (np.sum(self.mass_m) + epsilon)

        phoenix_position = self.population[best_index]
        phoenix_mass = self.mass_M[best_index]

        D_t = self.atp_mech.calculate_diversity(self.population, self.l_diagonal)
        R_heat_t = self.atp_mech.calculate_heat_radius(D_t, self.l_diagonal)
        t_0 = self.atp_mech.t_0

        R2 = np.random.rand()
        ST = self.params['st']

        for i, p_idx in enumerate(producer_indices):
            alpha = np.random.rand()
            if R2 < ST:
                self.population[p_idx] = self.population[p_idx] * np.exp(-i / (alpha * iter_max))
            else:
                Q = np.random.rand(self.dimension)
                self.population[p_idx] = self.population[p_idx] + Q

        self.population = self.atp_mech.scrounger_update(
            scrounger_indices, self.population, self.objective_function,
            self.G_t, self.phoenix_position, phoenix_mass,
            best_fit_val, R_heat_t, t_0
        )

        self.atp_mech.cool_temperature()

        num_danger = int(self.n_sparrow * self.params['sd_ratio'])
        all_indices = np.arange(self.n_sparrow)
        np.random.shuffle(all_indices)
        danger_indices = all_indices[:num_danger]

        self.population = self.fbs_mech.fbs_update(
            danger_indices, self.population, self.fitness_values,
            self.objective_function, self.lb_vec, self.ub_vec
        )

        self.population = np.clip(self.population, self.lb_vec[0], self.ub_vec[0])

    def get_best_sparrow(self):
        best_index = np.argmin(self.fitness_values)
        return self.population[best_index], self.fitness_values[best_index]

    def replace_worst_sparrow(self, new_sparrow_pos, new_sparrow_fitness):
        worst_index = np.argmax(self.fitness_values)
        self.population[worst_index] = new_sparrow_pos
        self.fitness_values[worst_index] = new_sparrow_fitness



