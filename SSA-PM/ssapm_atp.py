import numpy as np

class adaptive_Thermal_Attraction:
    def __init__(self, n_sparrows, dimension, t_0, sa_alpha, r_base_percent, r_lambda):
        self.velocity = np.zeros((n_sparrows, dimension))
        self.t_0 = t_0
        self.sa_alpha = sa_alpha
        self.r_base_percent = r_base_percent
        self.r_lambda = r_lambda

    def calculate_diversity(self, population, L):
        N, d = population.shape
        mean_position = np.mean(population, axis=0)
        distance_sum = 0
        for i in range(N):
            distance_sum += np.linalg.norm(population[i] - mean_position)

        diversity = (1.0 / (N * L + np.finfo(float).eps)) * distance_sum
        return diversity

    def calculate_heat_radius(self, diversity, L):
        r_base = self.r_base_percent * L
        r_heat_t = r_base * ((1.0 - diversity) ** self.r_lambda)
        return r_heat_t

    def cool_temperature(self):
        self.t_0 = self.sa_alpha * self.t_0

    def scrounger_update(self, scrounger_indices, population, fitness_function, g_t, pheonix_position, pheonix_mass, pheonix_fitness, R_heat_t, T_t):

        epsilon = np.finfo(float).eps

        for i in scrounger_indices:
            scrounger_pos = population[i]
            distance = np.linalg.norm(scrounger_pos - pheonix_position)
            acceleration = (g_t * pheonix_mass * (pheonix_position - scrounger_pos) / (distance + epsilon))
            rand = np.random.rand(population.shape[1])
            self.velocity[i] = rand * self.velocity[i] + acceleration
            X_i_attract = scrounger_pos + self.velocity[i]
            distance_to_pheonix = np.linalg.norm(X_i_attract - pheonix_position)
            if distance_to_pheonix < R_heat_t:
                f_attract = fitness_function(X_i_attract)
                delta_f = f_attract - pheonix_fitness
                # Calculate P_repel
                if delta_f > 0:
                    P_repel = 1.0
                else:
                    P_repel = np.exp(-delta_f / (T_t + epsilon))

                if np.random.rand() < P_repel:
                    repel_step = np.random.rand(population.shape[1])
                    population[i] = X_i_attract - repel_step * (X_i_attract - scrounger_pos)
                else:
                    population[i] = X_i_attract
            else:
                population[i] = X_i_attract

        return population