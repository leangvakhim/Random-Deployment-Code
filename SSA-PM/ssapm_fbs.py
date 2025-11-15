import numpy as np

class flare_Burst_Search:
    def __init__(self, S_min, S_max, A_min_percent, A_max_percent):
        self.S_min = S_min
        self.S_max = S_max
        self.A_min_percent = A_min_percent
        self.A_max_percent = A_max_percent

    def fbs_update(self, danger_indices, population, fitness_values, fitness_function, lb_vec, ub_vec):
        if len(danger_indices) == 0:
            return population

        epsilon = np.finfo(float).eps

        danger_fitness = fitness_values[danger_indices]
        f_best_D = np.min(danger_fitness)
        f_worst_D = np.max(danger_fitness)

        search_width = ub_vec - lb_vec
        A_min = self.A_min_percent * search_width
        A_max = self.A_max_percent * search_width

        for i, sparrow_idx in enumerate(danger_indices):
            f_i = danger_fitness[i]
            if f_worst_D == f_best_D:
                S_i = self.S_min
            else:
                ratio = (f_i - f_best_D) / (f_worst_D - f_best_D + epsilon)
                S_i = self.S_min + int(np.round((self.S_max - self.S_min) * ratio))

            if f_worst_D == f_best_D:
                A_i = A_min
            else:
                A_i = A_min + (A_max - A_min) * ratio

            sparks = []
            sparks_fitness = []

            current_sparrow_pos = population[sparrow_idx]

            for s in range(S_i):
                R = 2 * np.random.rand(population.shape[1]) - 1
                X_spark_s = current_sparrow_pos + A_i * R
                X_spark_s = lb_vec + (X_spark_s - lb_vec) % (ub_vec - lb_vec + epsilon)
                sparks.append(X_spark_s)
                sparks_fitness.append(fitness_function(X_spark_s))

            if not sparks_fitness:
                continue

            best_spark_idx = np.argmin(sparks_fitness)
            f_best_spark = sparks_fitness[best_spark_idx]
            X_best_spark = sparks[best_spark_idx]

            if f_best_spark < f_i:
                population[sparrow_idx] = X_best_spark
                fitness_values[sparrow_idx] = f_best_spark

        return population