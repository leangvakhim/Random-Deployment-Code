import numpy as np
import math

class ChaoticRebirth:
    def __init__(self, warmup_K=100, mu=4.0):
        self.warmup_K = warmup_K
        self.mu = mu

    def generate_chaotic_vector(self, d):
        z = np.random.rand(d)

        z = np.clip(z, 0.01, 0.99)
        z[z == 0.25] = 0.26
        z[z == 0.50] = 0.51
        z[z == 0.75] = 0.76


        for _ in range(self.warmup_K):
            z = self.mu * z * (1 - z)

        return z

    def execute(self, population, fitness_value, lb, ub):
        worst_index = np.argmin(fitness_value)
        d = population.shape[1]
        z_vector = self.generate_chaotic_vector(d)
        X_reborn = lb + z_vector * (ub - lb)
        population[worst_index] = X_reborn

        return population, worst_index

class LevyFlightRebirth:
    def __init__(self, beta=1.5, alpha=0.01):
        self.beta = beta
        self.alpha = alpha

        try:
            numerator = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
            denominator = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))

            self.sigma_u = (numerator / denominator) ** (1 / self.beta)

        except:
            print("Error")
            self.sigma_u = 0.69657

        self.sigma_v = 1.0

    def generate_levy_step(self, d):
        u = np.random.normal(0, self.sigma_u, d)
        v = np.random.normal(0, self.sigma_v, d)
        # s = u / |v|^(1/beta)
        step = u / (np.abs(v) ** (1/self.beta))
        return step

    def execute(self, X_phoenix):
        d=X_phoenix.shape[0]
        S_vector = self.generate_levy_step(d)
        X_reborn = X_phoenix + self.alpha * S_vector
        return X_reborn