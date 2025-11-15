import numpy as np

class adaptive_Role_Allocator:
    def __init__(self, r_start, r_end, r_lambda):
        self.r_start = r_start
        self.r_end = r_end
        self.r_lambda = r_lambda

    def get_roles(self, t, iter_max, total_sparrows, fitness_values):
        time_ratio = t / iter_max
        decay_term = 1.0 - (time_ratio ** self.r_lambda)
        r_t = self.r_end + (self.r_start - self.r_end) * decay_term

        n_p_t = int(np.round(total_sparrows * r_t))
        if n_p_t < 1:
            n_p_t = 1

        n_s_t = total_sparrows - n_p_t

        sorted_indices = np.argsort(fitness_values)

        producer_indices = sorted_indices[:n_p_t]

        scrounger_indices = sorted_indices[n_s_t:]

        return producer_indices, scrounger_indices