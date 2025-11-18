import numpy as np

class WSNObjective:
    def __init__(self, L, W, N, rs, rc, re, lam, beta, grid_res=1.0):
        self.L = L
        self.W = W
        self.N = N
        self.rs = rs
        self.rc = rc
        self.re = re
        self.lam = lam
        self.beta = beta

        x_range = np.arange(0, L, grid_res)
        y_range = np.arange(0, W, grid_res)
        self.grid_points = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
        self.num_target_points = len(self.grid_points)
        self.theoretical_max_area = N * np.pi * (rs ** 2)

    def probabilistic_sensing_model(self, distance):
        mask_certain = distance <= (self.rc - self.re)
        mask_decay = (distance > (self.rc - self.re)) & (distance <= (self.rc + self.re))

        probs = np.zeros_like(distance)
        probs[mask_certain] = 1.0

        if np.any(mask_decay):
            d_decay = distance[mask_decay]
            alpha = self.re - self.rc + d_decay
            probs[mask_decay] = np.exp(-self.lam * (alpha ** self.beta))

        return probs

    def evaluate(self, S_vector):
        sensors = S_vector.reshape(self.N, 2)
        d_x = self.grid_points[:, 0][:, np.newaxis] - sensors[:, 0]
        d_y = self.grid_points[:, 1][:, np.newaxis] - sensors[:, 1]
        distance = np.sqrt(d_x ** 2 + d_y ** 2)

        p_matrix = self.probabilistic_sensing_model(distance)

        p_not_covered = 1.0 - p_matrix
        prob_grid_point_covered = 1.0 - np.prod(p_not_covered, axis=1)

        total_prob_sum = np.sum(prob_grid_point_covered)
        f = total_prob_sum / self.num_target_points

        covered_area = f * self.L * self.W
        C = covered_area / self.theoretical_max_area

        return f, C
