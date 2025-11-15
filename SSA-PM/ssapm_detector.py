import numpy as np
import time

class stagnation_Detector:
    def __init__(self, tau_stagnate = 10):
        self.tau_stagnate = tau_stagnate
        self.C_stagnate = 0
        self.f_best_previous = float('inf')

    def update(self, f_best_current):
        if f_best_current < self.f_best_previous:
            self.C_stagnate = 0
            self.f_best_previous = f_best_current
        else:
            self.C_stagnate += 1

        if self.C_stagnate > self.tau_stagnate:
            return True

        return False

    def reset_counter(self):
        self.C_stagnate = 0