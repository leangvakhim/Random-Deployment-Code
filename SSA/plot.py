import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_curve(convergence_curve_data):
    plt.figure(figsize=(10, 6))

    plt.plot(convergence_curve_data, label='SSA Convergence', color='green')

    plt.title('SSA Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')

    # plt.yscale('log')

    # plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()

    plt.show()