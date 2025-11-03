import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_convergence_curve(convergence_curve_data):
    # seaborn alternative
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=convergence_curve_data,
        # palette="tab10",
        # linewidth=2.5,
        # ax=ax,
        # color='green',
        label='SSA Convergence'
    )

    ax.set_title('SSA Convergence Curve')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')

    ax.set_yscale('log')

    ax.legend()

    # matplotlib alternative
    # plt.figure(figsize=(10, 6))

    # plt.plot(convergence_curve_data, label='SSA Convergence', color='green')

    # plt.title('SSA Convergence Curve')
    # plt.xlabel('Iteration')
    # plt.ylabel('Fitness')

    # # plt.yscale('log')

    # # plt.grid(True, which="both", ls="--", linewidth=0.5)
    # plt.legend()

    plt.show()