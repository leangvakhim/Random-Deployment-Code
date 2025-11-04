import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Line graphs
def plot_convergence_curve(convergence_curve_data):
    # seaborn alternative
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    curve_data = np.array(convergence_curve_data)
    all_positive = np.all(curve_data > 0)
    sns.lineplot(
        # x=range(len(curve_data)),
        # y=curve_data,
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

    if all_positive:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    # ax.legend(labels=[formula])
    ax.legend()
    plt.show()