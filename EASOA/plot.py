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

def plot_node_deployment(nodes, area_size, sensing_radius, coverage_percent):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(nodes[:, 0], nodes[:, 1], c='blue', label='Sensor Nodes', zorder=5)
    for (x, y) in nodes:
        circle = plt.Circle((x, y), sensing_radius, color='blue', alpha=0.15, zorder=1)
        ax.add_patch(circle)

    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title(f'Final Sensor Node Deployment (Coverage: {coverage_percent:.2f}%)')
    ax.set_aspect('equal', 'box')
    ax.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()