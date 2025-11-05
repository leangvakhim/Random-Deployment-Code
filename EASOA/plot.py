import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from calculation import (
    calculate_coverage,
)

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
        label='EASOA Convergence'
    )

    ax.set_title('EASOA Convergence Curve')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')

    if all_positive:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
    # ax.legend(labels=[formula])
    ax.legend()
    plt.show()

def plot_node_deployment(nodes, area_size, sensing_radius, monitoring_points, coverage_percent):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 8))
    # 1. plot each sensor node and coverage area
    for i, node in enumerate(nodes):
        ax.plot(node[0], node[1], 'bo', markersize=5, label='Sensor' if i == 0 else "")
        ax.text(node[0] + 0.5, node[1] + 0.5, str(i+1), fontsize=9, color='black')
        coverage_circle = plt.Circle(node, sensing_radius, color='green', alpha=0.15, label='Coverage Area' if i ==0 else "")
        ax.add_patch(coverage_circle)

    # 2. plot each monitoring point and color it by coverage status
    # for i, target_pos in enumerate(monitoring_points):
    #     distances_to_nodes = np.linalg.norm(nodes - target_pos, axis=1)
    #     is_covered = np.any(distances_to_nodes <= sensing_radius)
    #     color = 'green' if is_covered else 'red'
    #     label = 'Covered Target' if is_covered and 'Covered Target' not in ax.get_legend_handles_labels()[1] else \
    #             'Uncovered Target' if not is_covered and 'Uncovered Target' not in ax.get_legend_handles_labels()[1] else ""
    #     ax.plot(target_pos[0], target_pos[1], 'o', color=color, markersize=3, label=label)

    # for i, point in enumerate(monitoring_points):
    #     # point_2d = np.array(point)
    #     is_covered = calculate_coverage(nodes, area_size, sensing_radius, monitoring_points) > 0
    #     color = 'green' if is_covered else 'red'
    #     label = 'Covered Point' if is_covered and 'Covered point' not in ax.get_legend_handles_labels()[1] else \
    #     'Uncovered Point' if not is_covered and 'Uncovered Point' not in ax.get_legend_handles_labels()[1] else ""
    #     ax.plot(point[0], point[1], marker='o', color=color, markersize=3, label=label)

    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title(f'Final Sensor Node Deployment (Coverage: {coverage_percent:.2f}%)')
    ax.set_aspect('equal', 'box')
    plt.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    all_x = [s[0] for s in nodes] + [t[0] for t in monitoring_points]
    all_y = [s[1] for s in nodes] + [t[1] for t in monitoring_points]
    ax.set_xlim(min(all_x) - sensing_radius, max(all_x) + sensing_radius)
    ax.set_ylim(min(all_y) - sensing_radius, max(all_y) + sensing_radius)

    plt.show()