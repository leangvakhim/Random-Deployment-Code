import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_node_deployment(nodes, area_size_l, area_size_w, sensing_radius_rs, coverage_percent_f, efficiency_percent_c):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, node in enumerate(nodes):
        ax.plot(node[0], node[1], 'bo', markersize=6, label='Sensor Node' if i == 0 else "")
        ax.text(node[0] + 0.5, node[1] + 0.5, str(i+1), fontsize=9, color='black')
        coverage_circle = plt.Circle(node, sensing_radius_rs, color='green', alpha=0.15, label='Sensing Area (rs)' if i == 0 else "")
        ax.add_patch(coverage_circle)

    ax.set_xlim(0, area_size_l)
    ax.set_ylim(0, area_size_w)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')

    title_str = (
        f'Final Sensor Node Deployment\n'
        f'Coverage Rate (f): {coverage_percent_f:.2f}% | '
        f'Coverage Efficiency (C): {efficiency_percent_c:.2f}%'
    )
    ax.set_title(title_str)

    ax.set_aspect('equal', 'box')
    plt.grid(True, linestyle='--', alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.show()

def plot_convergence_curve(convergence_curve_data, best_fitness):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    curve_data = np.array(convergence_curve_data)

    all_non_positive = np.all(curve_data <= 1e-9)

    plot_data = -curve_data

    sns.lineplot(
        data=plot_data,
        label='SSA-PM Coverage Convergence'
    )

    ax.set_title(f"SSA-PM Convergence Curve (Final Best Coverage: {-best_fitness:.4f})")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Coverage Rate (f)')

    # if all_non_positive and np.all(curve_data < 0):
    #     ax.set_yscale('symlog') # Use 'symlog' for negative values
    # else:
    #     ax.set_yscale('linear')

    ax.set_yscale('linear')
    ax.legend()
    plt.show()