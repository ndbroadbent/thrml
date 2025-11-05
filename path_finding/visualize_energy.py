#!/usr/bin/env python3
"""Visualize the energy landscape and show how it guides sampling.

This shows the 'invisible hand' - the distance potentials that create
the water-flow effect.
"""
import numpy as np
import matplotlib.pyplot as plt
from thrml_maze_solver import _compute_distance_field


def visualize_energy_landscape(maze, start, goal, output_file="energy_landscape.png"):
    """Show the bidirectional distance potential that guides sampling."""
    H, W = maze.shape

    # Compute distance fields
    dist_from_start = _compute_distance_field(maze, start)
    dist_from_goal = _compute_distance_field(maze, goal)

    # Normalize
    max_start = np.max(dist_from_start[dist_from_start != np.inf])
    max_goal = np.max(dist_from_goal[dist_from_goal != np.inf])

    norm_start = dist_from_start / max_start
    norm_goal = dist_from_goal / max_goal

    # Set inf to 1 for visualization
    norm_start[norm_start == np.inf] = 1.0
    norm_goal[norm_goal == np.inf] = 1.0

    # Create combined potential (like our energy function does)
    alpha = 2.0
    bridge_bonus = 1.0

    potential = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if maze[i, j] == 0:
                start_pull = alpha * (1.0 - norm_start[i, j])
                goal_pull = alpha * (1.0 - norm_goal[i, j])
                base = (start_pull + goal_pull) / 2.0

                accessibility = (1.0 - norm_start[i, j]) * (1.0 - norm_goal[i, j])
                bridge = bridge_bonus * accessibility

                potential[i, j] = base + bridge

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Distance from start
    ax = axes[0, 0]
    im = ax.imshow(1.0 - norm_start, cmap='YlOrRd', interpolation='nearest')
    ax.plot(start[1], start[0], 'g*', markersize=20, label='Start')
    ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
    ax.set_title('Distance from Start\n(Red = close, Yellow = far)', fontsize=12)
    ax.legend()
    plt.colorbar(im, ax=ax, label='Proximity')
    ax.invert_yaxis()

    # Distance from goal
    ax = axes[0, 1]
    im = ax.imshow(1.0 - norm_goal, cmap='YlGnBu', interpolation='nearest')
    ax.plot(start[1], start[0], 'g*', markersize=20, label='Start')
    ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
    ax.set_title('Distance from Goal\n(Blue = close, Yellow = far)', fontsize=12)
    ax.legend()
    plt.colorbar(im, ax=ax, label='Proximity')
    ax.invert_yaxis()

    # Combined potential (what guides sampling!)
    ax = axes[1, 0]
    im = ax.imshow(potential, cmap='hot', interpolation='nearest')
    ax.plot(start[1], start[0], 'g*', markersize=20, label='Start')
    ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
    ax.set_title('Combined Energy Potential\n(Bright = high energy, guides activation)', fontsize=12)
    ax.legend()
    plt.colorbar(im, ax=ax, label='Energy')
    ax.invert_yaxis()

    # Maze with optimal path corridor
    ax = axes[1, 1]
    corridor = np.zeros((H, W))
    optimal_dist = dist_from_goal[start]
    for i in range(H):
        for j in range(W):
            if maze[i, j] == 0:
                total = dist_from_start[i, j] + dist_from_goal[i, j]
                if total <= optimal_dist + 2:  # Within 2 of optimal
                    corridor[i, j] = 1.0

    ax.imshow(corridor, cmap='Greens', interpolation='nearest')
    ax.imshow(np.ma.masked_where(maze == 0, maze), cmap='binary', interpolation='nearest', alpha=0.8)
    ax.plot(start[1], start[0], 'g*', markersize=20, label='Start')
    ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
    ax.set_title('Optimal Path Corridor\n(Green = cells on/near shortest path)', fontsize=12)
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nEnergy landscape saved to {output_file}")
    print("This shows the 'invisible hand' guiding the sampling!")
    plt.close()


if __name__ == "__main__":
    # Demo
    H = W = 20
    rng = np.random.default_rng(42)
    maze = np.zeros((H, W), dtype=np.uint8)

    mask = (rng.random((H, W)) < 0.08)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    visualize_energy_landscape(maze, start, goal)

