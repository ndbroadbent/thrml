#!/usr/bin/env python3
"""Demo script for THRML maze solver.

This script demonstrates the THRML-based maze solver on various maze configurations.
Run with: python demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from thrml_maze_solver import solve_with_thrml


def demo_open_maze():
    """Demonstrate solving a completely open maze."""
    print("\n" + "=" * 60)
    print("Demo 1: Open 32x32 Maze")
    print("=" * 60)

    H = W = 32
    maze = np.zeros((H, W), dtype=np.uint8)
    start = (0, 0)
    goal = (H - 1, W - 1)

    print(f"Solving {H}x{W} open maze from {start} to {goal}...")

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=5.0,
        lambda_edge=0.05,
        n_chains=128,
        warmup=500,
        samples=40,
        steps_per_sample=12
    )

    if path:
        print(f"✓ Found path of length {len(path)}")
        print(f"  Manhattan distance: {abs(start[0] - goal[0]) + abs(start[1] - goal[1])}")
        visualize_solution("demo1_open_maze.png", maze, path, start, goal,
                          title="Open 32x32 Maze")
    else:
        print("✗ No path found")


def demo_sparse_obstacles():
    """Demonstrate solving a maze with sparse random obstacles."""
    print("\n" + "=" * 60)
    print("Demo 2: 48x48 Maze with Sparse Obstacles (15%)")
    print("=" * 60)

    H = W = 48
    rng = np.random.default_rng(42)
    maze = np.zeros((H, W), dtype=np.uint8)

    # Add sparse random obstacles
    mask = (rng.random((H, W)) < 0.15)
    # Keep borders clear
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (2, 2)
    goal = (H - 3, W - 3)
    maze[start] = 0
    maze[goal] = 0

    print(f"Solving {H}x{W} maze with {np.sum(maze)} obstacles...")

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=5.0,
        lambda_edge=0.05,
        n_chains=192,
        warmup=800,
        samples=50,
        steps_per_sample=15
    )

    if path:
        print(f"✓ Found path of length {len(path)}")
        visualize_solution("demo2_sparse_obstacles.png", maze, path, start, goal,
                          title=f"48x48 Maze with {np.sum(maze)} Random Obstacles")
    else:
        print("✗ No path found - try increasing n_chains or warmup")


def demo_small_detailed():
    """Demonstrate on a small maze with detailed visualization."""
    print("\n" + "=" * 60)
    print("Demo 3: Small 12x12 Maze (Detailed View)")
    print("=" * 60)

    H = W = 12
    maze = np.zeros((H, W), dtype=np.uint8)

    # Add a few strategic obstacles
    maze[5, 3:9] = 1  # Horizontal barrier (with gaps)
    maze[5, 5:7] = 0  # Gap in middle

    start = (0, 0)
    goal = (H - 1, W - 1)

    print(f"Solving {H}x{W} maze with a partial barrier...")

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=5.0,
        lambda_edge=0.05,
        n_chains=96,
        warmup=400,
        samples=30,
        steps_per_sample=10
    )

    if path:
        print(f"✓ Found path of length {len(path)}")
        visualize_solution("demo3_small_detailed.png", maze, path, start, goal,
                          title="12x12 Maze with Partial Barrier", show_grid=True)
    else:
        print("✗ No path found")


def demo_large_sparse():
    """Demonstrate on a larger maze."""
    print("\n" + "=" * 60)
    print("Demo 4: Large 64x64 Maze with Sparse Obstacles")
    print("=" * 60)

    H = W = 64
    rng = np.random.default_rng(123)
    maze = np.zeros((H, W), dtype=np.uint8)

    # Add very sparse obstacles
    mask = (rng.random((H, W)) < 0.12)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    print(f"Solving {H}x{W} maze with {np.sum(maze)} obstacles...")
    print("(This may take 10-20 seconds...)")

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=5.0,
        lambda_edge=0.05,
        n_chains=256,
        warmup=1000,
        samples=60,
        steps_per_sample=15
    )

    if path:
        print(f"✓ Found path of length {len(path)}")
        visualize_solution("demo4_large_sparse.png", maze, path, start, goal,
                          title=f"64x64 Maze ({np.sum(maze)} obstacles)")
    else:
        print("✗ No path found")


def visualize_solution(filename, maze, path, start, goal, title="Maze Solution", show_grid=False):
    """Visualize and save a maze solution."""
    H, W = maze.shape

    fig, ax = plt.subplots(figsize=(10, 10))

    # Show maze
    ax.imshow(maze, cmap="gray_r", interpolation="nearest")

    # Plot path
    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7, label='Path')
        ax.plot(xs[0], ys[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(xs[-1], ys[-1], 'ro', markersize=12, label='Goal', zorder=5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.invert_yaxis()

    if show_grid and H <= 20:
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {filename}")
    plt.close()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("THRML Maze Solver Demonstration")
    print("=" * 60)
    print("\nThis script will run 4 demonstrations of the THRML maze solver")
    print("and save visualizations to PNG files.")

    demo_open_maze()
    demo_sparse_obstacles()
    demo_small_detailed()
    demo_large_sparse()

    print("\n" + "=" * 60)
    print("All demos complete! Check the generated PNG files.")
    print("=" * 60)
    print("\nWith distance potentials, the solver now reliably finds paths!")
    print("The energy landscape guides sampling like water flowing downhill.")
    print("\nNote: THRML uses stochastic sampling, so exact paths vary between runs.")
    print("If a demo occasionally fails, try running again or increase n_chains/warmup.")
    print()


if __name__ == "__main__":
    main()

