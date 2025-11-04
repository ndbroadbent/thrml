#!/usr/bin/env python3
"""Create animated visualization of THRML maze solving process.

This script captures intermediate sampling states and creates a video
showing how the solution emerges over time.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from thrml_maze_solver import (
    build_thrml_program, _decode_path, STATE_BITS, DEG, N_STATES
)
from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_states
import jax
import jax.numpy as jnp


def solve_with_animation(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    beta: float = 5.0,
    lambda_edge: float = 0.05,
    hard: float = 1e6,
    n_chains: int = 128,
    warmup: int = 500,
    samples: int = 50,
    steps_per_sample: int = 10,
    output_dir: str = "animation_frames",
):
    """Solve maze and capture intermediate states for animation.

    Returns:
        Tuple of (best_path, all_samples, coords) where all_samples is
        [n_samples, n_nodes] showing evolution over time
    """
    prog, spec, nodes, coords, G = build_thrml_program(
        maze, start, goal, beta, lambda_edge, hard
    )

    # Initialize states
    key = jax.random.key(4242)
    init_state = []
    for block in spec.free_blocks:
        key, sub = jax.random.split(key)
        arr = jax.random.choice(sub, jnp.array([0, 1, 2, 3, 4], dtype=jnp.uint8),
                                shape=(n_chains, len(block.nodes)),
                                p=jnp.array([0.7, 0.075, 0.075, 0.075, 0.075]))

        block_coords = [G.nodes[n]["coord"] for n in block.nodes]
        wall_idx = [k for k, c in enumerate(block_coords) if maze[c] == 1]
        if wall_idx:
            arr = arr.at[:, jnp.array(wall_idx)].set(0)

        start_idx = [k for k, c in enumerate(block_coords) if c == start]
        goal_idx = [k for k, c in enumerate(block_coords) if c == goal]
        if start_idx:
            key, sub = jax.random.split(key)
            start_states = jax.random.choice(sub, jnp.array([1, 2, 3, 4], dtype=jnp.uint8),
                                            shape=(n_chains,))
            arr = arr.at[:, start_idx[0]].set(start_states)
        if goal_idx:
            key, sub = jax.random.split(key)
            goal_states = jax.random.choice(sub, jnp.array([1, 2, 3, 4], dtype=jnp.uint8),
                                           shape=(n_chains,))
            arr = arr.at[:, goal_idx[0]].set(goal_states)

        init_state.append(arr)

    schedule = SamplingSchedule(
        n_warmup=warmup, n_samples=samples, steps_per_sample=steps_per_sample
    )
    out_blocks = [Block(nodes)]
    keys = jax.random.split(key, n_chains)

    # Sample and get all intermediate states
    call = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], out_blocks)))
    stacked = call(init_state, keys)[0]  # [n_chains, n_samples, n_nodes]

    # Convert to numpy
    all_samples = np.array(stacked)  # [n_chains, n_samples, n_nodes]

    # Find best chain (one that produces valid path earliest)
    best_chain_idx = 0
    best_sample_idx = samples - 1
    best_path = None

    for chain_idx in range(n_chains):
        for sample_idx in range(samples):
            state_vec = all_samples[chain_idx, sample_idx, :]
            path = _decode_path(state_vec, coords, start, goal, maze)
            if path is not None:
                best_chain_idx = chain_idx
                best_sample_idx = sample_idx
                best_path = path
                break
        if best_path is not None:
            break

    # Return samples from best chain
    return best_path, all_samples[best_chain_idx], coords, best_sample_idx


def visualize_state(
    maze: np.ndarray,
    state_vec: np.ndarray,
    coords: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    sample_idx: int,
    total_samples: int,
    path: list[tuple[int, int]] | None,
    filename: str,
    dpi: int = 100
):
    """Visualize a single sampling state."""
    H, W = maze.shape
    coord_to_idx = {c: i for i, c in enumerate(coords)}

    # Use figsize that produces even dimensions for video encoding
    # At dpi=100, figsize=(8,8) gives 800x800 pixels (divisible by 2)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create visualization array
    vis = np.zeros((H, W, 3))

    # Walls in black
    vis[maze == 1] = [0, 0, 0]

    # Free cells in white
    vis[maze == 0] = [1, 1, 1]

    # Color cells based on their state
    for i in range(H):
        for j in range(W):
            if maze[i, j] == 0:
                idx = coord_to_idx[(i, j)]
                state = int(state_vec[idx])
                deg = DEG[state]

                if deg > 0:  # Has connections
                    # Color based on degree: blue for active path cells
                    intensity = 0.3 + 0.4 * (deg / 2.0)
                    vis[i, j] = [0.2, 0.4, intensity]

    # Highlight start and goal
    vis[start] = [0, 1, 0]  # Green
    vis[goal] = [1, 0, 0]   # Red

    # If path exists, highlight it
    if path:
        for p in path:
            if p != start and p != goal:
                vis[p] = [1, 1, 0]  # Yellow for path

    ax.imshow(vis, interpolation="nearest")

    # Add title with progress
    title = f"Sample {sample_idx + 1}/{total_samples}"
    if path:
        title += f" - Path Found! (length {len(path)})"
    else:
        title += " - Searching..."

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()


def create_animation(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    output_video: str = "maze_solve_animation.mp4",
    fps: int = 10,
    **solver_params
):
    """Create animated video of maze solving process.

    Args:
        maze: Maze array (1=wall, 0=free)
        start: Starting position
        goal: Goal position
        output_video: Output video filename
        fps: Frames per second for video
        **solver_params: Parameters passed to solver
    """
    print("=" * 60)
    print("Creating Maze Solving Animation")
    print("=" * 60)

    # Create output directory
    output_dir = Path("animation_frames")
    output_dir.mkdir(exist_ok=True)

    # Solve with animation
    print("\n1. Running THRML solver and capturing states...")
    H, W = maze.shape
    print(f"   Maze size: {H}x{W}")
    print(f"   From {start} to {goal}")

    best_path, all_samples, coords, solution_idx = solve_with_animation(
        maze, start, goal, output_dir=str(output_dir), **solver_params
    )

    n_samples = all_samples.shape[0]
    print(f"   Captured {n_samples} samples")
    if best_path:
        print(f"   ✓ Path found at sample {solution_idx + 1} (length {len(best_path)})")
    else:
        print(f"   ✗ No valid path found")

    # Generate frames
    print("\n2. Generating animation frames...")
    frame_files = []

    for sample_idx in range(n_samples):
        state_vec = all_samples[sample_idx]

        # Check if path exists at this point
        if sample_idx <= solution_idx and best_path:
            current_path = _decode_path(state_vec, coords, start, goal, maze)
        else:
            current_path = None

        filename = output_dir / f"frame_{sample_idx:04d}.png"
        frame_files.append(str(filename))

        visualize_state(
            maze, state_vec, coords, start, goal,
            sample_idx, n_samples, current_path, str(filename)
        )

        if (sample_idx + 1) % 10 == 0:
            print(f"   Generated {sample_idx + 1}/{n_samples} frames")

    print(f"   ✓ Generated all {n_samples} frames")

    # Create video with ffmpeg
    print("\n3. Creating video with ffmpeg...")

    # Frame pattern for ffmpeg
    frame_pattern = str(output_dir / "frame_%04d.png")

    # FFmpeg command with padding filter to ensure even dimensions
    cmd = [
        "ffmpeg", "-y",  # Overwrite output
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure dimensions divisible by 2
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",  # Quality (lower = better, 23 is good)
        output_video
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   ✓ Video created: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ FFmpeg error: {e.stderr}")
        return
    except FileNotFoundError:
        print("   ✗ ffmpeg not found. Please install: brew install ffmpeg")
        print("   Frames are saved in animation_frames/ directory")
        return

    # Cleanup frames
    print("\n4. Cleaning up frames...")
    for frame_file in frame_files:
        Path(frame_file).unlink()
    output_dir.rmdir()
    print("   ✓ Cleanup complete")

    print("\n" + "=" * 60)
    print(f"Animation complete! Watch: {output_video}")
    print("=" * 60)


def demo_animation():
    """Demo: Create animation for a simple maze."""
    print("\nCreating animation for 20x20 maze with sparse obstacles...")

    # Create maze - slightly smaller for faster animation generation
    H = W = 20
    rng = np.random.default_rng(42)
    maze = np.zeros((H, W), dtype=np.uint8)

    # Add very sparse obstacles
    mask = (rng.random((H, W)) < 0.08)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    # Create animation with robust parameters
    create_animation(
        maze, start, goal,
        output_video="maze_animation.mp4",
        fps=8,
        beta=5.0,
        lambda_edge=0.05,
        n_chains=256,  # More chains for reliability
        warmup=500,    # Longer warmup
        samples=60,    # More samples for longer animation
        steps_per_sample=10
    )


if __name__ == "__main__":
    demo_animation()

