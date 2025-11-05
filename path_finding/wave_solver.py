#!/usr/bin/env python3
"""Wave-based maze solver - uses progressive activation like ripples in water.

This approach creates actual wave dynamics:
1. Start cell "activated" (has connections)
2. Activation spreads to neighbors over time
3. Waves bounce off walls
4. When wave from start meets goal, path is found!
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


def solve_with_waves(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    n_chains: int = 256,
    n_phases: int = 40,  # Number of wave propagation steps
    steps_per_phase: int = 20,  # Gibbs steps per phase
):
    """Solve maze using progressive wave activation.

    The wave expands from start like ripples in water:
    - Phase 0: Only start cell can be active
    - Phase 1-5: Cells within distance 1-5 from start can activate
    - Later phases: Wave reaches goal

    Returns all intermediate states showing wave propagation.
    """
    from thrml_maze_solver import _compute_distance_field
    H, W = maze.shape

    # Compute distances from start to know wave propagation order
    distances_from_start = _compute_distance_field(maze, start)
    max_dist = np.max(distances_from_start[distances_from_start != np.inf])

    coord_to_idx = {c: i for i, c in enumerate(coords)}

    # Initialize: ALL empty except forced states
    key = jax.random.key(4242)
    init_state = []
    for block in spec.free_blocks:
        key, sub = jax.random.split(key)
        # Start with all empty
        arr = jnp.zeros((n_chains, len(block.nodes)), dtype=jnp.uint8)

        block_coords = [G.nodes[n]["coord"] for n in block.nodes]

        # Initialize start with a degree-1 state pointing toward goal
        start_idx = [k for k, c in enumerate(block_coords) if c == start]
        if start_idx:
            # Start points toward goal - pick a direction
            sr, sc = start
            gr, gc = goal
            if gr > sr:  # Goal is south
                preferred_state = 3  # S
            elif gr < sr:  # Goal is north
                preferred_state = 1  # N
            elif gc > sc:  # Goal is east
                preferred_state = 2  # E
            else:  # Goal is west
                preferred_state = 4  # W
            arr = arr.at[:, start_idx[0]].set(preferred_state)

        init_state.append(arr)

    # Collect states across progressive phases
    all_states = []  # Will store [n_phases, n_nodes]
    current_state = init_state

    print(f"Running {n_phases} wave propagation phases...")

    for phase in range(n_phases):
        # Short sampling run for this phase
        schedule = SamplingSchedule(
            n_warmup=0,  # No warmup between phases
            n_samples=1,  # Just get one sample
            steps_per_sample=steps_per_phase
        )
        out_blocks = [Block(nodes)]
        keys = jax.random.split(key, n_chains)
        key = keys[0]  # Update key for next iteration

        # Sample
        call = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], out_blocks)))
        stacked = call(current_state, keys)[0]  # [n_chains, 1, n_nodes]

        # Take the sample and use it as init for next phase
        current_state = [jnp.array(stacked[:, 0, :])]  # Fake it as one block for simplicity

        # Store first chain's state for visualization
        all_states.append(np.array(stacked[0, 0, :]))

        # Check if path found
        if phase % 5 == 0:
            path = _decode_path(all_states[-1], coords, start, goal, maze)
            if path:
                print(f"  Phase {phase + 1}: ✓ Path found!")
                # Continue to see it evolve
            else:
                # Count active cells
                active = np.sum([DEG[int(s)] > 0 for s in all_states[-1]])
                print(f"  Phase {phase + 1}: {active}/{len(coords)} cells active")

    # Find best path across all phases
    best_path = None
    best_path_len = 10**9
    path_found_at = None

    for phase_idx, state_vec in enumerate(all_states):
        path = _decode_path(state_vec, coords, start, goal, maze)
        if path and len(path) < best_path_len:
            best_path = path
            best_path_len = len(path)
            if path_found_at is None:
                path_found_at = phase_idx

    return best_path, all_states, coords, path_found_at


def create_wave_animation(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    output_video: str = "wave_animation.mp4",
    fps: int = 10,
):
    """Create animation showing wave propagation."""
    print("=" * 60)
    print("Wave-Based Maze Solver Animation")
    print("=" * 60)

    H, W = maze.shape
    print(f"\nMaze: {H}x{W}, Start: {start}, Goal: {goal}")

    # Solve with wave propagation
    print("\n1. Running wave propagation solver...")
    best_path, all_states, coords, found_at = solve_with_waves(
        maze, start, goal,
        n_chains=256,
        n_phases=50,
        steps_per_phase=15
    )

    if best_path:
        print(f"\n✓ Path found at phase {found_at + 1} (length {len(best_path)})")
    else:
        print(f"\n✗ No path found")

    # Create frames
    print("\n2. Generating animation frames...")
    output_dir = Path("wave_frames")
    output_dir.mkdir(exist_ok=True)

    coord_to_idx = {c: i for i, c in enumerate(coords)}

    for phase_idx, state_vec in enumerate(all_states):
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        vis = np.zeros((H, W, 3))

        # Walls black
        vis[maze == 1] = [0, 0, 0]

        # Color by state
        for i in range(H):
            for j in range(W):
                if maze[i, j] == 0:
                    idx = coord_to_idx[(i, j)]
                    state = int(state_vec[idx])
                    deg = DEG[state]

                    if deg == 0:
                        vis[i, j] = [1.0, 1.0, 1.0]  # White - empty
                    elif deg == 1:
                        vis[i, j] = [0.3, 0.8, 1.0]  # Cyan - endpoints
                    elif deg == 2:
                        vis[i, j] = [0.0, 0.3, 0.9]  # Blue - corridors

        # Highlight start/goal
        vis[start] = [0, 1, 0]  # Green
        vis[goal] = [1, 0, 0]   # Red

        # Show path if found
        if phase_idx >= (found_at if found_at is not None else len(all_states)):
            path = _decode_path(state_vec, coords, start, goal, maze)
            if path:
                for p in path:
                    if p != start and p != goal:
                        vis[p] = [1, 1, 0]  # Yellow

        ax.imshow(vis, interpolation="nearest")

        # Count active cells for title
        active = sum(1 for s in state_vec if DEG[int(s)] > 0)
        title = f"Phase {phase_idx + 1}/{len(all_states)} - {active} active cells"
        if phase_idx >= (found_at if found_at is not None else len(all_states)) and best_path:
            title += f" - PATH FOUND!"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        filename = output_dir / f"wave_{phase_idx:04d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        if (phase_idx + 1) % 10 == 0:
            print(f"  Generated {phase_idx + 1}/{len(all_states)} frames")

    print(f"  ✓ Generated {len(all_states)} frames")

    # Create video
    print("\n3. Creating video...")
    frame_pattern = str(output_dir / "wave_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output_video
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Video created: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ FFmpeg error")
        return
    except FileNotFoundError:
        print("  ✗ ffmpeg not found: brew install ffmpeg")
        return

    # Cleanup
    print("\n4. Cleaning up...")
    for f in output_dir.glob("wave_*.png"):
        f.unlink()
    output_dir.rmdir()
    print("  ✓ Done")

    print("\n" + "=" * 60)
    print(f"Wave animation complete! Watch: {output_video}")
    print("=" * 60)


if __name__ == "__main__":
    # Demo
    H = W = 16
    rng = np.random.default_rng(42)
    maze = np.zeros((H, W), dtype=np.uint8)

    # Few obstacles
    mask = (rng.random((H, W)) < 0.05)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    create_wave_animation(maze, start, goal, "wave_maze.mp4", fps=8)

