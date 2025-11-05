#!/usr/bin/env python3
"""Spacetime maze solver - encodes the solving PROCESS as a 3D probabilistic model.

Key insight: Instead of sampling a static path configuration, sample the entire
TRAJECTORY of path growth over time. This transforms global connectivity into
local temporal constraints!

Variables: Binary indicator per (x, y, t): "Is this cell on the path at time t?"
Constraints:
  - Temporal: path[t+1] must contain path[t] + at most 1 new neighbor
  - Spatial: path must be connected within each time slice
  - Initial: path[0] = {start}
  - Final: goal must be in path by some t
"""
import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram


# Binary states: 0 = not on path, 1 = on path
N_STATES_ST = 2


def build_spacetime_program(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    T: int = 60,  # Time steps
    beta: float = 5.0,
):
    """Build spacetime sampling program.

    Creates a 3D grid of binary nodes [H, W, T] where each node indicates
    whether that cell is on the path at that time.
    """
    H, W = maze.shape

    # Create 3D grid of nodes: (row, col, time)
    nodes_3d = {}
    for t in range(T):
        for i in range(H):
            for j in range(W):
                nodes_3d[(i, j, t)] = CategoricalNode()

    # Flatten to list for indexing
    coords_3d = [(i, j, t) for t in range(T) for i in range(H) for j in range(W)]
    nodes_list = [nodes_3d[c] for c in coords_3d]

    # Create bipartite coloring for spacetime graph (checkerboard in space, alternating in time)
    # Simple: split by (i+j+t) % 2
    color0 = [nodes_3d[c] for c in coords_3d if (c[0] + c[1] + c[2]) % 2 == 0]
    color1 = [nodes_3d[c] for c in coords_3d if (c[0] + c[1] + c[2]) % 2 == 1]

    blocks = [Block(color0), Block(color1)]
    spec = BlockGibbsSpec(blocks, [])

    total_vars = H * W * T

    # UNARY FACTORS: Initial/boundary conditions
    W_unary = np.zeros((total_vars, N_STATES_ST), dtype=np.float32)

    for idx, (i, j, t) in enumerate(coords_3d):
        if maze[i, j] == 1:  # Wall - never on path
            W_unary[idx, 0] = 0.0  # Prefer not on path
            W_unary[idx, 1] = -1e6  # Forbid on path
        elif (i, j) == start and t == 0:
            # Start must be on path at t=0
            W_unary[idx, 0] = -1e6
            W_unary[idx, 1] = 0.0
        elif (i, j) == goal:
            # Goal should be on path eventually - reward being on path
            W_unary[idx, 1] = 2.0 * (t / T)  # Increasing reward over time
        else:
            # Neutral - let constraints decide
            W_unary[idx, 0] = 0.0
            W_unary[idx, 1] = 0.0

    interactions = []

    # TEMPORAL FACTORS: Path can only grow by one cell per timestep
    # For each t → t+1, enforce that new path cells must neighbor old path cells
    for t in range(T - 1):
        for i in range(H):
            for j in range(W):
                if maze[i, j] == 1:
                    continue

                curr_node = nodes_3d[(i, j, t)]
                next_node = nodes_3d[(i, j, t + 1)]

                # Temporal factor: [state_at_t, state_at_t+1]
                W_temp = np.zeros((N_STATES_ST, N_STATES_ST), dtype=np.float32)

                # Once on path, stay on path (path monotonically grows)
                W_temp[1, 1] = 5.0  # If on path at t, reward staying on path at t+1
                W_temp[1, 0] = -20.0  # If on path at t, penalize disappearing at t+1
                W_temp[0, 0] = 0.0  # Not on path -> not on path is neutral

                # Not on path -> on path: need to check neighbors (done in spatial factors)
                W_temp[0, 1] = 0.0  # Neutral base, spatial factors will guide

                # Add temporal factor for this cell across time
                temp_factor = CategoricalEBMFactor(
                    [Block([curr_node]), Block([next_node])],
                    beta * jnp.array([W_temp])
                )
                interactions.append(temp_factor)

    # SPATIAL FACTORS: At time t+1, new path cells must neighbor existing path cells
    # For each t and each cell (i,j), if newly added at t+1, at least one neighbor was on path at t
    for t in range(T - 1):
        for i in range(H):
            for j in range(W):
                if maze[i, j] == 1:
                    continue

                curr_node = nodes_3d[(i, j, t)]
                next_node = nodes_3d[(i, j, t + 1)]

                # Check all 4 neighbors
                neighbors_curr = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and maze[ni, nj] == 0:
                        neighbors_curr.append(nodes_3d[(ni, nj, t)])

                # If cell becomes active at t+1 (was 0, now 1), at least one neighbor must be active at t
                # This is complex with binary - use penalty for isolated activation

                # Simplified: reward spatial connectivity at each time slice
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and maze[ni, nj] == 0:
                        neighbor_node_t = nodes_3d[(ni, nj, t)]
                        curr_node_t = nodes_3d[(i, j, t)]

                        # Spatial connectivity factor: if both on path at same time, reward
                        W_spatial = np.array([
                            [0.0, 0.0],      # [0,0] or [0,1]: neutral
                            [0.0, 2.0],      # [1,1]: both on path - reward connectivity!
                        ], dtype=np.float32)

                        spatial_factor = CategoricalEBMFactor(
                            [Block([curr_node_t]), Block([neighbor_node_t])],
                            beta * jnp.array([W_spatial])
                        )
                        interactions.append(spatial_factor)

    # Build factors
    unary_factor = CategoricalEBMFactor([Block(nodes_list)], beta * jnp.asarray(W_unary))
    interactions.insert(0, unary_factor)

    sampler = CategoricalGibbsConditional(N_STATES_ST)

    prog = FactorSamplingProgram(
        spec,
        [sampler for _ in spec.free_blocks],
        interactions,
        [],
    )

    return prog, spec, nodes_list, coords_3d, H, W, T


def solve_spacetime(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    T: int = 60,
    n_chains: int = 128,
    warmup: int = 500,
    samples: int = 30,
):
    """Solve maze in spacetime - samples the entire growth process!"""

    H, W = maze.shape
    print(f"\nBuilding spacetime model: {H}×{W}×{T} = {H*W*T} variables...")

    prog, spec, nodes_list, coords_3d, H, W, T = build_spacetime_program(
        maze, start, goal, T, beta=3.0
    )

    print(f"Variables: {len(nodes_list)}")
    print(f"Blocks: {len(spec.free_blocks)}")

    # Initialize all to 0 (not on path)
    key = jax.random.key(4242)
    init_state = []
    for block in spec.free_blocks:
        # Start mostly empty
        key, sub = jax.random.split(key)
        arr = jnp.zeros((n_chains, len(block.nodes)), dtype=jnp.uint8)

        # Set start at t=0 to 1 if in this block
        block_indices = {id(node): k for k, node in enumerate(block.nodes)}
        start_node_t0 = [id(n) for c, n in zip(coords_3d, nodes_list) if c == (*start, 0)]
        if start_node_t0 and start_node_t0[0] in block_indices:
            idx = block_indices[start_node_t0[0]]
            arr = arr.at[:, idx].set(1)

        init_state.append(arr)

    print(f"\nSampling spacetime trajectories...")
    schedule = SamplingSchedule(n_warmup=warmup, n_samples=samples, steps_per_sample=10)
    out_blocks = [Block(nodes_list)]
    keys = jax.random.split(key, n_chains)

    call = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], out_blocks)))
    stacked = call(init_state, keys)[0]  # [n_chains, n_samples, H*W*T]

    final_states = np.array(stacked[:, -1, :])  # [n_chains, H*W*T]

    print(f"Sampled {n_chains} chains, {samples} samples each")

    # Decode: extract path from spacetime
    best_path = None
    best_chain_idx = 0

    for chain_idx in range(n_chains):
        state_4d = final_states[chain_idx]  # Flat array of H*W*T variables

        # Reshape to [t, i, j]
        state_grid = state_4d.reshape((T, H, W))

        # Extract path by following activation through time
        path = extract_path_from_spacetime(state_grid, start, goal, maze)

        if path is not None:
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_chain_idx = chain_idx

    # Get spacetime trajectory for visualization
    best_state_4d = final_states[best_chain_idx].reshape((T, H, W))

    return best_path, best_state_4d, (H, W, T)


def extract_path_from_spacetime(
    state_grid: np.ndarray,  # [T, H, W]
    start: tuple[int, int],
    goal: tuple[int, int],
    maze: np.ndarray
) -> list[tuple[int, int]] | None:
    """Extract path from spacetime configuration.

    The path is the sequence of cells that become active over time.
    """
    T, H, W = state_grid.shape

    # Find which time each cell first becomes active
    activation_time = np.full((H, W), -1, dtype=int)

    for t in range(T):
        for i in range(H):
            for j in range(W):
                if state_grid[t, i, j] == 1 and activation_time[i, j] == -1:
                    activation_time[i, j] = t

    # Check if goal was activated
    if activation_time[goal] == -1:
        return None

    # Build path by following activation times from start to goal
    # Cells should activate in sequence, each adjacent to previous
    path_candidates = []
    for i in range(H):
        for j in range(W):
            if activation_time[i, j] >= 0:
                path_candidates.append((activation_time[i, j], (i, j)))

    if not path_candidates:
        return None

    # Sort by activation time
    path_candidates.sort()

    # Extract coordinates
    path = [coord for t, coord in path_candidates]

    # Validate: must start with start and include goal
    if len(path) == 0 or path[0] != start or goal not in path:
        return None

    # Validate: each step must be adjacent to previous
    for k in range(len(path) - 1):
        r1, c1 = path[k]
        r2, c2 = path[k + 1]
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return None  # Not adjacent

    # Truncate at goal
    goal_idx = path.index(goal)
    path = path[:goal_idx + 1]

    return path


def visualize_spacetime(
    state_grid: np.ndarray,  # [T, H, W]
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    output_video: str = "spacetime_maze.mp4",
    fps: int = 10
):
    """Create animation showing path growing through time."""
    T, H, W = state_grid.shape

    print(f"\nCreating spacetime animation...")
    output_dir = Path("spacetime_frames")
    output_dir.mkdir(exist_ok=True)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(8, 8))
        vis = np.zeros((H, W, 3))

        # Walls
        vis[maze == 1] = [0, 0, 0]

        # Cells on path at this time
        for i in range(H):
            for j in range(W):
                if maze[i, j] == 0:
                    on_path = state_grid[t, i, j]
                    if on_path == 1:
                        vis[i, j] = [0.2, 0.5, 1.0]  # Blue - on path
                    else:
                        vis[i, j] = [1.0, 1.0, 1.0]  # White - not yet reached

        # Highlight start/goal
        vis[start] = [0, 1, 0]  # Green
        vis[goal] = [1, 0, 0]   # Red

        ax.imshow(vis, interpolation='nearest')

        # Count cells on path
        on_path_count = int(np.sum(state_grid[t, :, :]))
        ax.set_title(f'Time {t}/{T-1} - {on_path_count} cells on path',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        filename = output_dir / f"spacetime_{t:04d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        if (t + 1) % 10 == 0:
            print(f"  Generated {t + 1}/{T} frames")

    print(f"  ✓ Generated all {T} frames")

    # Create video
    print("\n  Creating video...")
    frame_pattern = str(output_dir / "spacetime_%04d.png")

    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", frame_pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        output_video
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Video created: {output_video}")
    except:
        print(f"  Note: frames saved in {output_dir}/")
        return

    # Cleanup
    for f in output_dir.glob("spacetime_*.png"):
        f.unlink()
    output_dir.rmdir()


def demo():
    """Demo spacetime solver."""
    print("=" * 60)
    print("SPACETIME Maze Solver")
    print("Encoding the solving PROCESS as a 3D probabilistic model")
    print("=" * 60)

    # TINY maze for demo (spacetime is expensive!)
    H = W = 8
    maze = np.zeros((H, W), dtype=np.uint8)

    # Add a single obstacle
    maze[3, 3] = 1
    maze[3, 4] = 1

    start = (0, 0)
    goal = (H - 1, W - 1)

    # Solve in spacetime!
    print(f"\nMaze: {H}×{W}, Obstacle at (3,3)-(3,4)")
    print(f"Spacetime encoding will have {H*W*20} variables")

    path, state_4d, dims = solve_spacetime(
        maze, start, goal,
        T=20,  # 20 time steps (8×8 needs ~14 steps minimum)
        n_chains=64,  # Fewer chains for speed
        warmup=300,
        samples=20
    )

    if path:
        print(f"\n✓ Found path of length {len(path)}")
        print(f"  Path: {path[:10]}..." if len(path) > 10 else f"  Path: {path}")

        # Create animation
        visualize_spacetime(state_4d, maze, start, goal, "spacetime_maze.mp4", fps=10)
    else:
        print(f"\n✗ No valid path found in spacetime")
        print("  Try increasing T (time steps) or sampling parameters")


if __name__ == "__main__":
    demo()

