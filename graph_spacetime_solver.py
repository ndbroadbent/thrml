#!/usr/bin/env python3
"""Optimized spacetime solver using pure graph representation.

Key optimization: Only encode FREE cells and their valid edges.
Walls and impossible moves are removed during preprocessing.

Variables: Binary per (free_cell, time): "Is this cell on path at time t?"
Much smaller problem space!
"""
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram


N_STATES_BINARY = 2  # 0 = not on path, 1 = on path


def preprocess_maze_to_graph(maze: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """Compile maze into pure graph representation.

    Returns:
        - free_cells: List of (row, col) for non-wall cells
        - adjacency: Dict mapping cell_idx -> list of neighbor indices
        - start_idx, goal_idx: indices in free_cells list
    """
    H, W = maze.shape

    # Extract free cells
    free_cells = []
    coord_to_idx = {}

    for i in range(H):
        for j in range(W):
            if maze[i, j] == 0:  # Free cell
                idx = len(free_cells)
                free_cells.append((i, j))
                coord_to_idx[(i, j)] = idx

    # Build adjacency (only valid neighbors)
    adjacency = {i: [] for i in range(len(free_cells))}

    for idx, (i, j) in enumerate(free_cells):
        # Check 4 neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in coord_to_idx:
                neighbor_idx = coord_to_idx[(ni, nj)]
                adjacency[idx].append(neighbor_idx)

    start_idx = coord_to_idx[start]
    goal_idx = coord_to_idx[goal]

    print(f"  Compiled: {len(free_cells)} free cells (from {H*W} total)")
    print(f"  Average degree: {np.mean([len(v) for v in adjacency.values()]):.1f}")

    return free_cells, adjacency, start_idx, goal_idx


def build_spacetime_graph_program(
    free_cells: list[tuple[int, int]],
    adjacency: dict[int, list[int]],
    start_idx: int,
    goal_idx: int,
    T: int = 30,
    beta: float = 5.0,
):
    """Build spacetime model on PURE GRAPH (no redundant wall cells).

    Creates binary variables [cell_idx, time] indicating path membership.
    """
    N = len(free_cells)  # Number of free cells

    # Create spacetime nodes: (cell_idx, time)
    nodes_st = {}
    for t in range(T):
        for cell_idx in range(N):
            nodes_st[(cell_idx, t)] = CategoricalNode()

    # Flatten
    coords_st = [(cell_idx, t) for t in range(T) for cell_idx in range(N)]
    nodes_list = [nodes_st[c] for c in coords_st]

    # Bipartite coloring
    color0 = [nodes_st[c] for c in coords_st if (c[0] + c[1]) % 2 == 0]
    color1 = [nodes_st[c] for c in coords_st if (c[0] + c[1]) % 2 == 1]

    blocks = [Block(color0), Block(color1)]
    spec = BlockGibbsSpec(blocks, [])

    total_vars = N * T

    print(f"  Spacetime variables: {total_vars} (vs {len(free_cells)*T} grid would be worse)")

    # UNARY FACTORS
    W_unary = np.zeros((total_vars, N_STATES_BINARY), dtype=np.float32)

    for idx, (cell_idx, t) in enumerate(coords_st):
        if cell_idx == start_idx and t == 0:
            # Start MUST be on path at t=0
            W_unary[idx, 0] = -100.0  # Strong penalty for not on path
            W_unary[idx, 1] = 0.0
        elif cell_idx == goal_idx:
            # Goal should be on path - reward increases over time
            W_unary[idx, 1] = 3.0 * (t / T)  # Increasing reward
        else:
            # Neutral
            W_unary[idx, 0] = 0.0
            W_unary[idx, 1] = 0.0

    interactions = []

    # TEMPORAL FACTORS: Path grows monotonically (once on, stay on)
    for t in range(T - 1):
        for cell_idx in range(N):
            curr_node = nodes_st[(cell_idx, t)]
            next_node = nodes_st[(cell_idx, t + 1)]

            # [state_t, state_t+1]
            W_temp = np.array([
                [0.0, 0.0],      # [0,0]: not on path -> not on path (neutral)
                [-50.0, 5.0],    # [1,0]: penalty for leaving path, [1,1]: reward staying
            ], dtype=np.float32)

            temp_factor = CategoricalEBMFactor(
                [Block([curr_node]), Block([next_node])],
                beta * jnp.array([W_temp])
            )
            interactions.append(temp_factor)

    # SPATIAL FACTORS: Cells on path at time t must be connected
    # AND growth factor: if newly added at t+1, must neighbor something from t
    for t in range(T):
        for cell_idx in range(N):
            cell_node = nodes_st[(cell_idx, t)]

            # Spatial connectivity at time t
            for neighbor_idx in adjacency[cell_idx]:
                neighbor_node = nodes_st[(neighbor_idx, t)]

                # Both on path = reward (encourages connected paths)
                W_spatial = np.array([
                    [0.0, 0.0],    # [0,0] or [0,1]: neutral
                    [0.0, 2.0],    # [1,1]: both on path - good!
                ], dtype=np.float32)

                spatial_factor = CategoricalEBMFactor(
                    [Block([cell_node]), Block([neighbor_node])],
                    beta * jnp.array([W_spatial])
                )
                interactions.append(spatial_factor)

    # GROWTH CONSTRAINT: If cell becomes active at t+1 (wasn't active at t),
    # at least one neighbor must have been active at t
    for t in range(T - 1):
        for cell_idx in range(N):
            curr_cell = nodes_st[(cell_idx, t)]
            next_cell = nodes_st[(cell_idx, t + 1)]

            # For each neighbor, create factor:
            # If this cell goes from 0→1, neighbor at t should be 1
            for neighbor_idx in adjacency[cell_idx]:
                neighbor_curr = nodes_st[(neighbor_idx, t)]

                # This is tricky in binary - we want to encourage growth from neighbors
                # Simple version: if cell activates at t+1 and neighbor active at t, reward

                # We need 3-way factor: [curr_cell_t, next_cell_t+1, neighbor_t]
                # THRML supports this! Use CategoricalEBMFactor with 3 blocks

                # [curr, next, neighbor] states
                # We want: if curr=0, next=1 (newly added), then reward if neighbor=1
                W_growth = np.zeros((N_STATES_BINARY, N_STATES_BINARY, N_STATES_BINARY), dtype=np.float32)

                # [curr=0, next=1, neighbor=1]: newly added and neighbor was on path → reward
                W_growth[0, 1, 1] = 4.0

                # [curr=0, next=1, neighbor=0]: newly added but no neighbor → penalty
                W_growth[0, 1, 0] = -3.0

                # All other combinations neutral or already handled by other factors

                growth_factor = CategoricalEBMFactor(
                    [Block([curr_cell]), Block([next_cell]), Block([neighbor_curr])],
                    beta * jnp.array([W_growth])
                )
                interactions.append(growth_factor)

    # Unary factor
    unary_factor = CategoricalEBMFactor([Block(nodes_list)], beta * jnp.asarray(W_unary))
    interactions.insert(0, unary_factor)

    sampler = CategoricalGibbsConditional(N_STATES_BINARY)

    prog = FactorSamplingProgram(
        spec,
        [sampler for _ in spec.free_blocks],
        interactions,
        [],
    )

    return prog, spec, nodes_list, coords_st, N, T


def solve_graph_spacetime(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    T: int = 30,
    n_chains: int = 128,
    warmup: int = 500,
    samples: int = 30,
):
    """Solve using optimized graph spacetime encoding."""

    # STEP 1: Compile maze to pure graph
    print("\n1. Compiling maze to graph...")
    free_cells, adjacency, start_idx, goal_idx = preprocess_maze_to_graph(maze, start, goal)

    # STEP 2: Build spacetime model on graph
    print("\n2. Building spacetime model...")
    prog, spec, nodes_list, coords_st, N, T = build_spacetime_graph_program(
        free_cells, adjacency, start_idx, goal_idx, T, beta=4.0
    )

    print(f"  Total variables: {N*T}")

    # STEP 3: Initialize
    print("\n3. Initializing...")
    key = jax.random.key(4242)
    init_state = []
    for block in spec.free_blocks:
        # Start with all zeros (not on path)
        arr = jnp.zeros((n_chains, len(block.nodes)), dtype=jnp.uint8)

        # Find start at t=0 and set to 1
        block_node_ids = {id(node): k for k, node in enumerate(block.nodes)}
        start_t0_nodes = [nodes_list[i] for i, c in enumerate(coords_st) if c == (start_idx, 0)]

        if start_t0_nodes:
            node_id = id(start_t0_nodes[0])
            if node_id in block_node_ids:
                idx = block_node_ids[node_id]
                arr = arr.at[:, idx].set(1)

        init_state.append(arr)

    # STEP 4: Sample
    print(f"\n4. Sampling {n_chains} chains...")
    schedule = SamplingSchedule(n_warmup=warmup, n_samples=samples, steps_per_sample=15)
    out_blocks = [Block(nodes_list)]
    keys = jax.random.split(key, n_chains)

    call = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], out_blocks)))
    stacked = call(init_state, keys)[0]  # [n_chains, n_samples, N*T]

    final_states = np.array(stacked[:, -1, :])  # [n_chains, N*T]
    all_samples = np.array(stacked[0, :, :])  # [n_samples, N*T] from first chain

    print(f"  Done! Decoding results...")

    # STEP 5: Decode path from spacetime
    best_path = None
    for chain_idx in range(n_chains):
        state_flat = final_states[chain_idx]
        state_2d = state_flat.reshape((T, N))  # [time, cell]

        path = extract_path_from_graph_spacetime(state_2d, free_cells, start_idx, goal_idx, adjacency)

        if path is not None:
            if best_path is None or len(path) < len(best_path):
                best_path = path

    # Also return samples for animation
    samples_2d = all_samples.reshape((samples, T, N))

    return best_path, samples_2d, free_cells, start_idx, goal_idx


def extract_path_from_graph_spacetime(
    state_2d: np.ndarray,  # [T, N]
    free_cells: list[tuple[int, int]],
    start_idx: int,
    goal_idx: int,
    adjacency: dict[int, list[int]]
) -> list[tuple[int, int]] | None:
    """Extract path from graph spacetime configuration."""
    T, N = state_2d.shape

    # Find when each cell first becomes active
    activation_time = np.full(N, -1, dtype=int)

    for t in range(T):
        for cell_idx in range(N):
            if state_2d[t, cell_idx] == 1 and activation_time[cell_idx] == -1:
                activation_time[cell_idx] = t

    # Goal must be activated
    if activation_time[goal_idx] == -1:
        return None

    # Build path by activation order
    activated = [(activation_time[i], i) for i in range(N) if activation_time[i] >= 0]
    activated.sort()

    path_indices = [cell_idx for t, cell_idx in activated]

    # Validate
    if not path_indices or path_indices[0] != start_idx or goal_idx not in path_indices:
        return None

    # Check adjacency
    for k in range(len(path_indices) - 1):
        curr = path_indices[k]
        nxt = path_indices[k + 1]
        if nxt not in adjacency[curr]:
            return None  # Not adjacent

    # Truncate at goal
    goal_pos = path_indices.index(goal_idx)
    path_indices = path_indices[:goal_pos + 1]

    # Convert back to coordinates
    path = [free_cells[idx] for idx in path_indices]

    return path


def visualize_graph_spacetime(
    samples_2d: np.ndarray,  # [n_samples, T, N]
    maze: np.ndarray,
    free_cells: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    start_idx: int,
    goal_idx: int,
    output_video: str = "graph_spacetime.mp4",
    fps: int = 10
):
    """Create animation showing wave propagation in graph spacetime."""
    n_samples, T, N = samples_2d.shape
    H, W = maze.shape

    print(f"\nCreating animation from {n_samples} samples...")
    output_dir = Path("graph_st_frames")
    output_dir.mkdir(exist_ok=True)

    # For each sample (shows evolution as sampling progresses)
    for sample_idx in range(n_samples):
        # Take final time slice to show equilibrium at this sample
        # OR show animation through time for this sample
        # Let's show time evolution for the LAST sample

        if sample_idx == n_samples - 1:
            # Last sample: show all time slices
            for t in range(T):
                visualize_timeslice(
                    samples_2d[sample_idx, t, :], maze, free_cells,
                    start, goal, t, T, output_dir, sample_idx * T + t
                )
        else:
            # Earlier samples: just show final time
            visualize_timeslice(
                samples_2d[sample_idx, T-1, :], maze, free_cells,
                start, goal, T-1, T, output_dir, sample_idx
            )

        if (sample_idx + 1) % 5 == 0:
            print(f"  Processed sample {sample_idx + 1}/{n_samples}")

    total_frames = (n_samples - 1) + T
    print(f"  ✓ Generated {total_frames} frames")

    # Create video
    print("\n  Creating video...")
    frame_pattern = str(output_dir / "frame_%04d.png")

    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", frame_pattern,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        output_video
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Video: {output_video}")
    except:
        print(f"  Frames saved in {output_dir}/")
        return

    # Cleanup
    for f in output_dir.glob("frame_*.png"):
        f.unlink()
    output_dir.rmdir()


def visualize_timeslice(
    state_1d: np.ndarray,  # [N] - state of each free cell
    maze: np.ndarray,
    free_cells: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    t: int,
    T: int,
    output_dir: Path,
    frame_num: int
):
    """Visualize one time slice."""
    H, W = maze.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    vis = np.zeros((H, W, 3))

    # Walls black
    vis[maze == 1] = [0, 0, 0]

    # White background
    vis[maze == 0] = [1.0, 1.0, 1.0]

    # Color cells on path
    for cell_idx, (i, j) in enumerate(free_cells):
        if state_1d[cell_idx] == 1:
            vis[i, j] = [0.2, 0.5, 1.0]  # Blue - on path

    # Highlight start/goal
    vis[start] = [0, 1, 0]
    vis[goal] = [1, 0, 0]

    ax.imshow(vis, interpolation='nearest')

    on_path_count = int(np.sum(state_1d))
    ax.set_title(f'Time {t}/{T-1} - {on_path_count} cells on path',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    filename = output_dir / f"frame_{frame_num:04d}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def demo():
    """Demo optimized graph spacetime solver."""
    print("=" * 70)
    print("OPTIMIZED GRAPH SPACETIME MAZE SOLVER")
    print("Your brilliant idea: Encode time itself into the model!")
    print("=" * 70)

    # 10x10 maze
    H = W = 10
    rng = np.random.default_rng(42)
    maze = np.zeros((H, W), dtype=np.uint8)

    # Few obstacles
    mask = (rng.random((H, W)) < 0.15)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    # Solve!
    path, samples_2d, free_cells, start_idx, goal_idx = solve_graph_spacetime(
        maze, start, goal,
        T=25,  # Time steps
        n_chains=256,
        warmup=1000,
        samples=30
    )

    if path:
        print(f"\n✓ SPACETIME SOLVER FOUND PATH!")
        print(f"  Length: {len(path)}")
        print(f"  Path: {path}")

        # Create animation
        visualize_graph_spacetime(
            samples_2d, maze, free_cells, start, goal,
            start_idx, goal_idx,
            "graph_spacetime.mp4", fps=8
        )

        print("\n" + "=" * 70)
        print("This works because encoding TIME makes global→local!")
        print("Path connectivity is now just temporal consistency! 🌊")
        print("=" * 70)
    else:
        print(f"\n✗ No path found - try increasing T or sampling params")


if __name__ == "__main__":
    demo()

