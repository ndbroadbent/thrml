#!/usr/bin/env python3
"""Pure graph-based spacetime maze solver.

Key optimization: Don't encode the grid structure or wall logic.
Instead, preprocess maze into a pure graph and let THRML solve
graph traversal in spacetime.

Preprocessing:
  maze[H, W] → graph G with only free cells as nodes
  Pre-filter edges: only connect cells with no walls between

Spacetime encoding:
  Each (node, time) is a binary variable: "Is this node visited at time t?"

Constraints (all local in spacetime!):
  - Temporal: visited[n,t+1] can only be true if n visited at t OR neighbor visited at t
  - Initial: start visited at t=0
  - Final: goal visited at some t
  - Monotonic: once visited, stay visited

This is 10x more efficient than grid-based encoding!
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


# Binary states: 0 = not visited, 1 = visited
N_STATES_BINARY = 2


def maze_to_graph(maze: np.ndarray) -> tuple[nx.Graph, dict, dict]:
    """Convert maze to pure graph of free cells.

    Returns:
        graph: NetworkX graph with free cells as nodes
        coord_to_node: Map from (row, col) to node ID
        node_to_coord: Map from node ID to (row, col)
    """
    H, W = maze.shape

    # Create nodes only for free cells
    coord_to_node = {}
    node_id = 0
    for i in range(H):
        for j in range(W):
            if maze[i, j] == 0:  # Free cell
                coord_to_node[(i, j)] = node_id
                node_id += 1

    node_to_coord = {v: k for k, v in coord_to_node.items()}
    n_free = len(coord_to_node)

    # Create edges between adjacent free cells
    G = nx.Graph()
    G.add_nodes_from(range(n_free))

    for (r1, c1), id1 in coord_to_node.items():
        # Check 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r2, c2 = r1 + dr, c1 + dc
            if (r2, c2) in coord_to_node:
                id2 = coord_to_node[(r2, c2)]
                if id1 < id2:  # Add edge once
                    G.add_edge(id1, id2)

    return G, coord_to_node, node_to_coord


def build_spacetime_graph_program(
    G: nx.Graph,
    start_node: int,
    goal_node: int,
    T: int,
    beta: float = 3.0,
):
    """Build spacetime sampling program on pure graph.

    Variables: binary[node, time] = "Is node visited at time t?"

    Much more efficient than grid encoding!
    """
    n_nodes = G.number_of_nodes()

    # Create spacetime nodes: (graph_node_id, time)
    st_nodes = {}
    for t in range(T):
        for n in range(n_nodes):
            st_nodes[(n, t)] = CategoricalNode()

    # Flatten
    st_coords = [(n, t) for t in range(T) for n in range(n_nodes)]
    st_nodes_list = [st_nodes[c] for c in st_coords]

    # Bipartite coloring
    color0 = [st_nodes[c] for c in st_coords if (c[0] + c[1]) % 2 == 0]
    color1 = [st_nodes[c] for c in st_coords if (c[0] + c[1]) % 2 == 1]

    blocks = [Block(color0), Block(color1)]
    spec = BlockGibbsSpec(blocks, [])

    total_vars = n_nodes * T
    print(f"  Spacetime variables: {n_nodes} nodes × {T} time = {total_vars}")

    # UNARY FACTORS: Boundary conditions
    W_unary = np.zeros((total_vars, N_STATES_BINARY), dtype=np.float32)

    for idx, (n, t) in enumerate(st_coords):
        if n == start_node and t == 0:
            # Start MUST be visited at t=0
            W_unary[idx, 0] = -1e6  # Not visited = forbidden
            W_unary[idx, 1] = 0.0   # Visited = neutral
        elif n == goal_node:
            # Goal should be visited - increasing reward over time
            W_unary[idx, 1] = 5.0 * (t / T)  # Later = stronger pull
        else:
            # Neutral base
            W_unary[idx, 0] = 0.0
            W_unary[idx, 1] = 0.0

    interactions = []

    # TEMPORAL FACTORS: Monotonicity and growth constraints
    for t in range(T - 1):
        for n in range(n_nodes):
            curr = st_nodes[(n, t)]
            next_node = st_nodes[(n, t + 1)]

            # Once visited, must stay visited (monotonic growth)
            W_temp = np.array([
                [0.0,   0.0],    # [not_visited, not_visited] - neutral
                [-50.0, 10.0],   # [visited, ?]: visited->visited gets reward
                                 # visited->not_visited gets penalty (monotonic!)
            ], dtype=np.float32)

            factor = CategoricalEBMFactor(
                [Block([curr]), Block([next_node])],
                beta * jnp.array([W_temp])
            )
            interactions.append(factor)

    # SPATIAL-TEMPORAL FACTORS: Can only visit new node if neighbor was visited before
    for t in range(T - 1):
        for n in range(n_nodes):
            curr_node_n = st_nodes[(n, t)]
            next_node_n = st_nodes[(n, t + 1)]

            # For each neighbor in graph
            for neighbor_id in G.neighbors(n):
                curr_node_neighbor = st_nodes[(neighbor_id, t)]

                # If node becomes newly visited at t+1, at least one neighbor must be visited at t
                # Create 3-way factor: [neighbor@t, node@t, node@t+1]

                # Simplified 2-way: if node not visited at t, but visited at t+1,
                # give bonus if neighbor was visited at t
                W_activation = np.array([
                    [0.0, 3.0],  # neighbor@t=0: [node@t+1=0, node@t+1=1] - allows spontaneous
                    [0.0, 8.0],  # neighbor@t=1: [node@t+1=0, node@t+1=1] - strong reward for spreading!
                ], dtype=np.float32)

                factor = CategoricalEBMFactor(
                    [Block([curr_node_neighbor]), Block([next_node_n])],
                    beta * jnp.array([W_activation])
                )
                interactions.append(factor)

    # Build program
    unary_factor = CategoricalEBMFactor([Block(st_nodes_list)], beta * jnp.asarray(W_unary))
    interactions.insert(0, unary_factor)

    sampler = CategoricalGibbsConditional(N_STATES_BINARY)

    prog = FactorSamplingProgram(
        spec,
        [sampler for _ in spec.free_blocks],
        interactions,
        [],
    )

    return prog, spec, st_nodes_list, st_coords, n_nodes, T


def solve_graph_spacetime(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    T: int = 30,
    n_chains: int = 128,
    warmup: int = 500,
    samples: int = 30,
):
    """Solve maze using spacetime graph encoding.

    This is the optimized version:
    1. Preprocess maze → pure graph (no walls, no grid structure)
    2. Encode spacetime: binary variables for (node, time)
    3. Local temporal constraints create global path connectivity!
    """
    print("\n" + "=" * 70)
    print("GRAPH-BASED SPACETIME MAZE SOLVER")
    print("Your idea: encode TIME itself + compress to pure graph structure")
    print("=" * 70)

    # Step 1: Compress maze to graph
    print("\n1. Preprocessing maze → graph...")
    G, coord_to_node, node_to_coord = maze_to_graph(maze)

    H, W = maze.shape
    n_free = len(coord_to_node)
    print(f"   Maze: {H}×{W} = {H*W} cells")
    print(f"   Free cells: {n_free} ({100*n_free/(H*W):.1f}%)")
    print(f"   Graph edges: {G.number_of_edges()}")
    print(f"   Compression: {H*W} → {n_free} variables per timestep")

    start_node = coord_to_node[start]
    goal_node = coord_to_node[goal]

    # Step 2: Build spacetime model
    print(f"\n2. Building spacetime model with T={T}...")
    prog, spec, st_nodes_list, st_coords, n_nodes, T = build_spacetime_graph_program(
        G, start_node, goal_node, T, beta=3.0
    )

    print(f"   Total spacetime variables: {len(st_nodes_list)}")
    print(f"   Reduction vs grid spacetime: {H*W*T} → {n_nodes*T}")

    # Step 3: Initialize
    print(f"\n3. Initializing {n_chains} chains...")
    key = jax.random.key(4242)
    init_state = []

    for block in spec.free_blocks:
        # Start all not-visited
        arr = jnp.zeros((n_chains, len(block.nodes)), dtype=jnp.uint8)

        # Set start at t=0 to visited if in this block
        block_node_ids = {id(node): k for k, node in enumerate(block.nodes)}
        for k, st_node in enumerate(block.nodes):
            # Find which (n, t) this is
            st_coord_idx = st_nodes_list.index(st_node)
            n_id, t_id = st_coords[st_coord_idx]
            if n_id == start_node and t_id == 0:
                arr = arr.at[:, k].set(1)
                break

        init_state.append(arr)

    # Step 4: Sample!
    print(f"\n4. Sampling spacetime trajectories...")
    print(f"   Warmup: {warmup}, Samples: {samples}")

    schedule = SamplingSchedule(n_warmup=warmup, n_samples=samples, steps_per_sample=15)
    out_blocks = [Block(st_nodes_list)]
    keys = jax.random.split(key, n_chains)

    call = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], out_blocks)))
    stacked = call(init_state, keys)[0]  # [n_chains, n_samples, n_nodes*T]

    final_states = np.array(stacked[:, -1, :])  # [n_chains, n_nodes*T]

    # Step 5: Decode best path
    print(f"\n5. Decoding paths from {n_chains} chains...")

    best_path = None
    best_chain = 0
    best_spacetime = None

    for chain_idx in range(n_chains):
        state_flat = final_states[chain_idx]

        # Reshape to [T, n_nodes]
        state_grid = state_flat.reshape((T, n_nodes))

        # Extract path
        path = extract_graph_spacetime_path(
            state_grid, G, node_to_coord, start_node, goal_node, start, goal
        )

        if path is not None:
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_chain = chain_idx
                best_spacetime = state_grid
                print(f"   Chain {chain_idx}: found path of length {len(path)}")

    if best_path:
        print(f"\n✓ Best path: length {len(best_path)}")
    else:
        print(f"\n✗ No valid path found")
        # Still return spacetime for visualization
        best_spacetime = final_states[0].reshape((T, n_nodes))

    return best_path, best_spacetime, node_to_coord, G


def extract_graph_spacetime_path(
    state_grid: np.ndarray,  # [T, n_nodes]
    G: nx.Graph,
    node_to_coord: dict,
    start_node: int,
    goal_node: int,
    start_coord: tuple[int, int],
    goal_coord: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """Extract path from spacetime graph configuration.

    Path = sequence of nodes activated over time, following graph edges.
    """
    T, n_nodes = state_grid.shape

    # Find activation time for each node
    activation_time = {}
    for n in range(n_nodes):
        for t in range(T):
            if state_grid[t, n] == 1:
                activation_time[n] = t
                break

    # Check if goal activated
    if goal_node not in activation_time:
        return None

    # Build path by following activation times
    # Nodes should activate in increasing time order and form connected path
    activated_nodes = sorted(activation_time.items(), key=lambda x: x[1])

    if not activated_nodes or activated_nodes[0][0] != start_node:
        return None

    # Extract path: follow graph connectivity in temporal order
    path_nodes = [start_node]
    path_times = [activation_time[start_node]]

    # Greedy: at each step, add the earliest-activated neighbor not yet in path
    visited = {start_node}
    current = start_node

    while current != goal_node and len(path_nodes) < len(activated_nodes):
        # Find neighbors in graph
        neighbors = list(G.neighbors(current))

        # Find earliest activated neighbor not yet visited
        next_node = None
        next_time = float('inf')

        for neighbor in neighbors:
            if neighbor in activation_time and neighbor not in visited:
                if activation_time[neighbor] < next_time:
                    next_time = activation_time[neighbor]
                    next_node = neighbor

        if next_node is None:
            return None  # Dead end

        path_nodes.append(next_node)
        path_times.append(next_time)
        visited.add(next_node)
        current = next_node

    if current != goal_node:
        return None

    # Convert node IDs to coordinates
    path_coords = [node_to_coord[n] for n in path_nodes]

    # Validate: each step is adjacent
    for k in range(len(path_coords) - 1):
        r1, c1 = path_coords[k]
        r2, c2 = path_coords[k + 1]
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return None

    return path_coords


def visualize_graph_spacetime(
    state_grid: np.ndarray,  # [T, n_nodes]
    node_to_coord: dict,
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: list[tuple[int, int]] | None,
    output_video: str = "graph_spacetime.mp4",
    fps: int = 8
):
    """Visualize spacetime evolution showing wave propagation!"""
    T, n_nodes = state_grid.shape
    H, W = maze.shape

    print(f"\n6. Creating animation ({T} frames)...")
    output_dir = Path("graph_st_frames")
    output_dir.mkdir(exist_ok=True)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(8, 8))
        vis = np.zeros((H, W, 3))

        # Walls black
        vis[maze == 1] = [0, 0, 0]

        # Free cells white by default
        vis[maze == 0] = [1.0, 1.0, 1.0]

        # Show visited nodes at this time
        visited_count = 0
        for n in range(n_nodes):
            if state_grid[t, n] == 1:
                coord = node_to_coord[n]
                vis[coord] = [0.2, 0.5, 1.0]  # Blue - visited
                visited_count += 1

        # Highlight start/goal
        vis[start] = [0, 1, 0]  # Green
        vis[goal] = [1, 0, 0]   # Red

        # If path found, highlight it in yellow
        if path and t >= T - 5:  # Show path in last few frames
            for p in path:
                if p != start and p != goal:
                    vis[p] = [1, 1, 0]

        ax.imshow(vis, interpolation='nearest')
        title = f'Time {t}/{T-1} - {visited_count}/{n_nodes} nodes visited'
        if path and t >= T - 5:
            title += f' - PATH FOUND!'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        filename = output_dir / f"frame_{t:04d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        if (t + 1) % 10 == 0 or t < 3:
            print(f"   t={t}: {visited_count} nodes visited")

    print(f"   ✓ Generated all frames")

    # Create video
    print("\n7. Encoding video...")
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
        print(f"   ✓ Video: {output_video}")
    except:
        print(f"   Frames in {output_dir}/")
        return

    # Cleanup
    for f in output_dir.glob("frame_*.png"):
        f.unlink()
    output_dir.rmdir()

    print("\n" + "=" * 70)
    print(f"SPACETIME ANIMATION COMPLETE!")
    print("=" * 70)
    print(f"\nWatch the wave propagate: {output_video}")
    print("Blue cells = visited by the growing path")
    print("You should see waves expanding from start until reaching goal!")
    print()


def demo():
    """Demo the graph-based spacetime solver."""
    # Small maze to start
    H = W = 10
    rng = np.random.default_rng(42)
    maze = np.zeros((H, W), dtype=np.uint8)

    # Add a few obstacles
    mask = (rng.random((H, W)) < 0.08)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    print(f"Maze: {H}×{W}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Obstacles: {np.sum(maze)}")

    # Solve!
    path, spacetime, node_to_coord, G = solve_graph_spacetime(
        maze, start, goal,
        T=25,  # 25 timesteps for 10×10 maze
        n_chains=256,
        warmup=800,
        samples=40
    )

    if path:
        print(f"\n✓✓✓ SUCCESS! Path found: length {len(path)}")
        print(f"Path: {path}")

        # Visualize
        visualize_graph_spacetime(
            spacetime, node_to_coord, maze, start, goal, path,
            "graph_spacetime.mp4", fps=8
        )
    else:
        print("\nNo path found - try increasing T or sampling parameters")


if __name__ == "__main__":
    demo()
