#!/usr/bin/env python3
"""Pure graph-based maze solver - compressed representation.

Key optimization: Preprocess maze into a minimal graph:
- Only free cells become nodes
- Only valid movements become edges
- THRML operates on pure graph structure (no grid, no walls)

This is much more efficient than encoding the full grid!
"""
import numpy as np
import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram


# For graph traversal: each node can be in one of 3 states
# 0 = not on path, 1 = on path (intermediate), 2 = endpoint (start or goal)
N_STATES_GRAPH = 3


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


def build_graph_program(
    G: nx.Graph,
    start_node: int,
    goal_node: int,
    beta: float = 5.0,
):
    """Build THRML program on pure graph.

    States: 0=not_on_path, 1=on_path, 2=endpoint

    Constraints:
    - Start and goal must be state 2 (endpoints)
    - Other nodes can be 0 (not on path) or 1 (on path)
    - If node is state 1 or 2, it must have exactly 2 neighbors that are also 1 or 2
      (to form a path)
    - Strong reward for connectivity
    """
    n_nodes = G.number_of_nodes()

    # Create THRML nodes
    thrml_nodes = [CategoricalNode() for _ in range(n_nodes)]

    # Bipartite coloring
    colors = nx.bipartite.color(G)
    color0 = [thrml_nodes[n] for n, c in colors.items() if c == 0]
    color1 = [thrml_nodes[n] for n, c in colors.items() if c == 1]

    blocks = [Block(color0), Block(color1)]
    spec = BlockGibbsSpec(blocks, [])

    # UNARY FACTORS: Role constraints
    W_unary = np.zeros((n_nodes, N_STATES_GRAPH), dtype=np.float32)

    # Compute distances for guidance
    distances_from_start = dict(nx.single_source_shortest_path_length(G, start_node))
    distances_from_goal = dict(nx.single_source_shortest_path_length(G, goal_node))
    max_dist = max(distances_from_goal.values())

    for n in range(n_nodes):
        if n == start_node or n == goal_node:
            # Must be endpoint
            W_unary[n, 0] = -1e6  # Not on path - forbidden
            W_unary[n, 1] = -1e6  # On path - forbidden
            W_unary[n, 2] = 0.0   # Endpoint - required
        else:
            # Can be not-on-path or on-path
            W_unary[n, 0] = 0.0  # Not on path - neutral
            W_unary[n, 2] = -1e6  # Endpoint - forbidden (only start/goal)

            # Gradient: nodes closer to both endpoints prefer being on path
            if n in distances_from_start and n in distances_from_goal:
                d_start = distances_from_start[n]
                d_goal = distances_from_goal[n]

                # Normalize
                norm_start = d_start / max_dist
                norm_goal = d_goal / max_dist

                # Bidirectional pull
                pull = 1.5 * ((1.0 - norm_start) + (1.0 - norm_goal)) / 2.0
                W_unary[n, 1] = pull  # Reward being on path if close to endpoints

    interactions = []

    # PAIRWISE FACTORS: Path connectivity
    edge_list = list(G.edges())

    for node_a, node_b in edge_list:
        thrml_a = thrml_nodes[node_a]
        thrml_b = thrml_nodes[node_b]

        # Pairwise weight matrix [state_a, state_b]
        # Strong reward if both on path (forms connected path)
        # Penalty if only one is on path (creates dead ends)
        W_pair = np.array([
            # state_b:      0      1      2
            [0.0,   -2.0,  -2.0],   # state_a=0 (not on path)
            [-2.0,   5.0,   5.0],   # state_a=1 (on path) - reward if neighbor also on path!
            [-2.0,   5.0,   5.0],   # state_a=2 (endpoint) - reward if neighbor on path!
        ], dtype=np.float32)

        factor = CategoricalEBMFactor(
            [Block([thrml_a]), Block([thrml_b])],
            beta * jnp.array([W_pair])
        )
        interactions.append(factor)

    # Degree constraints: nodes on path should have degree 1 or 2
    # This is harder to encode... for now rely on pairwise rewards

    # Build program
    unary_factor = CategoricalEBMFactor([Block(thrml_nodes)], beta * jnp.asarray(W_unary))
    interactions.insert(0, unary_factor)

    sampler = CategoricalGibbsConditional(N_STATES_GRAPH)

    prog = FactorSamplingProgram(
        spec,
        [sampler for _ in spec.free_blocks],
        interactions,
        [],
    )

    return prog, spec, thrml_nodes, G


def decode_graph_path(
    state_vec: np.ndarray,
    G: nx.Graph,
    node_to_coord: dict,
    start_node: int,
    goal_node: int,
) -> list[tuple[int, int]] | None:
    """Decode path from graph node states."""
    n_nodes = len(state_vec)

    # Find all nodes on path (state 1 or 2)
    path_nodes = set()
    for n in range(n_nodes):
        if state_vec[n] >= 1:  # On path or endpoint
            path_nodes.add(n)

    if start_node not in path_nodes or goal_node not in path_nodes:
        return None

    # Build subgraph of path nodes
    path_graph = G.subgraph(path_nodes).copy()

    # Find shortest path from start to goal in path subgraph
    try:
        node_path = nx.shortest_path(path_graph, start_node, goal_node)
    except nx.NetworkXNoPath:
        return None

    # Convert to coordinates
    coord_path = [node_to_coord[n] for n in node_path]

    return coord_path


def solve_graph_maze(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    n_chains: int = 128,
    warmup: int = 500,
    samples: int = 30,
):
    """Solve maze using pure graph representation."""
    H, W = maze.shape

    print("\n" + "=" * 70)
    print("GRAPH-BASED MAZE SOLVER")
    print("Compressed representation: maze → graph → THRML")
    print("=" * 70)

    # Preprocess
    print("\n1. Preprocessing maze → graph...")
    G, coord_to_node, node_to_coord = maze_to_graph(maze)

    n_free = len(coord_to_node)
    print(f"   Maze: {H}×{W} = {H*W} cells")
    print(f"   Free cells: {n_free} ({100*n_free/(H*W):.1f}%)")
    print(f"   Graph edges: {G.number_of_edges()}")
    print(f"   Compression: {H*W} → {n_free} variables")

    if start not in coord_to_node or goal not in coord_to_node:
        print("   ✗ Start or goal is in a wall!")
        return None, None

    start_node = coord_to_node[start]
    goal_node = coord_to_node[goal]

    # Build program
    print(f"\n2. Building THRML program on graph...")
    prog, spec, thrml_nodes, G = build_graph_program(G, start_node, goal_node, beta=5.0)
    print(f"   Variables: {len(thrml_nodes)}")
    print(f"   Free blocks: {len(spec.free_blocks)}")

    # Initialize
    print(f"\n3. Initializing {n_chains} chains...")
    key = jax.random.key(4242)
    init_state = []

    for block in spec.free_blocks:
        # Most nodes not on path (state 0)
        key, sub = jax.random.split(key)
        arr = jax.random.choice(
            sub,
            jnp.array([0, 1], dtype=jnp.uint8),
            shape=(n_chains, len(block.nodes)),
            p=jnp.array([0.9, 0.1])  # 90% empty, 10% on path
        )

        # Force start and goal to be endpoints (state 2)
        block_node_ids = [id(node) for node in block.nodes]
        start_thrml_id = id(thrml_nodes[start_node])
        goal_thrml_id = id(thrml_nodes[goal_node])

        if start_thrml_id in block_node_ids:
            idx = block_node_ids.index(start_thrml_id)
            arr = arr.at[:, idx].set(2)
        if goal_thrml_id in block_node_ids:
            idx = block_node_ids.index(goal_thrml_id)
            arr = arr.at[:, idx].set(2)

        init_state.append(arr)

    # Sample
    print(f"\n4. Sampling...")
    schedule = SamplingSchedule(n_warmup=warmup, n_samples=samples, steps_per_sample=10)
    out_blocks = [Block(thrml_nodes)]
    keys = jax.random.split(key, n_chains)

    call = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], out_blocks)))
    stacked = call(init_state, keys)[0]  # [n_chains, n_samples, n_nodes]

    final_states = np.array(stacked[:, -1, :])  # [n_chains, n_nodes]

    print(f"   Sampled {n_chains} chains")

    # Decode
    print(f"\n5. Decoding paths...")
    best_path = None

    for chain_idx in range(n_chains):
        state_vec = final_states[chain_idx]
        path = decode_graph_path(state_vec, G, node_to_coord, start_node, goal_node)

        if path is not None:
            if best_path is None or len(path) < len(best_path):
                best_path = path
                print(f"   Chain {chain_idx}: found path of length {len(path)}")

    if best_path:
        print(f"\n✓ Best path: length {len(best_path)}")
    else:
        print(f"\n✗ No valid path found")

    return best_path, final_states[0] if best_path else None, node_to_coord, G


def visualize_graph_solution(
    maze: np.ndarray,
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    output_file: str = "graph_solution.png"
):
    """Visualize the solution."""
    H, W = maze.shape

    fig, ax = plt.subplots(figsize=(10, 10))
    vis = np.zeros((H, W, 3))

    # Walls black
    vis[maze == 1] = [0, 0, 0]

    # Free cells white
    vis[maze == 0] = [1.0, 1.0, 1.0]

    # Path in blue
    if path:
        for p in path:
            if p != start and p != goal:
                vis[p] = [0.2, 0.5, 1.0]

    # Start/goal
    vis[start] = [0, 1, 0]  # Green
    vis[goal] = [1, 0, 0]   # Red

    ax.imshow(vis, interpolation='nearest')
    title = f'Graph Solver: Path length {len(path) if path else "N/A"}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n   Saved: {output_file}")
    plt.close()


def demo():
    """Demo graph-based solver."""
    print("\n" + "=" * 70)
    print("GRAPH-BASED MAZE SOLVER DEMO")
    print("=" * 70)

    # Test 1: Simple small maze
    print("\n### Test 1: 10×10 with obstacles")
    H = W = 10
    rng = np.random.default_rng(42)
    maze = np.zeros((H, W), dtype=np.uint8)

    mask = (rng.random((H, W)) < 0.1)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    path, state_vec, node_to_coord, G = solve_graph_maze(
        maze, start, goal,
        n_chains=256,
        warmup=800,
        samples=40
    )

    if path:
        visualize_graph_solution(maze, path, start, goal, "graph_test1.png")

    # Test 2: Larger maze
    print("\n### Test 2: 20×20 with obstacles")
    H = W = 20
    maze = np.zeros((H, W), dtype=np.uint8)

    mask = (rng.random((H, W)) < 0.12)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    path, state_vec, node_to_coord, G = solve_graph_maze(
        maze, start, goal,
        n_chains=512,
        warmup=1200,
        samples=60
    )

    if path:
        visualize_graph_solution(maze, path, start, goal, "graph_test2.png")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    demo()

