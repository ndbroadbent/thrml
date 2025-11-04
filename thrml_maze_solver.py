"""THRML-based maze solver using block Gibbs sampling.

This module implements a probabilistic maze solver that encodes path tiles as
categorical variables and uses energy-based models with block Gibbs sampling
to find shortest paths through mazes.
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


# 11 tile states encoded as connectors [N,E,S,W]
# 0=empty, 1..4=single-ends (N,E,S,W), 5..10=two-ends (NE,NS,NW,ES,EW,SW)
STATE_BITS = np.array([
    [0, 0, 0, 0],  # 0  empty
    [1, 0, 0, 0],  # 1  N
    [0, 1, 0, 0],  # 2  E
    [0, 0, 1, 0],  # 3  S
    [0, 0, 0, 1],  # 4  W
    [1, 1, 0, 0],  # 5  NE
    [1, 0, 1, 0],  # 6  NS
    [1, 0, 0, 1],  # 7  NW
    [0, 1, 1, 0],  # 8  ES
    [0, 1, 0, 1],  # 9  EW
    [0, 0, 1, 1],  # 10 SW
], dtype=np.uint8)

DEG = STATE_BITS.sum(axis=1)
N_STATES = STATE_BITS.shape[0]


def _pair_matrix(lambda_edge: float, mismatch_penalty: float, axis: str) -> np.ndarray:
    """Create pairwise factor matrix for edge consistency.

    Args:
        lambda_edge: Penalty weight for using an edge (encourages shorter paths)
        mismatch_penalty: Penalty for edge mismatches (adjustable for annealing)
        axis: Either 'h' (horizontal) or 'v' (vertical)
            - 'h': enforce left.E == right.W
            - 'v': enforce top.S == bottom.N

    Returns:
        N_STATES x N_STATES matrix of pairwise weights
    """
    # Use soft penalty for mismatches - strength controlled by caller
    W = np.full((N_STATES, N_STATES), -mismatch_penalty, dtype=np.float32)

    if axis == "h":
        for a in range(N_STATES):
            for b in range(N_STATES):
                ea, wb = STATE_BITS[a, 1], STATE_BITS[b, 3]
                if ea == wb:
                    W[a, b] = -lambda_edge if ea == 1 else 0.0
    else:  # axis == "v"
        for a in range(N_STATES):
            for b in range(N_STATES):
                sa, nb = STATE_BITS[a, 2], STATE_BITS[b, 0]
                if sa == nb:
                    W[a, b] = -lambda_edge if sa == 1 else 0.0

    return W


def _compute_distance_field(maze: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Compute distance from each free cell to goal using BFS.

    Creates a potential field that guides sampling toward the goal.
    """
    from collections import deque

    H, W = maze.shape
    distances = np.full((H, W), np.inf)
    distances[goal] = 0

    queue = deque([goal])

    while queue:
        r, c = queue.popleft()
        current_dist = distances[r, c]

        # Check 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and maze[nr, nc] == 0:
                if distances[nr, nc] > current_dist + 1:
                    distances[nr, nc] = current_dist + 1
                    queue.append((nr, nc))

    return distances


def build_thrml_program(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    beta: float = 5.0,
    lambda_edge: float = 0.05,
    hard: float = 1e6,
    constraint_strength: float = 10.0,  # NEW: adjustable constraint strength for annealing
):
    """Build THRML sampling program for maze solving.

    Args:
        maze: HxW array with 1=wall, 0=free
        start: (row, col) starting position
        goal: (row, col) goal position
        beta: Inverse temperature for sampling
        lambda_edge: Penalty per edge (higher = prefer shorter paths)
        hard: Large negative weight for constraint violations

    Returns:
        Tuple of (prog, spec, nodes, coords, G) where:
        - prog: FactorSamplingProgram for block Gibbs sampling
        - spec: BlockGibbsSpec with free blocks
        - nodes: List of CategoricalNodes in grid order
        - coords: List of (row, col) coordinates corresponding to nodes
        - G: NetworkX graph of the grid
    """
    H, W = maze.shape

    # Compute distance fields from BOTH start and goal (bidirectional gradient)
    distances_from_goal = _compute_distance_field(maze, goal)
    distances_from_start = _compute_distance_field(maze, start)

    max_dist_goal = np.max(distances_from_goal[distances_from_goal != np.inf])
    max_dist_start = np.max(distances_from_start[distances_from_start != np.inf])

    if max_dist_goal == 0:
        max_dist_goal = 1.0
    if max_dist_start == 0:
        max_dist_start = 1.0

    # Grid graph over all cells
    G = nx.grid_graph(dim=(H, W), periodic=False)

    # Relabel with THRML categorical nodes and stash coords for each node
    coord_to_node = {(i, j): CategoricalNode() for i in range(H) for j in range(W)}
    nx.relabel_nodes(G, coord_to_node, copy=False)
    for coord, node in coord_to_node.items():
        G.nodes[node]["coord"] = coord

    # Bipartite coloring -> 2 update blocks
    colors = nx.bipartite.color(G)
    color0 = [n for n, c in colors.items() if c == 0]
    color1 = [n for n, c in colors.items() if c == 1]
    blocks = [Block(color0), Block(color1)]
    spec = BlockGibbsSpec(blocks, [])

    # Node order we will use when requesting states back
    coords = [(i, j) for i in range(H) for j in range(W)]
    nodes = [coord_to_node[c] for c in coords]

    # Unary weights per node with distance-based potential field
    W_unary = np.full((H * W, N_STATES), -hard, dtype=np.float32)
    start_set = {start}
    goal_set = {goal}

    # Distance potential strength - creates the "water flow" gradient
    # Balance: strong enough to guide, weak enough to allow exploration
    # With constraint_strength=10, we need gentle guidance
    alpha = 0.3  # Gentle gradient - works with soft constraints
    bridge_bonus = 0.2  # Subtle bridge bonus

    # Optimal path length (if path exists)
    optimal_dist = distances_from_goal[start]
    if optimal_dist == np.inf or optimal_dist == 0:
        optimal_dist = max(max_dist_goal, 1.0)  # No path exists or start==goal, use safe fallback

    for idx, coord in enumerate(coords):
        if maze[coord] == 1:  # wall
            # Use HARD wall constraints - walls are absolute boundaries
            W_unary[idx, :] = -hard
            W_unary[idx, 0] = 0.0  # force empty
        else:
            # Base constraints
            if coord in start_set or coord in goal_set:
                allowed = (DEG == 1)  # endpoints must be degree 1
            else:
                allowed = (DEG == 0) | (DEG == 2)  # interior: empty or path-through
            W_unary[idx, allowed] = 0.0

            # Add bidirectional distance potential (water flows from start to goal)
            dist_to_goal = distances_from_goal[coord]
            dist_from_start = distances_from_start[coord]

            if dist_to_goal != np.inf and dist_from_start != np.inf:
                if coord not in (start_set | goal_set):
                    # Normalize distances (0 = at endpoint, 1 = farthest away)
                    norm_dist_to_goal = dist_to_goal / max_dist_goal
                    norm_dist_from_start = dist_from_start / max_dist_start

                    for state in range(N_STATES):
                        if DEG[state] > 0 and allowed[state]:  # If this is a path tile
                            # Bidirectional search: encourage path growth from BOTH endpoints

                            # Base attraction from each endpoint
                            start_pull = alpha * (1.0 - norm_dist_from_start)
                            goal_pull = alpha * (1.0 - norm_dist_to_goal)

                            # Use average so both endpoints contribute
                            # (max would only use the stronger pull)
                            base_reward = (start_pull + goal_pull) / 2.0

                            # Bridge bonus: extra reward for cells accessible from BOTH sides
                            # These are the cells that can connect the two growing paths
                            # Highest bonus in the middle, decreases toward endpoints
                            accessibility = (1.0 - norm_dist_from_start) * (1.0 - norm_dist_to_goal)
                            bridge_reward = bridge_bonus * accessibility

                            W_unary[idx, state] += base_reward + bridge_reward

    # Pairwise weights for each neighbor direction
    # Build edge lists once per direction so factor batches align properly
    u_right, v_right = [], []
    u_down, v_down = [], []

    for i in range(H):
        for j in range(W):
            if j + 1 < W:
                u_right.append(coord_to_node[(i, j)])
                v_right.append(coord_to_node[(i, j + 1)])
            if i + 1 < H:
                u_down.append(coord_to_node[(i, j)])
                v_down.append(coord_to_node[(i + 1, j)])

    Wh = _pair_matrix(lambda_edge, constraint_strength, axis="h")
    Wv = _pair_matrix(lambda_edge, constraint_strength, axis="v")

    # ADD: Activation spreading - neighbors prefer similar activity levels
    # This creates wave-like propagation naturally through sampling!
    activation_coupling = 1.5  # Strength of spreading
    Wh_spread = np.zeros((N_STATES, N_STATES), dtype=np.float32)
    Wv_spread = np.zeros((N_STATES, N_STATES), dtype=np.float32)

    for a in range(N_STATES):
        for b in range(N_STATES):
            # Reward when neighbors have similar "activity" (degree)
            deg_a, deg_b = DEG[a], DEG[b]

            # If both active (deg > 0), give bonus
            if deg_a > 0 and deg_b > 0:
                # Stronger bonus if both are deg=2 (corridor forming)
                if deg_a == 2 and deg_b == 2:
                    Wh_spread[a, b] = activation_coupling * 1.5
                    Wv_spread[a, b] = activation_coupling * 1.5
                else:
                    Wh_spread[a, b] = activation_coupling
                    Wv_spread[a, b] = activation_coupling
            # Small bonus even if one is empty and other deg=1 (allows initiation)
            elif (deg_a == 0 and deg_b == 1) or (deg_a == 1 and deg_b == 0):
                Wh_spread[a, b] = activation_coupling * 0.3
                Wv_spread[a, b] = activation_coupling * 0.3

    weights_unary = beta * jnp.asarray(W_unary)
    weights_h = beta * jnp.broadcast_to(jnp.asarray(Wh + Wh_spread), (len(u_right), N_STATES, N_STATES))
    weights_v = beta * jnp.broadcast_to(jnp.asarray(Wv + Wv_spread), (len(u_down), N_STATES, N_STATES))

    # Build factors
    unary_factor = CategoricalEBMFactor([Block(nodes)], weights_unary)
    horiz_factor = CategoricalEBMFactor([Block(u_right), Block(v_right)], weights_h)
    vertical_factor = CategoricalEBMFactor([Block(u_down), Block(v_down)], weights_v)

    interactions = [unary_factor, horiz_factor, vertical_factor]
    sampler = CategoricalGibbsConditional(N_STATES)

    prog = FactorSamplingProgram(
        spec,
        [sampler for _ in spec.free_blocks],
        interactions,
        [],
    )

    return prog, spec, nodes, coords, G


def _decode_path(
    final_states_vec: np.ndarray,
    coords: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    maze: np.ndarray | None = None
) -> list[tuple[int, int]] | None:
    """Decode a path from state vector by following connectors.

    Args:
        final_states_vec: Array of length H*W with state index per cell
        coords: List of (row, col) coordinates corresponding to state vector
        start: Starting position
        goal: Goal position
        maze: Optional maze array to verify path doesn't go through walls

    Returns:
        List of (row, col) coordinates from start to goal, or None if invalid
    """
    H = max(c[0] for c in coords) + 1
    W = max(c[1] for c in coords) + 1
    coord_to_idx = {c: i for i, c in enumerate(coords)}

    def neighbors_from_state(coord):
        """Get neighbors and bits for a coordinate."""
        i, j = coord
        s = final_states_vec[coord_to_idx[(i, j)]]
        bits = STATE_BITS[int(s)]
        nbrs = []
        if bits[0] and i - 1 >= 0:
            nbrs.append((i - 1, j))  # N
        if bits[1] and j + 1 < W:
            nbrs.append((i, j + 1))  # E
        if bits[2] and i + 1 < H:
            nbrs.append((i + 1, j))  # S
        if bits[3] and j - 1 >= 0:
            nbrs.append((i, j - 1))  # W
        return nbrs, bits

    path = [start]
    visited = set([start])
    prev = None
    cur = start
    max_steps = H * W * 2

    for _ in range(max_steps):
        if cur == goal:
            return path

        nbrs, bits_cur = neighbors_from_state(cur)

        # Degree checks
        deg = int(bits_cur.sum())
        if cur == start:
            if deg != 1:
                return None
        elif cur == goal:
            if deg != 1:
                return None
        else:
            if deg not in (0, 2):
                return None

        # Choose next
        cand = [n for n in nbrs if n != prev]
        if len(cand) == 0:
            return None
        nxt = cand[0]

        # Consistency check on neighbor reciprocal bit
        ni, nj = nxt
        s_nxt = final_states_vec[coord_to_idx[(ni, nj)]]
        bits_nxt = STATE_BITS[int(s_nxt)]
        di, dj = ni - cur[0], nj - cur[1]

        # Edge must be reciprocated
        if di == -1 and bits_nxt[2] != 1:
            return None  # neighbor S
        if di == 1 and bits_nxt[0] != 1:
            return None  # neighbor N
        if dj == 1 and bits_nxt[3] != 1:
            return None  # neighbor W
        if dj == -1 and bits_nxt[1] != 1:
            return None  # neighbor E

        if nxt in visited:  # loop encountered in main path - invalid
            return None

        # Check if next cell is a wall (if maze provided)
        if maze is not None and maze[nxt] == 1:
            return None

        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    return None


def solve_with_thrml(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    beta: float = 5.0,
    lambda_edge: float = 0.05,
    hard: float = 1e6,
    n_chains: int = 64,
    warmup: int = 300,
    samples: int = 16,
    steps_per_sample: int = 8,
):
    """Solve a maze using THRML block Gibbs sampling.

    Args:
        maze: HxW array with 1=wall, 0=free
        start: (row, col) starting position
        goal: (row, col) goal position
        beta: Inverse temperature (higher = more deterministic)
        lambda_edge: Penalty per edge (higher = prefer shorter paths)
        hard: Large negative weight for constraint violations
        n_chains: Number of parallel sampling chains
        warmup: Number of warmup steps before collecting samples
        samples: Number of samples to collect
        steps_per_sample: Block Gibbs steps between samples

    Returns:
        Tuple of (path, state_vec, coords) where:
        - path: List of (row, col) from start to goal, or None if no path found
        - state_vec: Final state vector for the best chain (or None)
        - coords: List of (row, col) coordinates corresponding to state vector
    """
    prog, spec, nodes, coords, G = build_thrml_program(
        maze, start, goal, beta, lambda_edge, hard
    )

    # Initialize states per free block
    key = jax.random.key(4242)
    init_state = []
    for block in spec.free_blocks:
        key, sub = jax.random.split(key)
        # Initialize mostly empty (state 0) with occasional path hints
        arr = jax.random.choice(sub, jnp.array([0, 1, 2, 3, 4], dtype=jnp.uint8),
                                shape=(n_chains, len(block.nodes)),
                                p=jnp.array([0.7, 0.075, 0.075, 0.075, 0.075]))

        # Set walls to 0 in init
        block_coords = [G.nodes[n]["coord"] for n in block.nodes]
        wall_idx = [k for k, c in enumerate(block_coords) if maze[c] == 1]
        if wall_idx:
            arr = arr.at[:, jnp.array(wall_idx)].set(0)

        # Try to initialize start and goal with reasonable states
        # (This is a hint but sampling will adjust if needed)
        start_idx = [k for k, c in enumerate(block_coords) if c == start]
        goal_idx = [k for k, c in enumerate(block_coords) if c == goal]
        if start_idx:
            # Give start some random degree-1 state
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

    # Vectorize over chains
    call = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], out_blocks)))

    # Returns a PyTree with one tensor (we asked for one Block)
    # Shape: (n_chains, n_samples, n_nodes)
    stacked = call(init_state, keys)[0]

    # Convert to numpy, take final state for each chain
    # Pick first chain that decodes a valid path, preferring the shortest
    final_all = np.array(stacked[:, -1, :])  # [n_chains, n_nodes]

    best_path, best_idx = None, None
    best_len = 10**9

    for b in range(final_all.shape[0]):
        path = _decode_path(final_all[b], coords, start, goal, maze)
        if path is not None and len(path) < best_len:
            best_path, best_idx, best_len = path, b, len(path)

    # For debugging: return first chain's state even if no path found
    return_state = final_all[best_idx] if best_path is not None else final_all[0]
    return best_path, return_state, coords


def demo():
    """Run a demo on a 64x64 grid with random obstacles."""
    # Simple demo on a mostly-open 64x64 grid with a few random blocks
    H = W = 64
    rng = np.random.default_rng(0)
    maze = np.zeros((H, W), dtype=np.uint8)

    # Sprinkle walls but keep borders open
    mask = (rng.random((H, W)) < 0.18)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (1, 1)
    goal = (H - 2, W - 2)
    maze[start] = 0
    maze[goal] = 0

    print(f"Solving {H}x{W} maze from {start} to {goal}...")

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=6.0, lambda_edge=0.05, hard=1e6,
        n_chains=128, warmup=500, samples=30, steps_per_sample=10
    )

    print("Found path:", path is not None)
    if path:
        print(f"Path length: {len(path)}")

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap="gray_r", interpolation="nearest")

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        plt.plot(xs, ys, 'r-', linewidth=2, label='Path')
        plt.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
        plt.plot(xs[-1], ys[-1], 'ro', markersize=10, label='Goal')

    plt.title(f"THRML Maze Solver ({H}x{W})")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("maze_solution.png", dpi=150)
    print("Saved visualization to maze_solution.png")
    plt.show()


if __name__ == "__main__":
    demo()

