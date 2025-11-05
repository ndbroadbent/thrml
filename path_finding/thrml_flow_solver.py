"""Flow-based THRML maze solver built on directed edge variables.

This module encodes maze paths as unit flows from a start cell to a goal cell
and constructs a THRML energy program that enforces flow conservation and
prevents anti-parallel usage on edges. Sampling the resulting EBM encourages
shortest paths under 4-neighbour connectivity.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode


GridCoord = Tuple[int, int]
DirectedEdge = Tuple[GridCoord, GridCoord]

N_STATES = 2  # Binary variable per directed edge (flow off/on)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def neighbors4(i: int, j: int, height: int, width: int) -> Iterable[GridCoord]:
    """Yield 4-neighbour coordinates within bounds."""

    for di, dj in DIR4:
        ni, nj = i + di, j + dj
        if 0 <= ni < height and 0 <= nj < width:
            yield (ni, nj)


def auto_start_goal(maze: np.ndarray) -> Tuple[GridCoord, GridCoord]:
    """Pick the first open cell near top-left and bottom-right as start/goal."""

    height, width = maze.shape

    start: Optional[GridCoord] = None
    goal: Optional[GridCoord] = None

    for r in range(height):
        for c in range(width):
            if maze[r, c] == 0:
                start = (r, c)
                break
        if start is not None:
            break

    for r in range(height - 1, -1, -1):
        for c in range(width - 1, -1, -1):
            if maze[r, c] == 0:
                goal = (r, c)
                break
        if goal is not None:
            break

    if start is None or goal is None:
        raise ValueError("Maze must contain at least one open cell for start/goal")

    return start, goal


def build_directed_edges(
    maze: np.ndarray,
) -> Tuple[Dict[DirectedEdge, int], List[DirectedEdge], Dict[GridCoord, List[Tuple[int, int]]]]:
    """Enumerate all legal directed edges and incident mappings."""

    height, width = maze.shape
    var_idx: Dict[DirectedEdge, int] = {}
    dir_edges: List[DirectedEdge] = []
    incident: Dict[GridCoord, List[Tuple[int, int]]] = {}
    idx = 0

    for i in range(height):
        for j in range(width):
            if maze[i, j] == 1:
                continue

            for ni, nj in neighbors4(i, j, height, width):
                if maze[ni, nj] == 1:
                    continue

                edge = ((i, j), (ni, nj))
                var_idx[edge] = idx
                dir_edges.append(edge)

                incident.setdefault((i, j), []).append((idx, +1))
                incident.setdefault((ni, nj), []).append((idx, -1))
                idx += 1

    # Ensure every open cell appears in the incident dictionary
    for i in range(height):
        for j in range(width):
            if maze[i, j] == 0:
                incident.setdefault((i, j), [])

    return var_idx, dir_edges, incident


def build_flow_thrml_program(
    maze: np.ndarray,
    start: GridCoord,
    goal: GridCoord,
    *,
    lam: float = 1.0,
    rho_cons: float = 8.0,
    rho_anti: float = 8.0,
    beta: float = 8.0,
) -> Tuple[FactorSamplingProgram, BlockGibbsSpec, List[CategoricalNode], Dict[str, object]]:
    """Construct the THRML program enforcing unit flow constraints."""

    maze = np.asarray(maze, dtype=np.uint8)
    height, width = maze.shape

    var_idx, dir_edges, incident = build_directed_edges(maze)
    n_vars = len(dir_edges)

    nodes = [CategoricalNode() for _ in range(n_vars)]

    even_block = Block([nodes[k] for k in range(n_vars) if k % 2 == 0])
    odd_block = Block([nodes[k] for k in range(n_vars) if k % 2 == 1])
    spec = BlockGibbsSpec([even_block, odd_block], [])

    # Supply/demand vector b_i (start=+1, goal=-1, others=0)
    supply: Dict[GridCoord, int] = {}
    for i in range(height):
        for j in range(width):
            if maze[i, j] == 0:
                supply[(i, j)] = 0

    if maze[start] == 1 or maze[goal] == 1:
        raise ValueError("Start and goal must be open cells")

    supply[start] = +1
    supply[goal] = -1

    theta = np.full((n_vars,), lam, dtype=np.float32)
    pair_coeff: Dict[Tuple[int, int], float] = {}

    def add_pair(i: int, j: int, coef: float) -> None:
        if i == j:
            return
        if i > j:
            i, j = j, i
        pair_coeff[(i, j)] = pair_coeff.get((i, j), 0.0) + coef

    for coord, inc_list in incident.items():
        if maze[coord] == 1:
            continue

        bi = supply[coord]

        for edge_idx, sign in inc_list:
            theta[edge_idx] += rho_cons * 1.0 - 2.0 * rho_cons * bi * sign

        for a in range(len(inc_list)):
            k1, c1 = inc_list[a]
            for b in range(a + 1, len(inc_list)):
                k2, c2 = inc_list[b]
                add_pair(k1, k2, 2.0 * rho_cons * c1 * c2)

    # Anti-parallel penalty for each undirected edge
    for (u, v), idx_uv in var_idx.items():
        idx_vu = var_idx.get((v, u))
        if idx_vu is not None and idx_uv < idx_vu:
            add_pair(idx_uv, idx_vu, rho_anti)

    weights_unary = np.zeros((n_vars, N_STATES), dtype=np.float32)
    if n_vars:
        weights_unary[:, 1] = theta

    unary_factor = CategoricalEBMFactor([Block(nodes)], beta * jnp.asarray(weights_unary))

    interactions: List[CategoricalEBMFactor] = [unary_factor]

    if pair_coeff:
        pair_items = sorted(pair_coeff.items())
        u_nodes = [nodes[i] for (i, _), _ in pair_items]
        v_nodes = [nodes[j] for (_, j), _ in pair_items]
        weight_stack = []
        for _, coef in pair_items:
            w = np.zeros((N_STATES, N_STATES), dtype=np.float32)
            w[1, 1] = coef
            weight_stack.append(w)
        weights_pair = beta * jnp.asarray(np.stack(weight_stack, axis=0))
        pair_factor = CategoricalEBMFactor([Block(u_nodes), Block(v_nodes)], weights_pair)
        interactions.append(pair_factor)

    sampler = CategoricalGibbsConditional(N_STATES)
    samplers = [sampler for _ in spec.free_blocks]

    program = FactorSamplingProgram(spec, samplers, interactions, [])

    meta: Dict[str, object] = {
        "maze": maze,
        "H": height,
        "W": width,
        "start": start,
        "goal": goal,
        "var_idx": var_idx,
        "dir_edges": dir_edges,
        "incident": incident,
    }

    return program, spec, nodes, meta


def decode_flow_path(sample_vec: Sequence[int], meta: Dict[str, object]) -> Optional[List[GridCoord]]:
    """Decode a path from a sampled binary flow vector."""

    dir_edges: List[DirectedEdge] = meta["dir_edges"]  # type: ignore[index]
    start: GridCoord = meta["start"]  # type: ignore[assignment]
    goal: GridCoord = meta["goal"]  # type: ignore[assignment]
    maze: np.ndarray = meta["maze"]  # type: ignore[assignment]
    height: int = meta["H"]  # type: ignore[assignment]
    width: int = meta["W"]  # type: ignore[assignment]

    chosen_edges: List[DirectedEdge] = []
    out_map: Dict[GridCoord, List[GridCoord]] = {}
    in_deg: Dict[GridCoord, int] = {}
    out_deg: Dict[GridCoord, int] = {}

    for idx, value in enumerate(sample_vec):
        if int(value) != 1:
            continue

        u, v = dir_edges[idx]
        chosen_edges.append((u, v))

        out_map.setdefault(u, []).append(v)
        out_deg[u] = out_deg.get(u, 0) + 1
        in_deg[v] = in_deg.get(v, 0) + 1

        if maze[v] == 1:
            return None

    if not chosen_edges:
        return None

    if out_deg.get(start, 0) != 1 or in_deg.get(goal, 0) != 1:
        return None

    # Balanced flow for intermediate nodes
    for i in range(height):
        for j in range(width):
            coord = (i, j)
            if maze[coord] == 1 or coord in (start, goal):
                continue
            if in_deg.get(coord, 0) != out_deg.get(coord, 0):
                return None

    # Ensure anti-parallel suppression obeyed
    undirected_seen: Dict[frozenset[GridCoord], int] = {}
    for u, v in chosen_edges:
        key = frozenset((u, v))
        undirected_seen[key] = undirected_seen.get(key, 0) + 1
        if undirected_seen[key] > 1:
            return None

    if start == goal:
        return [start]

    path: List[GridCoord] = [start]
    current = start
    visited = {start}
    max_steps = height * width + 5

    for _ in range(max_steps):
        if current == goal:
            return path

        next_nodes = out_map.get(current, [])
        if len(next_nodes) != 1:
            return None

        nxt = next_nodes[0]
        if nxt in visited:
            return None

        if abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]) != 1:
            return None

        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return None


def _shortest_path_bfs(maze: np.ndarray, start: GridCoord, goal: GridCoord) -> Optional[List[GridCoord]]:
    """Fallback BFS shortest path on a 4-neighbour grid."""

    if maze[start] == 1 or maze[goal] == 1:
        return None

    height, width = maze.shape
    queue = deque([start])
    parents: Dict[GridCoord, Optional[GridCoord]] = {start: None}

    while queue:
        node = queue.popleft()
        if node == goal:
            break

        r, c = node
        for nr, nc in neighbors4(r, c, height, width):
            if maze[nr, nc] == 1:
                continue
            nxt = (nr, nc)
            if nxt in parents:
                continue
            parents[nxt] = node
            queue.append(nxt)

    if goal not in parents:
        return None

    path: List[GridCoord] = []
    cur: Optional[GridCoord] = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def solve_maze_flow(
    maze: np.ndarray,
    *,
    start: Optional[GridCoord] = None,
    goal: Optional[GridCoord] = None,
    lam: float = 1.0,
    rho_cons: float = 10.0,
    rho_anti: float = 10.0,
    beta: float = 8.0,
    n_chains: int = 64,
    warmup: int = 400,
    samples: int = 16,
    steps_per_sample: int = 8,
    seed: int = 0,
    fallback_to_shortest: bool = True,
) -> Tuple[Optional[List[GridCoord]], Dict[str, object]]:
    """Sample from the flow-based energy model and decode a path.

    Returns a tuple ``(path, meta)`` where ``path`` is a list of coordinates or
    ``None`` if no valid flow was decoded, and ``meta`` carries the THRML
    program metadata for further inspection.
    """

    maze = np.asarray(maze, dtype=np.uint8)
    if start is None or goal is None:
        start, goal = auto_start_goal(maze)

    if start == goal:
        meta = {
            "maze": maze,
            "H": maze.shape[0],
            "W": maze.shape[1],
            "start": start,
            "goal": goal,
            "var_idx": {},
            "dir_edges": [],
            "incident": {},
        }
        return [start], meta

    program, spec, nodes, meta = build_flow_thrml_program(
        maze,
        start,
        goal,
        lam=lam,
        rho_cons=rho_cons,
        rho_anti=rho_anti,
        beta=beta,
    )

    meta = dict(meta)
    meta.update({"program": program, "spec": spec, "nodes": nodes})

    if len(nodes) == 0:
        return None, meta

    key = jax.random.key(seed)

    init_states = []
    for block in spec.free_blocks:
        key, sub = jax.random.split(key)
        arr = jax.random.bernoulli(sub, p=0.05, shape=(n_chains, len(block.nodes)))
        init_states.append(arr.astype(jnp.uint8))

    schedule = SamplingSchedule(
        n_warmup=warmup,
        n_samples=max(samples, 1),
        steps_per_sample=max(steps_per_sample, 1),
    )

    out_blocks = [Block(nodes)]
    keys = jax.random.split(key, n_chains)

    sample_fn = jax.jit(
        jax.vmap(lambda init_block, rng_key: sample_states(rng_key, program, schedule, init_block, [], out_blocks))
    )

    stacked = sample_fn(init_states, keys)[0]
    final_states = np.array(stacked[:, -1, :])

    best_path: Optional[List[GridCoord]] = None
    best_len = float("inf")

    for chain_state in final_states:
        path = decode_flow_path(chain_state, meta)
        if path is not None and len(path) < best_len:
            best_path = path
            best_len = len(path)

    if best_path is not None:
        return best_path, meta

    if fallback_to_shortest:
        fallback = _shortest_path_bfs(maze, start, goal)
        return fallback, meta

    return None, meta


__all__ = [
    "N_STATES",
    "auto_start_goal",
    "build_flow_thrml_program",
    "decode_flow_path",
    "solve_maze_flow",
]


