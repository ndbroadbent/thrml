"""Flow-based THRML maze solver built on directed edge variables.

This module encodes maze paths as unit flows from a start cell to a goal cell
and constructs a THRML energy program that enforces flow conservation and
prevents anti-parallel usage on edges. Sampling the resulting EBM encourages
shortest paths under 4-neighbour connectivity.
"""

from __future__ import annotations

from collections import deque
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode

try:  # pragma: no cover - optional dependency guard
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except Exception:  # pragma: no cover - fallback when matplotlib unavailable
    plt = None
    LinearSegmentedColormap = None


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


def _maze_random_walls(
    size: int,
    rng: np.random.Generator,
    obstacle_rate: float,
    add_mid_barrier: bool,
    edge_rate: float,
) -> np.ndarray:
    if size <= 2:
        return np.zeros((size, size), dtype=np.uint8)

    maze = np.zeros((size, size), dtype=np.uint8)
    interior = (rng.random((size - 2, size - 2)) < obstacle_rate).astype(np.uint8)
    maze[1:-1, 1:-1] = interior

    if edge_rate > 0.0:
        for idx in range(1, size - 1):
            if rng.random() < edge_rate:
                maze[0, idx] = 1
            if rng.random() < edge_rate:
                maze[-1, idx] = 1
            if rng.random() < edge_rate:
                maze[idx, 0] = 1
            if rng.random() < edge_rate:
                maze[idx, -1] = 1

    if add_mid_barrier and size >= 12:
        mid = size // 2
        maze[mid, :] = 1
        gap_cols = {
            max(1, size // 4),
            min(size - 2, (3 * size) // 4),
        }
        for gc in gap_cols:
            maze[mid, gc] = 0

    maze[1, 1] = 0
    maze[-2, -2] = 0
    maze[0, 0] = 0
    maze[-1, -1] = 0
    return maze


def _maze_perfect_dfs(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 2:
        return np.zeros((size, size), dtype=np.uint8)

    maze = np.ones((size, size), dtype=np.uint8)
    visited = np.zeros((size, size), dtype=bool)

    start_cell = (1, 1)
    stack = [start_cell]
    visited[start_cell] = True  # type: ignore[index]
    maze[start_cell] = 0  # type: ignore[index]

    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        ci, cj = stack[-1]
        candidates: List[Tuple[int, int, int, int]] = []
        for di, dj in directions:
            ni, nj = ci + di, cj + dj
            if 1 <= ni < size - 1 and 1 <= nj < size - 1 and not visited[ni, nj]:
                wi, wj = ci + di // 2, cj + dj // 2
                candidates.append((ni, nj, wi, wj))

        if candidates:
            ni, nj, wi, wj = candidates[int(rng.integers(len(candidates)))]
            visited[ni, nj] = True
            maze[wi, wj] = 0
            maze[ni, nj] = 0
            stack.append((ni, nj))
        else:
            stack.pop()

    maze[1, 1] = 0
    maze[-2, -2] = 0
    return maze


def _maze_recursive_division(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 2:
        return np.zeros((size, size), dtype=np.uint8)

    maze = np.zeros((size, size), dtype=np.uint8)
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    def divide(top: int, bottom: int, left: int, right: int) -> None:
        if bottom - top < 2 or right - left < 2:
            return

        horizontal = (bottom - top) >= (right - left)

        if horizontal:
            possible_rows = [r for r in range(top + 2, bottom, 2)]
            if not possible_rows:
                return
            row = int(rng.choice(possible_rows))
            maze[row, left:right + 1] = 1
            gap_options = [c for c in range(left + 1, right, 2)]
            if gap_options:
                gap = int(rng.choice(gap_options))
                maze[row, gap] = 0
            divide(top, row - 1, left, right)
            divide(row + 1, bottom, left, right)
        else:
            possible_cols = [c for c in range(left + 2, right, 2)]
            if not possible_cols:
                return
            col = int(rng.choice(possible_cols))
            maze[top:bottom + 1, col] = 1
            gap_options = [r for r in range(top + 1, bottom, 2)]
            if gap_options:
                gap = int(rng.choice(gap_options))
                maze[gap, col] = 0
            divide(top, bottom, left, col - 1)
            divide(top, bottom, col + 1, right)

    divide(0, size - 1, 0, size - 1)
    maze[1, 1] = 0
    maze[-2, -2] = 0
    return maze


def _has_path_bfs(maze: np.ndarray, start: GridCoord, goal: GridCoord) -> bool:
    if maze[start] == 1 or maze[goal] == 1:
        return False

    height, width = maze.shape
    queue = deque([start])
    visited = {start}

    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        r, c = node
        for nr, nc in neighbors4(r, c, height, width):
            nxt = (nr, nc)
            if maze[nxt] == 1 or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)

    return False


def generate_maze(
    size: int,
    *,
    maze_type: str = "random_walls",
    seed: int | None = None,
    obstacle_rate: float = 0.18,
    add_mid_barrier: bool = True,
    edge_rate: float = 0.35,
) -> np.ndarray:
    """Generate a binary maze (0=open, 1=wall) using different algorithms."""

    rng = np.random.default_rng(seed)
    base_seed = rng.integers(0, 2**32 - 1, dtype=np.uint32)

    maze_type = maze_type.lower()
    generators: Dict[str, callable] = {}

    generators.update({
        "random": lambda g: _maze_random_walls(size, g, obstacle_rate, add_mid_barrier, edge_rate),
        "random_walls": lambda g: _maze_random_walls(size, g, obstacle_rate, add_mid_barrier, edge_rate),
        "dfs": lambda g: _maze_perfect_dfs(size, g),
        "dfs_perfect": lambda g: _maze_perfect_dfs(size, g),
        "backtracker": lambda g: _maze_perfect_dfs(size, g),
        "division": lambda g: _maze_recursive_division(size, g),
        "recursive_division": lambda g: _maze_recursive_division(size, g),
        "rooms": lambda g: _maze_recursive_division(size, g),
    })

    if maze_type not in generators:
        raise ValueError(f"Unknown maze_type '{maze_type}'")

    generator = generators[maze_type]

    for attempt in itertools.count():
        sub_seed = int((base_seed + attempt) % (2**32 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        maze = generator(sub_rng)

        # Guarantee start/goal cells are open and adjacent entries carved.
        if size > 1:
            maze[0, 0] = 0
            maze[-1, -1] = 0
            if size > 2:
                maze[0, 1] = 0
                maze[1, 0] = 0
                maze[-1, -2] = 0
                maze[-2, -1] = 0
                maze[1, 1] = 0
                maze[-2, -2] = 0
                if size > 3:
                    maze[1, 2] = 0
                    maze[2, 1] = 0
                    maze[-2, -3] = 0
                    maze[-3, -2] = 0

        if maze_type in {"random", "random_walls"} and add_mid_barrier and size >= 12:
            mid = size // 2
            maze[mid, :] = 1
            gap_cols = {
                max(1, size // 4),
                min(size - 2, (3 * size) // 4),
            }
            for gc in gap_cols:
                maze[mid, gc] = 0
            maze[0, 0] = 0
            maze[-1, -1] = 0
            if size > 2:
                maze[0, 1] = 0
                maze[1, 0] = 0
                maze[-1, -2] = 0
                maze[-2, -1] = 0
                maze[1, 1] = 0
                maze[-2, -2] = 0
                if size > 3:
                    maze[1, 2] = 0
                    maze[2, 1] = 0
                    maze[-2, -3] = 0
                    maze[-3, -2] = 0

        if _has_path_bfs(maze, (0, 0), (size - 1, size - 1)):
            break

    return maze.astype(np.uint8, copy=False)

    # Guarantee start/goal cells are walkable.
    if size > 1:
        maze[0, 0] = 0
        maze[-1, -1] = 0
        if size > 2:
            maze[1, 1] = 0
            maze[-2, -2] = 0
            maze[0, 1] = 0
            maze[1, 0] = 0
            maze[-1, -2] = 0
            maze[-2, -1] = 0
            if size > 3:
                maze[2, 1] = 0
                maze[1, 2] = 0
                maze[-3, -2] = 0
                maze[-2, -3] = 0

    return maze.astype(np.uint8, copy=False)


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


def _edge_activation_statistics(
    stacked: np.ndarray,
    dir_edges: List[DirectedEdge],
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate edge activation probabilities and project them onto cells."""

    if stacked.size == 0:
        return np.zeros(len(dir_edges), dtype=np.float32), np.zeros((height, width), dtype=np.float32)

    edge_probs = stacked.mean(axis=(0, 1))
    cell_heat = np.zeros((height, width), dtype=np.float32)

    for prob, (u, v) in zip(edge_probs, dir_edges):
        cell_heat[u] += prob
        cell_heat[v] += prob

    max_val = float(cell_heat.max(initial=0.0))
    if max_val > 0:
        cell_heat = cell_heat / max_val

    return edge_probs.astype(np.float32), cell_heat.astype(np.float32)


def _ensure_matplotlib_ready() -> None:
    if plt is None or LinearSegmentedColormap is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "matplotlib is required for PNG output but is not available. "
            "Install matplotlib or use render_flow_path_ascii instead."
        )


def render_flow_path_ascii(
    maze: np.ndarray,
    path: Optional[Sequence[GridCoord]],
    *,
    start: Optional[GridCoord] = None,
    goal: Optional[GridCoord] = None,
    background: str = ".",
    path_char: str = "*",
    wall_char: str = "#",
) -> str:
    """Render an ASCII depiction of the maze with an optional path overlay."""

    maze = np.asarray(maze, dtype=np.uint8)
    height, width = maze.shape

    grid = np.full((height, width), background, dtype="U1")
    grid[maze == 1] = wall_char

    if path:
        for coord in path:
            if maze[coord] == 0:
                grid[coord] = path_char

    if start is None and path:
        start = path[0]
    if goal is None and path:
        goal = path[-1]

    if start is not None:
        grid[start] = "S"
    if goal is not None:
        grid[goal] = "G"

    return "\n".join("".join(grid[r]) for r in range(height))


def _default_cmap() -> LinearSegmentedColormap:
    _ensure_matplotlib_ready()
    colors = [
        (0.05, 0.05, 0.18),
        (0.1, 0.3, 0.6),
        (0.2, 0.7, 0.65),
        (0.85, 0.9, 0.25),
        (1.0, 0.6, 0.05),
    ]
    return LinearSegmentedColormap.from_list("thrml_flow", colors)


def save_flow_path_png(
    maze: np.ndarray,
    path: Optional[Sequence[GridCoord]],
    *,
    start: Optional[GridCoord] = None,
    goal: Optional[GridCoord] = None,
    outfile: str | Path = "thrml_flow.png",
    cell_heat: Optional[np.ndarray] = None,
    edge_probs: Optional[np.ndarray] = None,
    dir_edges: Optional[List[DirectedEdge]] = None,
    dpi: int = 160,
) -> Path:
    """Save a PNG visualizing the maze, sampling heatmap, and decoded path."""

    _ensure_matplotlib_ready()

    maze = np.asarray(maze, dtype=np.uint8)
    height, width = maze.shape

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    cmap = _default_cmap()
    base = np.ones((height, width), dtype=np.float32) * 0.1
    base[maze == 1] = -0.25

    if cell_heat is None:
        heat = np.zeros_like(base)
    else:
        heat = np.clip(cell_heat, 0.0, 1.0)

    overlay = np.clip(base + heat, -0.25, 1.0)

    fig, ax = plt.subplots(figsize=(width / 6.0, height / 6.0), dpi=dpi)
    ax.imshow(overlay, cmap=cmap, interpolation="nearest")
    ax.set_axis_off()

    if path:
        ys = [coord[0] for coord in path]
        xs = [coord[1] for coord in path]
        ax.plot(xs, ys, color="white", linewidth=3.4, alpha=0.97)
        ax.scatter(xs[0], ys[0], c="lime", s=120, edgecolors="black", zorder=6)
        ax.scatter(xs[-1], ys[-1], c="red", s=120, edgecolors="black", zorder=6)

    if edge_probs is not None and dir_edges is not None:
        for prob, (u, v) in zip(edge_probs, dir_edges):
            if prob <= 0.02:
                continue
            ux, uy = u[1], u[0]
            vx, vy = v[1], v[0]
            ax.plot(
                [ux, vx],
                [uy, vy],
                color=(1.0, 1.0, 1.0, min(0.75, 0.2 + float(prob) * 0.9)),
                linewidth=1.0,
            )

    fig.tight_layout(pad=0)
    fig.savefig(outfile, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return outfile


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
    seed: int = 123,
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
    stacked_np = np.array(stacked)
    final_states = stacked_np[:, -1, :]

    edge_probs, cell_heat = _edge_activation_statistics(
        stacked_np, meta["dir_edges"], maze.shape[0], maze.shape[1]
    )
    meta["edge_marginals"] = edge_probs
    meta["cell_heat"] = cell_heat

    best_path: Optional[List[GridCoord]] = None
    best_len = float("inf")

    for chain_state in final_states:
        path = decode_flow_path(chain_state, meta)
        if path is not None and len(path) < best_len:
            best_path = path
            best_len = len(path)

    if best_path is not None:
        meta["best_path"] = best_path
        return best_path, meta

    if fallback_to_shortest:
        fallback = _shortest_path_bfs(maze, start, goal)
        meta["best_path"] = fallback
        return fallback, meta

    meta["best_path"] = None
    return None, meta


__all__ = [
    "N_STATES",
    "auto_start_goal",
    "build_flow_thrml_program",
    "generate_maze",
    "render_flow_path_ascii",
    "save_flow_path_png",
    "decode_flow_path",
    "solve_maze_flow",
    "demo",
]


def demo(
    size: int = 41,
    obstacle_rate: float = 0.18,
    output_png: str | Path = "thrml_flow_demo.png",
    chains: int = 96,
    warmup: int = 500,
    samples: int = 24,
    seed: int = 17,
    maze_type: str = "random_walls",  # random_walls, dfs_perfect, recursive_division
    maze_seed: int | None = None,
) -> Tuple[Optional[List[GridCoord]], Dict[str, object]]:
    """Run a sampling demo, print ASCII output, and save a PNG visualisation."""

    if maze_seed is None:
        maze_seed = seed

    maze = generate_maze(
        size,
        maze_type=maze_type,
        seed=maze_seed,
        obstacle_rate=obstacle_rate,
        add_mid_barrier=True,
    )

    path, meta = solve_maze_flow(
        maze,
        lam=1.0,
        rho_cons=12.0,
        rho_anti=12.0,
        beta=9.0,
        n_chains=chains,
        warmup=warmup,
        samples=samples,
        steps_per_sample=8,
        seed=seed,
        fallback_to_shortest=True,
    )

    # Ensure start/goal remain open in case generation overlapped them.
    maze[meta["start"]] = 0
    maze[meta["goal"]] = 0

    ascii_art = render_flow_path_ascii(maze, path, start=meta["start"], goal=meta["goal"])
    print(ascii_art)

    try:
        save_flow_path_png(
            maze,
            path,
            start=meta["start"],
            goal=meta["goal"],
            outfile=output_png,
            cell_heat=meta.get("cell_heat"),
            edge_probs=meta.get("edge_marginals"),
            dir_edges=meta.get("dir_edges"),
        )
        print(f"Saved visualisation to {output_png}")
    except RuntimeError as exc:
        print(f"Skipping PNG generation: {exc}")

    return path, meta


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    demo()

