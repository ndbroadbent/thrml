"""Tests for the THRML flow-based maze solver."""

import numpy as np

from path_finding.thrml_flow_solver import (
    build_flow_thrml_program,
    decode_flow_path,
    generate_maze,
    render_flow_path_ascii,
    save_flow_path_png,
    solve_maze_flow,
)


def _simple_flow_vector(var_idx, edges_on):
    """Utility to build a binary flow vector with given active edges."""

    vec = np.zeros(len(var_idx), dtype=np.uint8)
    for edge in edges_on:
        vec[var_idx[edge]] = 1
    return vec


def test_build_flow_program_counts():
    """Basic sanity checks on the flow program structure."""

    maze = np.zeros((2, 2), dtype=np.uint8)
    start = (0, 0)
    goal = (1, 1)

    prog, spec, nodes, meta = build_flow_thrml_program(
        maze,
        start,
        goal,
        lam=1.0,
        rho_cons=4.0,
        rho_anti=4.0,
        beta=5.0,
    )

    # 2x2 open grid has 4 undirected edges -> 8 directed variables
    assert len(meta["dir_edges"]) == 8
    assert len(meta["var_idx"]) == 8
    assert len(nodes) == 8

    # Two update blocks (even/odd indexing)
    assert len(spec.free_blocks) == 2
    total_nodes = sum(len(block.nodes) for block in spec.free_blocks)
    assert total_nodes == len(nodes)

    # Metadata retains maze info
    assert meta["H"] == 2 and meta["W"] == 2
    assert meta["start"] == start
    assert meta["goal"] == goal


def test_decode_flow_path_simple():
    """Decoding should recover a simple path from active flows."""

    maze = np.zeros((2, 2), dtype=np.uint8)
    start = (0, 0)
    goal = (1, 1)

    prog, spec, nodes, meta = build_flow_thrml_program(
        maze,
        start,
        goal,
        lam=1.0,
        rho_cons=6.0,
        rho_anti=6.0,
        beta=5.0,
    )

    # Activate flows along (0,0)->(0,1)->(1,1)
    vec = _simple_flow_vector(
        meta["var_idx"],
        [
            ((0, 0), (0, 1)),
            ((0, 1), (1, 1)),
        ],
    )

    path = decode_flow_path(vec, meta)

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) == 3
    assert path == [(0, 0), (0, 1), (1, 1)]


def test_solve_maze_flow_simple():
    """Solver should find a Manhattan path on an open grid."""

    maze = np.zeros((4, 4), dtype=np.uint8)
    start = (0, 0)
    goal = (3, 3)

    path, meta = solve_maze_flow(
        maze,
        start=start,
        goal=goal,
        lam=1.0,
        rho_cons=12.0,
        rho_anti=12.0,
        beta=6.0,
        n_chains=24,
        warmup=120,
        samples=8,
        steps_per_sample=6,
        seed=123,
        fallback_to_shortest=True,
    )

    assert path is not None, "Expected a valid path"
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) >= 7  # Manhattan distance is 6 (needs 7 nodes)

    # All moves must be 4-neighbour steps and stay in free cells
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert maze[r2, c2] == 0
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_render_flow_path_ascii(tmp_path):
    """ASCII renderer should highlight walls, start/goal, and the path."""

    maze = np.zeros((3, 4), dtype=np.uint8)
    maze[1, 2] = 1  # a wall

    path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
    ascii_art = render_flow_path_ascii(maze, path, start=(0, 0), goal=(2, 3))

    lines = ascii_art.strip().splitlines()
    # Ensure grid dimensions
    assert len(lines) == 3
    assert all(len(line) == 4 for line in lines)

    assert lines[0][0] == "S"  # start
    assert lines[2][3] == "G"  # goal
    assert lines[1][2] == "#"  # wall
    assert "*" in ascii_art  # path markers

    # round-trip: write to file so manual demo can read it later
    ascii_file = tmp_path / "maze.txt"
    ascii_file.write_text(ascii_art)
    assert ascii_file.read_text() == ascii_art


def test_save_flow_path_png(tmp_path):
    """PNG renderer should create an image without errors."""

    maze = np.zeros((4, 4), dtype=np.uint8)
    path = [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3)]

    outfile = tmp_path / "flow.png"
    save_flow_path_png(maze, path, start=(0, 0), goal=(3, 3), outfile=outfile)

    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_generate_maze_variants():
    """Maze generator should produce diverse layouts with open endpoints."""

    for maze_type in ["random_walls", "dfs_perfect", "recursive_division"]:
        maze = generate_maze(25, maze_type=maze_type, seed=321)
        assert maze.shape == (25, 25)
        assert maze.dtype == np.uint8
        # Start/goal cells open
        assert maze[1, 1] == 0
        assert maze[-2, -2] == 0
        # There should be at least one wall besides the border
        assert np.any(maze[1:-1, 1:-1] == 1)


