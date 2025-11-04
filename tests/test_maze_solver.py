"""Tests for THRML maze solver."""
import numpy as np
import pytest


def test_state_bits_structure():
    """Test that STATE_BITS has correct structure."""
    from thrml_maze_solver import STATE_BITS, N_STATES, DEG

    # Should have 11 states (0=empty, 1-4=single, 5-10=double)
    assert STATE_BITS.shape == (11, 4), "STATE_BITS should be 11x4"
    assert N_STATES == 11, "N_STATES should be 11"

    # Empty state should have all zeros
    assert np.all(STATE_BITS[0] == 0), "State 0 should be empty"

    # Degrees should match bit counts
    assert np.array_equal(DEG, STATE_BITS.sum(axis=1)), "DEG should match sum of bits"

    # Degree 0: only state 0
    assert DEG[0] == 0

    # Degree 1: states 1-4
    for i in range(1, 5):
        assert DEG[i] == 1, f"State {i} should have degree 1"

    # Degree 2: states 5-10
    for i in range(5, 11):
        assert DEG[i] == 2, f"State {i} should have degree 2"


def test_pair_matrix_horizontal():
    """Test horizontal pair matrix enforces E-W consistency."""
    from thrml_maze_solver import _pair_matrix, STATE_BITS

    lambda_edge = 0.5
    hard = 1e6
    W = _pair_matrix(lambda_edge, hard, axis='h')

    # Should be square matrix of N_STATES x N_STATES
    assert W.shape == (11, 11)

    # Check a few specific pairs
    # State 0 (empty) to State 0 (empty): both E and W are 0, should match
    assert W[0, 0] == 0.0, "Empty-Empty should have 0 weight"

    # State 2 (E) to State 4 (W): left.E=1, right.W=1, should match with penalty
    assert W[2, 4] == -lambda_edge, "E-W connection should have -lambda_edge weight"

    # State 2 (E) to State 0 (empty): left.E=1, right.W=0, should not match
    assert W[2, 0] == -hard, "Mismatched edges should have -hard weight"


def test_pair_matrix_vertical():
    """Test vertical pair matrix enforces N-S consistency."""
    from thrml_maze_solver import _pair_matrix, STATE_BITS

    lambda_edge = 0.5
    hard = 1e6
    W = _pair_matrix(lambda_edge, hard, axis='v')

    # State 3 (S) to State 1 (N): top.S=1, bottom.N=1, should match with penalty
    assert W[3, 1] == -lambda_edge, "S-N connection should have -lambda_edge weight"

    # State 3 (S) to State 0 (empty): top.S=1, bottom.N=0, should not match
    assert W[3, 0] == -hard, "Mismatched edges should have -hard weight"


def test_decode_path_simple():
    """Test path decoding on a simple straight path."""
    from thrml_maze_solver import _decode_path, STATE_BITS

    # Create a simple 3x3 grid with a straight path from (0,0) to (2,0)
    coords = [(i, j) for i in range(3) for j in range(3)]
    maze = np.zeros((3, 3), dtype=np.uint8)

    # State vector: path goes (0,0) -> (1,0) -> (2,0)
    # (0,0): State 3 (S) - degree 1 endpoint
    # (1,0): State 6 (NS) - degree 2 corridor
    # (2,0): State 1 (N) - degree 1 endpoint
    # All others: State 0 (empty)
    states = np.zeros(9, dtype=np.uint8)
    states[0] = 3  # (0,0): S
    states[3] = 6  # (1,0): NS
    states[6] = 1  # (2,0): N

    start = (0, 0)
    goal = (2, 0)

    path = _decode_path(states, coords, start, goal, maze)

    assert path is not None, "Should find a valid path"
    assert len(path) == 3, "Path should have 3 cells"
    assert path[0] == start, "Path should start at start"
    assert path[-1] == goal, "Path should end at goal"
    assert path[1] == (1, 0), "Path should go through (1,0)"


def test_decode_path_invalid_degree():
    """Test that path decoding rejects invalid degrees."""
    from thrml_maze_solver import _decode_path

    coords = [(i, j) for i in range(3) for j in range(3)]
    maze = np.zeros((3, 3), dtype=np.uint8)

    # Invalid: start has degree 2
    states = np.zeros(9, dtype=np.uint8)
    states[0] = 6  # (0,0): NS - degree 2, invalid for start
    states[3] = 6  # (1,0): NS
    states[6] = 1  # (2,0): N

    path = _decode_path(states, coords, (0, 0), (2, 0), maze)
    assert path is None, "Should reject path with invalid start degree"


def test_decode_path_mismatch():
    """Test that path decoding rejects mismatched edges."""
    from thrml_maze_solver import _decode_path

    coords = [(i, j) for i in range(3) for j in range(3)]
    maze = np.zeros((3, 3), dtype=np.uint8)

    # Mismatch: (0,0) has S but (1,0) doesn't have N
    states = np.zeros(9, dtype=np.uint8)
    states[0] = 3  # (0,0): S
    states[3] = 0  # (1,0): empty - no N to match
    states[6] = 1  # (2,0): N

    path = _decode_path(states, coords, (0, 0), (2, 0), maze)
    assert path is None, "Should reject path with mismatched edges"


def test_simple_maze_solvable():
    """Test solving a simple open maze."""
    from thrml_maze_solver import solve_with_thrml

    # 5x5 completely open maze
    maze = np.zeros((5, 5), dtype=np.uint8)
    start = (0, 0)
    goal = (4, 4)

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=5.0,
        lambda_edge=0.05,
        n_chains=256,  # More chains with soft constraints
        warmup=800,    # More warmup for exploration
        samples=50,
        steps_per_sample=15
    )

    # Should find a path
    assert path is not None, "Should find a path in open maze"
    assert len(path) >= 9, "Path should be at least 9 steps (Manhattan distance)"
    assert path[0] == start, "Path should start at start"
    assert path[-1] == goal, "Path should end at goal"

    # Path should be valid (each step is adjacent)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dist = abs(r1 - r2) + abs(c1 - c2)
        assert dist == 1, f"Path steps should be adjacent: {path[i]} -> {path[i+1]}"


def test_maze_with_wall():
    """Test solving a maze with a small obstacle."""
    from thrml_maze_solver import solve_with_thrml

    # 6x6 maze with a single wall cell
    maze = np.zeros((6, 6), dtype=np.uint8)
    maze[2, 2] = 1  # Single wall obstacle

    start = (0, 0)
    goal = (5, 5)

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=5.0,
        lambda_edge=0.05,
        n_chains=256,
        warmup=800,
        samples=50,
        steps_per_sample=15
    )

    # Should find a path (going around the single wall cell)
    assert path is not None, "Should find a path around wall"
    assert path[0] == start
    assert path[-1] == goal

    # Path should not go through walls
    for r, c in path:
        assert maze[r, c] == 0, f"Path should not go through wall at {(r, c)}"


def test_unsolvable_maze():
    """Test that unsolvable maze returns None."""
    from thrml_maze_solver import solve_with_thrml

    # 5x5 maze with goal completely walled off
    maze = np.zeros((5, 5), dtype=np.uint8)
    maze[3, :] = 1  # Horizontal wall

    start = (1, 2)
    goal = (4, 2)  # Below the wall

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=6.0,
        lambda_edge=0.35,
        n_chains=32,
        warmup=100,
        samples=10,
        steps_per_sample=4
    )

    # Should not find a path
    assert path is None, "Should return None for unsolvable maze"


def test_build_thrml_program():
    """Test that build_thrml_program creates proper structures."""
    from thrml_maze_solver import build_thrml_program

    maze = np.zeros((4, 4), dtype=np.uint8)
    maze[1, 1] = 1  # One wall
    start = (0, 0)
    goal = (3, 3)

    prog, spec, nodes, coords, G = build_thrml_program(
        maze, start, goal,
        beta=5.0,
        lambda_edge=0.2,
        hard=1e6
    )

    # Should have created nodes for all cells
    assert len(nodes) == 16, "Should have 16 nodes for 4x4 grid"
    assert len(coords) == 16, "Should have 16 coordinates"

    # Should have 2 free blocks (bipartite coloring)
    assert len(spec.free_blocks) == 2, "Should have 2 free blocks"

    # Total nodes in blocks should equal grid size
    total_nodes = sum(len(block.nodes) for block in spec.free_blocks)
    assert total_nodes == 16, "All nodes should be in free blocks"

    # Graph should have grid structure
    assert G.number_of_nodes() == 16
    # 4x4 grid has (4-1)*4 + 4*(4-1) = 12 + 12 = 24 edges
    assert G.number_of_edges() == 24


def test_start_goal_same_position():
    """Test edge case where start equals goal."""
    from thrml_maze_solver import solve_with_thrml

    maze = np.zeros((3, 3), dtype=np.uint8)
    start = (1, 1)
    goal = (1, 1)

    path, state_vec, coords = solve_with_thrml(
        maze, start, goal,
        beta=6.0,
        lambda_edge=0.35,
        n_chains=16,
        warmup=50,
        samples=5,
        steps_per_sample=2
    )

    # Path should either be [start] or None (depends on interpretation)
    # For now, let's just check it doesn't crash
    assert path is None or (len(path) == 1 and path[0] == start)


def test_state_bits_connectivity():
    """Test that state bits correctly encode connectivity."""
    from thrml_maze_solver import STATE_BITS

    # State 5 should be NE (bits [1,1,0,0])
    assert STATE_BITS[5, 0] == 1  # N
    assert STATE_BITS[5, 1] == 1  # E
    assert STATE_BITS[5, 2] == 0  # S
    assert STATE_BITS[5, 3] == 0  # W

    # State 6 should be NS (bits [1,0,1,0])
    assert STATE_BITS[6, 0] == 1  # N
    assert STATE_BITS[6, 1] == 0  # E
    assert STATE_BITS[6, 2] == 1  # S
    assert STATE_BITS[6, 3] == 0  # W

    # State 9 should be EW (bits [0,1,0,1])
    assert STATE_BITS[9, 0] == 0  # N
    assert STATE_BITS[9, 1] == 1  # E
    assert STATE_BITS[9, 2] == 0  # S
    assert STATE_BITS[9, 3] == 1  # W


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

