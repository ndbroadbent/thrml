# THRML Maze Solver

An experimental maze solver using THRML (Thermodynamic Hypergraphical Model Library) that works on arbitrary-sized grids. It encodes path tiles as categorical variables, uses pairwise factors to force edge consistency between neighbors, unary factors to enforce degree constraints, and a per-edge penalty to prefer the shortest valid path.

## Features

- **Block Gibbs Sampling**: Uses efficient two-color block sampling on the grid graph
- **Categorical Variables**: Each cell is represented as one of 11 tile states (empty, single-ends, two-ends)
- **Constraint Enforcement**:
  - Walls forced to empty
  - Start/goal must have degree 1
  - Interior cells must be degree 0 or 2
  - Edge consistency between neighbors
- **Shortest Path Preference**: Small penalty per edge encourages minimal path length
- **Parallel Chains**: Runs multiple chains simultaneously for robustness

## Installation

Requires Python >= 3.10

```bash
pip install thrml networkx matplotlib numpy pytest
# JAX will be pulled in by thrml; for GPU use, follow JAX's wheel instructions.
```

Or using uv:

```bash
uv pip install thrml networkx matplotlib numpy pytest
```

For development:

```bash
git clone <repository>
cd thrml
pip install -e .
pip install -r requirements.txt
```

## Quick Usage

```python
import numpy as np
from thrml_maze_solver import solve_with_thrml

# Create a maze with obstacles (0=free, 1=wall)
maze = np.zeros((32, 32), dtype=np.uint8)
maze[10:20, 15] = 1  # Add a vertical wall

start = (1, 1)
goal = (30, 30)

# Solve - the distance potential creates a gradient guiding sampling toward goal
# (like water flowing downhill through channels!)
path, state_vec, coords = solve_with_thrml(
    maze, start, goal,
    beta=5.0,
    lambda_edge=0.05,
    n_chains=96,
    warmup=400,
    samples=30
)

if path:
    print(f"Found path of length {len(path)}")
    print(f"Path goes around obstacles naturally!")
else:
    print("No path found")
```

## Maze Format

Mazes are represented as NumPy arrays:
- Shape: `(H, W)` where H is height and W is width
- Values: `1` = wall, `0` = free space
- Coordinates: `(row, col)` tuples, 0-indexed
- Connectivity: Four-neighbor grid (N, E, S, W)

## Parameters

- `beta`: Inverse temperature for sampling (higher = more deterministic). Default: 5.0
- `lambda_edge`: Penalty per edge to prefer shorter paths. Default: 0.2
- `hard`: Large negative weight for forbidden states. Default: 1e6
- `n_chains`: Number of parallel sampling chains. Default: 64
- `warmup`: Number of warmup steps before collecting samples. Default: 300
- `samples`: Number of samples to collect. Default: 16
- `steps_per_sample`: Block Gibbs steps between samples. Default: 8

## Tuning Tips

- **No path found**: Lower `lambda_edge` (try 0.0) or increase `n_chains` and `warmup`
- **Want shorter paths**: Slightly increase `lambda_edge` (e.g., 0.1-0.2)
- **Slow convergence**: Increase `n_chains`, `warmup`, or `steps_per_sample`

## Known Limitations

This is an **experimental** solver demonstrating THRML's capabilities. With distance potentials, it now works well for:
- Open mazes and mazes with obstacles
- Finding valid paths through complex layouts
- Demonstrating energy-based modeling with gradient guidance

**Limitations:**
- **Optimality**: Paths found may not be shortest (though `lambda_edge` helps prefer shorter paths)
- **Stochastic**: Results vary between runs due to probabilistic sampling
- **Very Dense Obstacles**: Extremely constrained mazes may need more chains/warmup steps
- **Computational Cost**: Sampling can be slower than deterministic algorithms like A*

**Best For:**
- Understanding probabilistic graphical models
- Demonstrating thermodynamic computing concepts
- Problems where multiple solutions exist and any valid path is acceptable

For production pathfinding, traditional algorithms (A*, Dijkstra, BFS) are still faster and more reliable. This implementation demonstrates how energy-based models with proper potential fields can solve complex problems through local sampling - the key insight for thermodynamic computing!

## Running Tests

```bash
pytest tests/
```

Or with parallel execution:

```bash
pytest -n auto tests/
```

## Creating Animations

Watch the solver in action! Create a video showing the sampling process:

```bash
python animate_solver.py
```

This will:
1. Run the solver and capture intermediate sampling states
2. Generate frames showing the solution emerging over time
3. Use `ffmpeg` to create a video (requires: `brew install ffmpeg`)
4. Output: `maze_animation.mp4`

The animation shows:
- **White**: Free cells
- **Black**: Walls
- **Blue**: Active path tiles (darker = higher degree)
- **Yellow**: Valid path (when found)
- **Green**: Start
- **Red**: Goal

You can also create custom animations:

```python
from animate_solver import create_animation
import numpy as np

maze = np.zeros((32, 32), dtype=np.uint8)
# ... add obstacles ...

create_animation(
    maze, start=(0, 0), goal=(31, 31),
    output_video="my_maze.mp4",
    fps=10,
    n_chains=128,
    warmup=500,
    samples=60
)
```

## How It Works

### Tile Encoding

Each cell is a categorical variable with 11 possible states:
- State 0: Empty (no connections)
- States 1-4: Single ends (N, E, S, W)
- States 5-10: Two ends (NE, NS, NW, ES, EW, SW)

### Energy Function

The model uses three types of factors plus a **distance-based potential field**:

1. **Unary Factors**: Enforce local constraints + guide toward goal
   - Walls must be empty
   - Start/goal must have degree 1
   - Other cells must have degree 0 or 2
   - **Distance potential**: Cells closer to goal receive energy bonuses for being "active" (having connections)
     - Creates a gradient like water flowing downhill
     - Guides sampling naturally toward the goal

2. **Horizontal Pairwise Factors**: Ensure left.E == right.W
   - Apply `-lambda_edge` when edge is used
   - Large negative weight for mismatches

3. **Vertical Pairwise Factors**: Ensure top.S == bottom.N
   - Apply `-lambda_edge` when edge is used
   - Large negative weight for mismatches

### The "Water Flow" Effect

The distance potential creates an energy landscape where:
- Empty cells far from goal have low energy (neutral)
- Active path cells close to goal have high energy (favorable)
- This creates a natural gradient guiding the sampler toward the goal
- Like water flowing downhill through channels!

### Sampling Strategy

1. Build a grid graph over all cells
2. Perform bipartite coloring (2-coloring) for block updates
3. Initialize random states (respecting wall constraints)
4. Run block Gibbs sampling with warmup
5. Collect samples and decode paths
6. Return the shortest valid path found

## Examples

See `demo.py` for a complete example on a 64x64 grid with random obstacles.

## References

- [THRML Documentation](https://docs.thrml.ai/en/latest/)
- [THRML GitHub](https://github.com/extropic-ai/thrml)
- [Getting Started with THRML](https://docs.thrml.ai/en/latest/examples/00_probabilistic_computing/)

## License

MIT

