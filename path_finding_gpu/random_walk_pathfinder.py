#!/usr/bin/env python3
"""Random walk pathfinder with backpropagation.

Active particles do random walks leaving decaying trails.
When two particles meet → entire chains light up!
Particles race back to tail ends and continue exploring.

Like neural growth cones finding synaptic connections! 🧠
"""
import torch as T
import time
import sys
import argparse
import warnings
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

warnings.filterwarnings('ignore')

device = T.device("cuda" if T.cuda.is_available() else "cpu")
console = Console()


def wait_for_space(console):
    """Wait for spacebar."""
    import termios, tty
    console.print("[dim]SPACE=next Ctrl+C=exit[/dim]", end="")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x03': raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    console.print("\r" + " " * 30 + "\r", end="")


class Walker:
    """Single random walker with position and trail."""
    def __init__(self, pos, energy, particle_id, is_bastion=False):
        self.pos = pos  # (row, col)
        self.energy = energy
        self.trail = [pos]  # History of positions
        self.alive = True
        self.id = particle_id
        self.direction = None  # Last direction moved (dr, dc)
        self.is_bastion = is_bastion  # Spawned from terminal


def flood_fill_connected_with_distance(field, start_pos, threshold, H, W):
    """
    Find all cells connected to start_pos with energy > threshold.
    Returns dict: {pos: distance_from_start}
    """
    visited = set()
    to_visit = [(start_pos, 0)]  # (position, distance)
    connected = {}

    while to_visit:
        pos, dist = to_visit.pop(0)
        if pos in visited:
            continue
        visited.add(pos)

        if field[pos] > threshold:
            connected[pos] = dist
            # Check 4 neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    to_visit.append(((nr, nc), dist + 1))

    return connected


def random_walk_pathfinder(
    maze01,
    start,
    goal,
    steps=2000,
    bastion_rate=0.05,              # Probability per frame terminals emit walker
    random_spawn_rate=0.0003,       # Random walkers appear
    trail_decay=0.02,               # Trail decay per frame (regular trails)
    bastion_decay=0.005,            # Trail decay for bastion trails (slower!)
    particle_halflife=200,          # Steps before particle dies (regular)
    bastion_halflife=600,           # Steps before bastion particle dies (longer!)
    backprop_threshold=0.1,         # Min energy to connect to regular trail
    bastion_threshold=0.05,         # Min energy to connect to bastion trail (easier!)
    backprop_falloff=0.02,          # Energy decrease per cell distance (regular)
    bastion_backprop_falloff=0.01,  # Energy decrease per cell (bastion - spreads further!)
    turn_chance=0.1,                # Probability to turn left/right instead of forward
    seed=0,
    visualize_fn=None,
):
    """
    Random walk with chain backpropagation.

    Walkers leave trails, when they meet, chains reinforce and walkers
    race to tail ends to continue exploring!
    """
    g = T.Generator(device=device).manual_seed(seed)
    H, W = maze01.shape

    # Energy field (trails left by walkers)
    field = T.zeros((H, W), device=device)

    # Bastion mask (1.0 where bastion trails exist, 0.0 for regular)
    bastion_mask = T.zeros((H, W), device=device)

    # Active walkers
    walkers = []
    next_id = [0]  # Counter for walker IDs

    # Initialize start and goal as permanent bastion sources
    field[start] = 1.0
    field[goal] = 1.0
    bastion_mask[start] = 1.0
    bastion_mask[goal] = 1.0

    for t in range(steps):
        # === SPAWN NEW WALKERS ===

        # 1. Bastion particles from terminals
        if T.rand(1, generator=g).item() < bastion_rate:
            # Spawn from start
            neighbors = get_free_neighbors(start, maze01, H, W)
            if neighbors:
                pos = neighbors[T.randint(0, len(neighbors), (1,), generator=g).item()]
                walkers.append(Walker(pos, 1.0, next_id[0], is_bastion=True))
                next_id[0] += 1
                field[pos] = 1.0
                bastion_mask[pos] = 1.0

        if T.rand(1, generator=g).item() < bastion_rate:
            # Spawn from goal
            neighbors = get_free_neighbors(goal, maze01, H, W)
            if neighbors:
                pos = neighbors[T.randint(0, len(neighbors), (1,), generator=g).item()]
                walkers.append(Walker(pos, 1.0, next_id[0], is_bastion=True))
                next_id[0] += 1
                field[pos] = 1.0
                bastion_mask[pos] = 1.0

        # 2. Random ionization spawns
        if random_spawn_rate > 0:
            ion_mask = (T.rand((H, W), generator=g, device=device) < random_spawn_rate)
            ion_mask = ion_mask & (maze01 == 0) & (field < 0.05)  # Only in calm free cells
            ion_pos = ion_mask.nonzero(as_tuple=False)

            for pos in ion_pos:
                walkers.append(Walker((pos[0].item(), pos[1].item()), 1.0, next_id[0]))
                next_id[0] += 1
                field[pos[0], pos[1]] = 1.0

        # === MOVE WALKERS ===

        new_walkers = []
        connections_made = []
        collided_walkers = set()  # Track which walkers collided
        occupied_positions = set()  # Track current walker positions
        cells_to_flash = []  # Collect cells to light up AFTER decay

        for walker in walkers:
            if not walker.alive:
                continue

            # Skip if already involved in collision
            if walker.id in collided_walkers:
                continue

            # Age walker (bastion walkers live longer!)
            walker.energy *= 0.997
            max_age = bastion_halflife if walker.is_bastion else particle_halflife
            if walker.energy < 0.1 or len(walker.trail) > max_age:
                walker.alive = False
                continue

            # === DIRECTIONAL MOVEMENT (not random!) ===

            # Determine preferred direction (away from tail)
            if walker.direction is None or len(walker.trail) < 2:
                # No momentum yet - pick random direction
                neighbors = get_free_neighbors(walker.pos, maze01, H, W)
                if not neighbors:
                    walker.alive = False
                    continue
                next_pos = neighbors[T.randint(0, len(neighbors), (1,), generator=g).item()]
                walker.direction = (next_pos[0] - walker.pos[0], next_pos[1] - walker.pos[1])
            else:
                # Has direction - prefer continuing forward
                dr, dc = walker.direction

                # Should we turn?
                if T.rand(1, generator=g).item() < turn_chance:
                    # Turn left or right
                    if T.rand(1, generator=g).item() < 0.5:
                        # Turn left: (dr,dc) → (-dc, dr)
                        dr, dc = -dc, dr
                    else:
                        # Turn right: (dr,dc) → (dc, -dr)
                        dr, dc = dc, -dr

                # Try to move in preferred direction
                next_pos = (walker.pos[0] + dr, walker.pos[1] + dc)

                # Check if valid
                if not (0 <= next_pos[0] < H and 0 <= next_pos[1] < W) or maze01[next_pos] == 1:
                    # Hit wall - MUST turn
                    # Try left
                    dr_left, dc_left = -walker.direction[1], walker.direction[0]
                    left_pos = (walker.pos[0] + dr_left, walker.pos[1] + dc_left)

                    # Try right
                    dr_right, dc_right = walker.direction[1], -walker.direction[0]
                    right_pos = (walker.pos[0] + dr_right, walker.pos[1] + dc_right)

                    # Pick valid turn
                    if (0 <= left_pos[0] < H and 0 <= left_pos[1] < W and maze01[left_pos] == 0
                        and left_pos not in walker.trail):
                        next_pos = left_pos
                        dr, dc = dr_left, dc_left
                    elif (0 <= right_pos[0] < H and 0 <= right_pos[1] < W and maze01[right_pos] == 0
                          and right_pos not in walker.trail):
                        next_pos = right_pos
                        dr, dc = dr_right, dc_right
                    else:
                        # Can't turn, stuck
                        walker.alive = False
                        continue

                # Check if hitting own trail
                if next_pos in walker.trail:
                    walker.alive = False
                    continue

                # Update direction
                walker.direction = (dr, dc)

            # Check if position already occupied by another walker
            if next_pos in occupied_positions:
                # Can't move there, skip this walker's move
                new_walkers.append(walker)
                continue

            # CHECK FOR COLLISION
            # Hit ANY existing trail (not own) or another walker
            collision = False

            # Use appropriate threshold based on trail type
            threshold = bastion_threshold if bastion_mask[next_pos] > 0.5 else backprop_threshold

            # Check if stepping onto ANY trail (field energy > threshold, not own trail)
            if field[next_pos] > threshold and next_pos not in walker.trail:
                # Hit an existing trail!
                collision = True

                # First, add walker's trail to field
                for pos in walker.trail:
                    field[pos] = max(field[pos].item(), walker.energy)
                    if walker.is_bastion:
                        bastion_mask[pos] = 1.0
                field[next_pos] = max(field[next_pos].item(), walker.energy)

                # Now flood-fill to find ALL connected cells with distances
                connected_with_dist = flood_fill_connected_with_distance(
                    field, next_pos, threshold, H, W
                )

                # Collect cells to flash AFTER decay, with energy gradient
                # Use bastion falloff if walker is bastion (spreads further!)
                falloff = bastion_backprop_falloff if walker.is_bastion else backprop_falloff
                for pos, dist in connected_with_dist.items():
                    # Energy decreases with distance from collision
                    energy = max(1.0 - dist * falloff, threshold)
                    cells_to_flash.append((pos, energy))

                collided_walkers.add(walker.id)
                # Don't add to new_walkers - walker consumed by connection
                continue

            # Check if adjacent to another active walker head
            for other in walkers:
                if other.alive and other.id != walker.id and other.id not in collided_walkers:
                    if abs(next_pos[0] - other.pos[0]) + abs(next_pos[1] - other.pos[1]) == 1:
                        # Head-to-head collision
                        collision = True
                        collided_walkers.add(walker.id)
                        collided_walkers.add(other.id)

                        # Add both trails to field
                        for pos in walker.trail:
                            field[pos] = max(field[pos].item(), walker.energy)
                            if walker.is_bastion:
                                bastion_mask[pos] = 1.0
                        for pos in other.trail:
                            field[pos] = max(field[pos].item(), other.energy)
                            if other.is_bastion:
                                bastion_mask[pos] = 1.0

                        # Use appropriate threshold
                        thresh = bastion_threshold if (walker.is_bastion or other.is_bastion) else backprop_threshold

                        # Flood fill from collision point to find ALL connected cells
                        connected_with_dist = flood_fill_connected_with_distance(
                            field, walker.pos, thresh, H, W
                        )

                        # Collect cells to flash AFTER decay, with energy gradient
                        # Use bastion falloff if either walker is bastion (spreads further!)
                        falloff = bastion_backprop_falloff if (walker.is_bastion or other.is_bastion) else backprop_falloff
                        for pos, dist in connected_with_dist.items():
                            energy = max(1.0 - dist * falloff, thresh)
                            cells_to_flash.append((pos, energy))
                        break

            if collision:
                continue  # Don't add to new_walkers

            # Move
            walker.pos = next_pos
            walker.trail.append(next_pos)
            field[next_pos] = walker.energy
            if walker.is_bastion:
                bastion_mask[next_pos] = 1.0
            occupied_positions.add(next_pos)  # Mark position as occupied

            new_walkers.append(walker)

        # Use the walkers that didn't collide
        walkers = new_walkers

        # === FIELD DECAY ===
        # Apply different decay rates for bastion vs regular trails
        decay_rate = bastion_mask * bastion_decay + (1.0 - bastion_mask) * trail_decay
        field = field * (1.0 - decay_rate)
        field = field.clamp(0, 1)

        # === FLASH CONNECTED NETWORKS (after decay!) ===
        for pos, energy in cells_to_flash:
            field[pos] = max(field[pos].item(), energy)

        # Keep terminals alive
        field[start] = T.maximum(field[start], T.tensor(0.8, device=device))
        field[goal] = T.maximum(field[goal], T.tensor(0.8, device=device))

        # Visualization
        if visualize_fn:
            visualize_fn(t, field, walkers)

    return None


def get_free_neighbors(pos, maze, H, W):
    """Get list of free neighbor positions."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < H and 0 <= c < W and maze[r, c] == 0:
            neighbors.append((r, c))
    return neighbors


def find_tail_end(trail, field, threshold):
    """Find furthest position in trail above threshold energy."""
    for pos in reversed(trail):
        if field[pos] >= threshold:
            return pos
    return None


def render_terminal(t, field, walkers, maze01, start, goal, H, W):
    """Render field with active walkers highlighted."""

    excitement = field.cpu().numpy()

    # Mark active walker positions
    active_pos = set()
    for w in walkers:
        if w.alive:
            active_pos.add(w.pos)

    lines = []

    for i in range(H):
        row = ""
        for j in range(W):
            if (i, j) in active_pos:
                # Active walker head - yellow
                row += "[yellow]██[/yellow]"
            elif (i, j) == start:
                row += "[green]██[/green]"
            elif (i, j) == goal:
                row += "[red]██[/red]"
            elif maze01[i, j] == 1:
                row += "░░"
            else:
                # Precise RGB gradient based on energy
                exc = excitement[i, j]

                if exc >= 1.0:
                    # Maximum - bright white
                    row += "[rgb(255,255,255)]██[/rgb(255,255,255)]"
                elif exc >= 0.96:
                    # Very high - bright cyan
                    row += "[rgb(0,255,255)]██[/rgb(0,255,255)]"
                elif exc >= 0.85:
                    # High - cyan
                    row += f"[rgb(0,{int(180 + (exc - 0.85) * 682)},{int(200 + (exc - 0.85) * 500)})]██[/rgb(0,{int(180 + (exc - 0.85) * 682)},{int(200 + (exc - 0.85) * 500)})]"
                elif exc >= 0.6:
                    # Medium-high - bright blue
                    r = int((exc - 0.6) * 0 / 0.25)
                    g = int(100 + (exc - 0.6) * 320)  # 100 → 180
                    b = int(200 + (exc - 0.6) * 220)  # 200 → 255
                    row += f"[rgb({r},{g},{b})]██[/rgb({r},{g},{b})]"
                elif exc >= 0.3:
                    # Medium - blue
                    r = 0
                    g = int(50 + (exc - 0.3) * 167)  # 50 → 100
                    b = int(150 + (exc - 0.3) * 167)  # 150 → 200
                    row += f"[rgb({r},{g},{b})]██[/rgb({r},{g},{b})]"
                elif exc >= 0.1:
                    # Low - dark blue
                    r = 0
                    g = int((exc - 0.1) * 250)  # 0 → 50
                    b = int(80 + (exc - 0.1) * 350)  # 80 → 150
                    row += f"[rgb({r},{g},{b})]██[/rgb({r},{g},{b})]"
                elif exc >= 0.01:
                    # Very low - very dark blue
                    r = 0
                    g = 0
                    b = int(30 + (exc - 0.01) * 556)  # 30 → 80
                    row += f"[rgb({r},{g},{b})]██[/rgb({r},{g},{b})]"
                else:
                    # No energy - deep dark blue (almost black)
                    row += "[rgb(0,0,30)]██[/rgb(0,0,30)]"

        lines.append(row)

    n_walkers = sum(1 for w in walkers if w.alive)
    total = float(excitement.sum())
    max_exc = float(excitement.max())

    stats = f"Step {t:4d} | Walkers: {n_walkers:3d} | Trail Energy: {total:6.1f} | Max: {max_exc:.2f}"

    return Panel("\n".join(lines), subtitle=stats, border_style="bright_blue", expand=False)


def main():
    parser = argparse.ArgumentParser(description="⚛️  Random Walk Pathfinder with Backpropagation")

    parser.add_argument('--size', type=int, default=42, help='Maze size (default: 42)')
    parser.add_argument('--obstacles', type=float, default=0.10, help='Obstacle density (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42, help='Maze seed (default: 42)')

    parser.add_argument('--steps', type=int, default=2000, help='Simulation steps (default: 2000)')
    parser.add_argument('--bastion-rate', type=float, default=0.05, help='Terminal spawn rate per frame (default: 0.05)')
    parser.add_argument('--random-rate', type=float, default=0.0003, help='Random walker spawn rate (default: 0.0003)')
    parser.add_argument('--decay', type=float, default=0.02, help='Trail decay rate regular (default: 0.02)')
    parser.add_argument('--bastion-decay', type=float, default=0.005, help='Trail decay rate bastion (default: 0.005)')
    parser.add_argument('--halflife', type=int, default=200, help='Particle lifetime steps regular (default: 200)')
    parser.add_argument('--bastion-halflife', type=int, default=600, help='Particle lifetime steps bastion (default: 600)')
    parser.add_argument('--backprop-threshold', type=float, default=0.1, help='Min energy to connect regular (default: 0.1)')
    parser.add_argument('--bastion-threshold', type=float, default=0.05, help='Min energy to connect bastion (default: 0.05)')
    parser.add_argument('--backprop-falloff', type=float, default=0.02, help='Energy falloff per cell regular (default: 0.02)')
    parser.add_argument('--bastion-backprop-falloff', type=float, default=0.01, help='Energy falloff per cell bastion (default: 0.01)')
    parser.add_argument('--turn-chance', type=float, default=0.1, help='Probability to turn instead of forward (default: 0.1)')

    parser.add_argument('--fps', type=int, default=60, help='Steps per second (default: 60)')
    parser.add_argument('--debugger', action='store_true', help='Step-through mode (SPACE to advance)')

    args = parser.parse_args()

    console.clear()
    console.print(f"\n[bold cyan]⚛️  Random Walk Pathfinder {'[DEBUG]' if args.debugger else ''} ⚛️[/bold cyan]\n")

    H = W = args.size
    maze = T.zeros((H, W))

    rng = T.Generator().manual_seed(args.seed)
    mask = T.rand((H, W), generator=rng) < args.obstacles
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (2, 2)
    goal = (H - 3, W - 3)
    maze[start] = 0
    maze[goal] = 0

    console.print(f"Maze: {H}×{W} | Obstacles: {int(maze.sum())}")
    console.print(f"Bastion: {args.bastion_rate} | Random: {args.random_rate} | Turn: {args.turn_chance}")
    console.print(f"Decay: {args.decay} | Halflife: {args.halflife} | Backprop: {args.backprop_threshold}\n")
    console.print("[dim]Legend: [/dim][yellow]██[/yellow][dim]=walker head | Trails: [/dim][rgb(255,255,255)]██[/rgb(255,255,255)][dim]→[/dim][rgb(0,255,255)]██[/rgb(0,255,255)][dim]→blue→dark[/dim]\n")

    time.sleep(1)

    if args.debugger:
        def viz(t, field, walkers):
            panel = render_terminal(t, field, walkers, maze, start, goal, H, W)
            console.clear()
            console.print(panel)
            wait_for_space(console)

        try:
            random_walk_pathfinder(
                maze, start, goal,
                steps=args.steps,
                bastion_rate=args.bastion_rate,
                random_spawn_rate=args.random_rate,
                trail_decay=args.decay,
                bastion_decay=args.bastion_decay,
                particle_halflife=args.halflife,
                bastion_halflife=args.bastion_halflife,
                backprop_threshold=args.backprop_threshold,
                bastion_threshold=args.bastion_threshold,
                backprop_falloff=args.backprop_falloff,
                bastion_backprop_falloff=args.bastion_backprop_falloff,
                turn_chance=args.turn_chance,
                seed=int(time.time()) % 1000,
                visualize_fn=viz
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Exited[/yellow]")
    else:
        step_delay = 1.0 / args.fps

        with Live(console=console, refresh_per_second=30) as live:
            def viz(t, field, walkers):
                panel = render_terminal(t, field, walkers, maze, start, goal, H, W)
                live.update(panel)
                time.sleep(step_delay)

            try:
                random_walk_pathfinder(
                    maze, start, goal,
                    steps=args.steps,
                    bastion_rate=args.bastion_rate,
                    random_spawn_rate=args.random_rate,
                    trail_decay=args.decay,
                    bastion_decay=args.bastion_decay,
                    particle_halflife=args.halflife,
                    bastion_halflife=args.bastion_halflife,
                    backprop_threshold=args.backprop_threshold,
                    bastion_threshold=args.bastion_threshold,
                    backprop_falloff=args.backprop_falloff,
                    bastion_backprop_falloff=args.bastion_backprop_falloff,
                    turn_chance=args.turn_chance,
                    seed=int(time.time()) % 1000,
                    visualize_fn=viz
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")

if __name__ == "__main__":
    main()

