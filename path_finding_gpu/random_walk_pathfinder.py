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
    def __init__(self, pos, energy, particle_id):
        self.pos = pos  # (row, col)
        self.energy = energy
        self.trail = [pos]  # History of positions
        self.alive = True
        self.id = particle_id


def random_walk_pathfinder(
    maze01,
    start,
    goal,
    steps=2000,
    bastion_rate=0.05,      # Probability per frame terminals emit walker
    random_spawn_rate=0.0003,  # Random walkers appear
    trail_decay=0.02,        # Trail decay per frame
    particle_halflife=200,   # Steps before particle dies
    backprop_threshold=0.1,  # Min energy to race back to
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

    # Active walkers
    walkers = []
    next_id = [0]  # Counter for walker IDs

    # Initialize start and goal as permanent walkers (but they don't move)
    field[start] = 1.0
    field[goal] = 1.0

    for t in range(steps):
        # === SPAWN NEW WALKERS ===

        # 1. Bastion particles from terminals
        if T.rand(1, generator=g).item() < bastion_rate:
            # Spawn from start
            neighbors = get_free_neighbors(start, maze01, H, W)
            if neighbors:
                pos = neighbors[T.randint(0, len(neighbors), (1,), generator=g).item()]
                walkers.append(Walker(pos, 1.0, next_id[0]))
                next_id[0] += 1
                field[pos] = 1.0

        if T.rand(1, generator=g).item() < bastion_rate:
            # Spawn from goal
            neighbors = get_free_neighbors(goal, maze01, H, W)
            if neighbors:
                pos = neighbors[T.randint(0, len(neighbors), (1,), generator=g).item()]
                walkers.append(Walker(pos, 1.0, next_id[0]))
                next_id[0] += 1
                field[pos] = 1.0

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

        for walker in walkers:
            if not walker.alive:
                continue

            # Skip if already involved in collision
            if walker.id in collided_walkers:
                continue

            # Age walker
            walker.energy *= 0.997
            if walker.energy < 0.1 or len(walker.trail) > particle_halflife:
                walker.alive = False
                continue

            # Find valid moves (free neighbors with low energy)
            neighbors = get_free_neighbors(walker.pos, maze01, H, W)

            # Prefer low-energy neighbors (exploration, not retracing)
            valid = []
            for n in neighbors:
                if field[n] < 0.5:  # Don't walk on already-strong trails
                    valid.append(n)

            if not valid:
                # Stuck - try any neighbor
                valid = neighbors

            if not valid:
                walker.alive = False
                continue

            # Random step
            next_pos = valid[T.randint(0, len(valid), (1,), generator=g).item()]

            # Check if position already occupied by another walker
            if next_pos in occupied_positions:
                # Can't move there, skip this walker's move
                new_walkers.append(walker)
                continue

            # CHECK FOR COLLISION
            # 1. Head-to-head: stepping next to another active walker
            # 2. Head-to-tail: stepping onto ANOTHER walker's trail (not own!)
            collision = False
            collision_partner = None

            # Check if stepping onto existing trail (but NOT own trail!)
            if field[next_pos] > backprop_threshold and next_pos not in walker.trail:
                # Hit another walker's trail!
                for other in walkers:
                    if other.alive and other.id != walker.id and next_pos in other.trail:
                        collision = True
                        collision_partner = other
                        break

            # Check if adjacent to another active walker head
            if not collision:
                for other in walkers:
                    if other.alive and other.id != walker.id and other.id not in collided_walkers:
                        if abs(next_pos[0] - other.pos[0]) + abs(next_pos[1] - other.pos[1]) == 1:
                            collision = True
                            collision_partner = other
                            break

            if collision and collision_partner:
                collided_walkers.add(walker.id)
                collided_walkers.add(collision_partner.id)
                connections_made.append((walker, collision_partner))
                continue  # Handle collision later, don't add to new_walkers

            # Move
            walker.pos = next_pos
            walker.trail.append(next_pos)
            field[next_pos] = walker.energy
            occupied_positions.add(next_pos)  # Mark position as occupied

            new_walkers.append(walker)

        # Use the walkers that didn't collide
        walkers = new_walkers

        # === FIELD DECAY ===
        field = field * (1.0 - trail_decay)
        field = field.clamp(0, 1)

        # === HANDLE COLLISIONS (BACKPROPAGATION!) ===
        # Do this AFTER decay so the white flash is visible!

        for walker1, walker2 in connections_made:
            # Light up BOTH trails to 1.0!
            for pos in walker1.trail:
                field[pos] = 1.0
            for pos in walker2.trail:
                field[pos] = 1.0

            # The lit-up trails will be visible this frame as white
            # Next frame they'll decay like normal

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

    parser.add_argument('--size', type=int, default=25, help='Maze size (default: 25)')
    parser.add_argument('--obstacles', type=float, default=0.10, help='Obstacle density (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42, help='Maze seed (default: 42)')

    parser.add_argument('--steps', type=int, default=2000, help='Simulation steps (default: 2000)')
    parser.add_argument('--bastion-rate', type=float, default=0.05, help='Terminal spawn rate per frame (default: 0.05)')
    parser.add_argument('--random-rate', type=float, default=0.0003, help='Random walker spawn rate (default: 0.0003)')
    parser.add_argument('--decay', type=float, default=0.02, help='Trail decay rate (default: 0.02)')
    parser.add_argument('--halflife', type=int, default=200, help='Particle lifetime steps (default: 200)')
    parser.add_argument('--backprop-threshold', type=float, default=0.1, help='Min energy to race back to (default: 0.1)')

    parser.add_argument('--fps', type=int, default=10, help='Steps per second (default: 10)')
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
    console.print(f"Bastion: {args.bastion_rate} | Random: {args.random_rate} | Decay: {args.decay}")
    console.print(f"Halflife: {args.halflife} | Backprop threshold: {args.backprop_threshold}\n")
    console.print("[dim]Legend: [/dim][magenta]██[/magenta][dim]=active walker | Trails fade from white→blue→dark[/dim]\n")

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
                particle_halflife=args.halflife,
                backprop_threshold=args.backprop_threshold,
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
                    particle_halflife=args.halflife,
                    backprop_threshold=args.backprop_threshold,
                    seed=int(time.time()) % 1000,
                    visualize_fn=viz
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")

if __name__ == "__main__":
    main()

