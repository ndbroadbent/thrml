#!/usr/bin/env python3
"""Hybrid field + particle ionization pathfinder.

Combines:
- Excitable field dynamics (spreading, decay, reinforcement)
- Particles with velocity spawned from ionization events
- Particles leave trails as they move, creating directional tendrils
- Not blobs - lightning-like branching exploration!

Ionization events spawn particles with random velocity → directional exploration
Field dynamics reinforce connected trails → chains persist
Result: Lightning tendrils from terminals exploring and connecting!
"""
import torch as T
import torch.nn.functional as F
import time
import sys
import argparse
import warnings
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

warnings.filterwarnings('ignore', message='.*padding.*same.*even kernel.*')

device = T.device("cuda" if T.cuda.is_available() else "cpu")
console = Console()


def wait_for_space(console):
    """Wait for spacebar press."""
    import termios
    import tty

    console.print("[dim]SPACE=next Ctrl+C=exit[/dim]", end="")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x03':
            raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    console.print("\r" + " " * 30 + "\r", end="")


class ParticleSwarm:
    """Particles with position and velocity."""

    def __init__(self, max_particles, H, W, device):
        self.max_particles = max_particles
        self.H = H
        self.W = W
        self.device = device

        # [x, y, vx, vy] - float positions and velocities
        self.particles = T.zeros((max_particles, 4), device=device)
        self.active = T.zeros(max_particles, dtype=T.bool, device=device)
        self.lifetime = T.zeros(max_particles, device=device)
        self.deposit_amount = T.zeros(max_particles, device=device)  # Energy each particle deposits

    def spawn(self, positions, velocities, lifetimes, deposits):
        """Spawn particles at positions with given velocities and deposit amounts."""
        n = len(positions)
        if n == 0:
            return

        inactive = (~self.active).nonzero(as_tuple=True)[0]
        if len(inactive) < n:
            n = len(inactive)

        slots = inactive[:n]
        self.particles[slots, :2] = positions[:n]  # x, y
        self.particles[slots, 2:] = velocities[:n]  # vx, vy
        self.active[slots] = True
        self.lifetime[slots] = lifetimes[:n]
        self.deposit_amount[slots] = deposits[:n]

    def update(self, maze, field):
        """Move particles, deposit energy, check collisions."""
        if not self.active.any():
            return

        active_idx = self.active.nonzero(as_tuple=True)[0]

        # Move
        self.particles[active_idx, 0] += self.particles[active_idx, 2]  # x += vx
        self.particles[active_idx, 1] += self.particles[active_idx, 3]  # y += vy

        # Get grid positions
        px = self.particles[active_idx, 0].long().clamp(0, self.W - 1)
        py = self.particles[active_idx, 1].long().clamp(0, self.H - 1)

        # Deposit energy in field (using per-particle deposit amount!)
        for i, (y, x) in enumerate(zip(py, px)):
            if maze[y, x] == 0:  # Not a wall
                # Deposit based on particle's energy AND lifetime
                field[0, 0, y, x] += self.deposit_amount[active_idx[i]] * self.lifetime[active_idx[i]]

        # Check walls - stop particles immediately
        hit_wall = maze[py, px] == 1
        stopped = active_idx[hit_wall]
        self.active[stopped] = False

        # Age particles
        self.lifetime[active_idx] *= 0.95
        depleted = self.lifetime < 0.1
        self.active[depleted] = False


def hybrid_pathfinder(
    maze01,
    start,
    goal,
    steps=1000,
    max_particles=1000,
    ionization_rate=0.0005,  # Random ionization events
    terminal_emit=3,         # Particles from start/goal per frame
    speed_range=(0.2, 0.8),  # Particle speed
    deposit_range=(0.1, 0.4), # Energy deposited by particles (min, max)
    alpha=0.15,              # Field decay
    kappa=3.0,               # Reinforcement from neighbors
    gamma=0.05,              # Field spreading
    theta=0.2,               # Excitation threshold
    seed=0,
    visualize_fn=None,
):
    """
    Hybrid: Field dynamics + particle exploration

    - Random ionization events spawn particles with momentum
    - Particles explore creating directional tendrils (not blobs!)
    - Field dynamics reinforce connected regions
    - Start/goal emit particles and charge neighbors
    - Lightning-like branching emerges!
    """
    g = T.Generator(device=device).manual_seed(seed)
    H, W = maze01.shape

    open_mask = (1 - maze01).to(device).float()
    x = T.zeros((1, 1, H, W), device=device)  # Excitement field

    swarm = ParticleSwarm(max_particles, H, W, device)

    # 4-neighbor kernel for field reinforcement
    K4 = T.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=T.float32, device=device)[None, None]

    for t in range(steps):
        # === PARTICLE DYNAMICS ===

        # 1. Random ionization events spawn particles
        if ionization_rate > 0:
            ion_mask = (T.rand((H, W), generator=g, device=device) < ionization_rate) & (maze01 == 0)
            ion_mask = ion_mask & (x[0, 0] < 0.05)  # Only spawn in calm regions
            ion_pos = ion_mask.nonzero(as_tuple=False)

            if len(ion_pos) > 0:
                # Random velocities for ionization particles
                n = len(ion_pos)
                angles = T.rand(n, generator=g, device=device) * 2 * 3.14159
                speeds = speed_range[0] + T.rand(n, generator=g, device=device) * (speed_range[1] - speed_range[0])

                positions = T.stack([ion_pos[:, 1].float(), ion_pos[:, 0].float()], dim=1)  # x, y
                velocities = T.stack([T.cos(angles) * speeds, T.sin(angles) * speeds], dim=1)
                lifetimes = T.ones(n, device=device)
                # Random deposit amounts
                deposits = deposit_range[0] + T.rand(n, generator=g, device=device) * (deposit_range[1] - deposit_range[0])

                swarm.spawn(positions, velocities, lifetimes, deposits)

        # 2. Terminals emit particles TOWARD each other (directional, not spray)
        excited = (x > theta).float()

        # Direction from start to goal
        dx_to_goal = goal[1] - start[1]
        dy_to_goal = goal[0] - start[0]
        angle_to_goal = T.atan2(T.tensor(dy_to_goal, dtype=T.float32), T.tensor(dx_to_goal, dtype=T.float32)).item()

        # Emit from start (biased toward goal with some randomness)
        for _ in range(terminal_emit):
            # Random angle centered on direction to goal, with ±60° spread
            spread = (T.rand(1, generator=g, device=device).item() - 0.5) * 2.1  # ±60° ≈ 1.05 rad
            angle = angle_to_goal + spread
            speed = speed_range[0] + T.rand(1, generator=g, device=device).item() * (speed_range[1] - speed_range[0])

            pos = T.tensor([[start[1], start[0]]], dtype=T.float32, device=device)
            vel = T.tensor([[T.cos(T.tensor(angle)).item() * speed,
                           T.sin(T.tensor(angle)).item() * speed]], dtype=T.float32, device=device)
            dep = T.tensor([deposit_range[0] + T.rand(1, generator=g, device=device).item() * (deposit_range[1] - deposit_range[0])], device=device)
            swarm.spawn(pos, vel, T.tensor([1.5], device=device), dep)

        # Emit from goal (biased toward start)
        angle_to_start = angle_to_goal + 3.14159  # Opposite direction
        for _ in range(terminal_emit):
            spread = (T.rand(1, generator=g, device=device).item() - 0.5) * 2.1
            angle = angle_to_start + spread
            speed = speed_range[0] + T.rand(1, generator=g, device=device).item() * (speed_range[1] - speed_range[0])

            pos = T.tensor([[goal[1], goal[0]]], dtype=T.float32, device=device)
            vel = T.tensor([[T.cos(T.tensor(angle)).item() * speed,
                           T.sin(T.tensor(angle)).item() * speed]], dtype=T.float32, device=device)
            dep = T.tensor([deposit_range[0] + T.rand(1, generator=g, device=device).item() * (deposit_range[1] - deposit_range[0])], device=device)
            swarm.spawn(pos, vel, T.tensor([1.5], device=device), dep)

        # 3. Update particles - they deposit energy and create trails
        swarm.update(maze01, x)

        # === FIELD DYNAMICS ===

        # Neighbor reinforcement (keeps chains alive)
        N = F.conv2d(excited, K4, padding="same") * open_mask
        reinforce = kappa * N

        # Decay with reinforcement slowdown
        leak = alpha / (1.0 + reinforce)
        leak = T.maximum(leak, T.tensor(alpha * 0.15, device=device))  # Min decay

        # Field spreading from excited neighbors
        spread = gamma * N

        # Update field
        x = (1 - leak) * x + spread
        x = x.clamp(0, 1) * open_mask

        # Keep terminals alive
        x[0, 0, start[0], start[1]] = T.maximum(x[0, 0, start[0], start[1]], T.tensor(0.3, device=device))
        x[0, 0, goal[0], goal[1]] = T.maximum(x[0, 0, goal[0], goal[1]], T.tensor(0.3, device=device))

        # Visualization
        if visualize_fn:
            visualize_fn(t, x, swarm)

    return None


def render_terminal(t, x, swarm, maze01, start, goal, H, W):
    """Render field + particles."""

    excitement = x[0, 0].cpu().numpy()

    lines = []

    for i in range(H):
        row = ""
        for j in range(W):
            if (i, j) == start:
                row += "[green]██[/green]"
            elif (i, j) == goal:
                row += "[red]██[/red]"
            elif maze01[i, j] == 1:
                row += "░░"
            else:
                exc = excitement[i, j]

                if exc < 0.02:
                    row += "[dim]██[/dim]"
                elif exc < 0.08:
                    row += "[rgb(0,0,80)]██[/rgb(0,0,80)]"
                elif exc < 0.15:
                    row += "[rgb(0,0,120)]██[/rgb(0,0,120)]"
                elif exc < 0.3:
                    row += "[blue]██[/blue]"
                elif exc < 0.5:
                    row += "[cyan]██[/cyan]"
                elif exc < 0.7:
                    row += "[bright_cyan]██[/bright_cyan]"
                elif exc < 0.9:
                    row += "[white]██[/white]"
                else:
                    row += "[bright_white]██[/bright_white]"

        lines.append(row)

    active_p = int(swarm.active.sum())
    total = float(excitement.sum())
    max_exc = float(excitement.max())

    stats = f"Step {t:4d} | Particles: {active_p:3d} | Energy: {total:6.1f} | Max: {max_exc:.2f}"

    return Panel("\n".join(lines), subtitle=stats, border_style="bright_blue", expand=False)


def main():
    parser = argparse.ArgumentParser(description="⚛️  Particle + Field Hybrid Pathfinder")

    parser.add_argument('--size', type=int, default=25, help='Maze size (default: 25)')
    parser.add_argument('--obstacles', type=float, default=0.10, help='Obstacle density (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42, help='Maze seed (default: 42)')

    parser.add_argument('--steps', type=int, default=1500, help='Simulation steps (default: 1500)')
    parser.add_argument('--ionization', type=float, default=0.0005, help='Random ionization rate (default: 0.0005)')
    parser.add_argument('--emit', type=int, default=3, help='Particles from terminals per frame (default: 3)')
    parser.add_argument('--speed-min', type=float, default=0.2, help='Min particle speed (default: 0.2)')
    parser.add_argument('--speed-max', type=float, default=0.8, help='Max particle speed (default: 0.8)')
    parser.add_argument('--min-deposit', type=float, default=0.1, help='Min energy deposited by particles (default: 0.1)')
    parser.add_argument('--max-deposit', type=float, default=0.4, help='Max energy deposited by particles (default: 0.4)')
    parser.add_argument('--decay', type=float, default=0.15, help='Field decay rate (default: 0.15)')
    parser.add_argument('--kappa', type=float, default=3.0, help='Neighbor reinforcement (default: 3.0)')
    parser.add_argument('--gamma', type=float, default=0.05, help='Field spread rate (default: 0.05)')

    parser.add_argument('--fps', type=int, default=10, help='Steps per second (default: 10)')
    parser.add_argument('--debugger', action='store_true', help='Step-through mode (SPACE to advance)')

    args = parser.parse_args()

    console.clear()
    console.print(f"\n[bold cyan]⚛️  Particle Ionization Pathfinder {'[DEBUG]' if args.debugger else ''} ⚛️[/bold cyan]\n")

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
    console.print(f"Ionization: {args.ionization} | Emit: {args.emit} particles/frame")
    console.print(f"Speed: {args.speed_min}-{args.speed_max} | Deposit: {args.min_deposit}-{args.max_deposit} | Decay: {args.decay}\n")

    time.sleep(1)

    if args.debugger:
        def viz(t, x, swarm):
            panel = render_terminal(t, x, swarm, maze, start, goal, H, W)
            console.clear()
            console.print(panel)
            wait_for_space(console)

        try:
            hybrid_pathfinder(
                maze, start, goal,
                steps=args.steps,
                max_particles=2000,
                ionization_rate=args.ionization,
                terminal_emit=args.emit,
                speed_range=(args.speed_min, args.speed_max),
                deposit_range=(args.min_deposit, args.max_deposit),
                alpha=args.decay,
                kappa=args.kappa,
                gamma=args.gamma,
                theta=0.2,
                seed=int(time.time()) % 1000,
                visualize_fn=viz
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Exited[/yellow]")
    else:
        step_delay = 1.0 / args.fps

        with Live(console=console, refresh_per_second=30) as live:
            def viz(t, x, swarm):
                panel = render_terminal(t, x, swarm, maze, start, goal, H, W)
                live.update(panel)
                time.sleep(step_delay)

            try:
                hybrid_pathfinder(
                    maze, start, goal,
                    steps=args.steps,
                    max_particles=2000,
                    ionization_rate=args.ionization,
                    terminal_emit=args.emit,
                    speed_range=(args.speed_min, args.speed_max),
                    deposit_range=(args.min_deposit, args.max_deposit),
                    alpha=args.decay,
                    kappa=args.kappa,
                    gamma=args.gamma,
                    theta=0.2,
                    seed=int(time.time()) % 1000,
                    visualize_fn=viz
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")

    console.print(f"\n[dim]Lightning tendrils create branching exploration![/dim]")


if __name__ == "__main__":
    main()
