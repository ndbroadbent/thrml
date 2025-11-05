#!/usr/bin/env python3
"""Particle-based ionization pathfinder.

Instead of field dynamics, simulate actual particles:
- Particles emitted from start/goal with random velocities
- Move through grid leaving ionization trails
- Stop immediately when hitting walls
- Trails decay over time
- When trails overlap between terminals → path emerges!

Like watching a cloud chamber! ☁️⚛️
"""
import torch as T
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
    """Wait for spacebar press in debugger mode."""
    import termios
    import tty

    console.print("[dim]Press SPACE for next frame, Ctrl+C to exit...[/dim]", end="")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x03':  # Ctrl+C
            raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    console.print("\r" + " " * 60 + "\r", end="")


class ParticleSwarm:
    """Manage a swarm of particles with position and velocity."""

    def __init__(self, max_particles, H, W, device):
        self.max_particles = max_particles
        self.H = H
        self.W = W
        self.device = device

        # Particle state: [max_particles, 4] = [x, y, vx, vy]
        # x, y are float positions, vx, vy are velocities
        self.particles = T.zeros((max_particles, 4), device=device)
        self.active = T.zeros(max_particles, dtype=T.bool, device=device)
        self.energy = T.zeros(max_particles, device=device)  # Particle energy

    def emit(self, pos, n_particles, speed_range, rng):
        """Emit particles from a position with random velocities."""
        # Find inactive slots
        inactive = (~self.active).nonzero(as_tuple=True)[0]
        if len(inactive) < n_particles:
            n_particles = len(inactive)

        if n_particles == 0:
            return

        slots = inactive[:n_particles]

        # Set positions
        self.particles[slots, 0] = pos[1]  # x (col)
        self.particles[slots, 1] = pos[0]  # y (row)

        # Random velocities
        angles = T.rand(n_particles, device=self.device, generator=rng) * 2 * 3.14159
        speeds = speed_range[0] + T.rand(n_particles, device=self.device, generator=rng) * (speed_range[1] - speed_range[0])

        self.particles[slots, 2] = T.cos(angles) * speeds  # vx
        self.particles[slots, 3] = T.sin(angles) * speeds  # vy

        self.active[slots] = True
        self.energy[slots] = 1.0

    def step(self, maze, trail_field, deposit_amount):
        """Move particles and deposit energy in trail field."""
        if not self.active.any():
            return

        # Move active particles
        active_mask = self.active

        # Update positions
        self.particles[active_mask, 0] += self.particles[active_mask, 2]  # x += vx
        self.particles[active_mask, 1] += self.particles[active_mask, 3]  # y += vy

        # Get integer grid positions
        px = self.particles[active_mask, 0].long().clamp(0, self.W - 1)
        py = self.particles[active_mask, 1].long().clamp(0, self.H - 1)

        # Check for walls or out of bounds
        hit_wall = maze[py, px] == 1

        # Deposit energy in trail (only if not hitting wall)
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        for idx, (i, j, wall) in enumerate(zip(py, px, hit_wall)):
            if not wall:
                trail_field[0, 0, i, j] += deposit_amount * self.energy[active_indices[idx]]

        # Deactivate particles that hit walls
        hit_indices = active_indices[hit_wall]
        self.active[hit_indices] = False

        # Deactivate particles that went out of bounds or decayed
        self.energy[active_mask] *= 0.98  # Particle energy decays
        depleted = self.energy < 0.01
        self.active[depleted] = False


def particle_pathfinder(
    maze01,
    start,
    goal,
    steps=1000,
    max_particles=2000,
    emit_rate=2,           # Particles emitted per frame from each terminal
    speed_range=(0.3, 1.5), # Min/max particle speed
    deposit=0.15,          # Energy deposited in trail
    decay=0.03,            # Trail decay rate
    seed=0,
    visualize_fn=None,
):
    """
    Particle-based pathfinder:
    - Emit particles from start and goal with random velocities
    - Particles leave ionization trails
    - Trails decay over time
    - Solution = bright connected trail between terminals
    """
    g = T.Generator(device=device).manual_seed(seed)
    H, W = maze01.shape

    # Trail field - accumulated ionization
    trail = T.zeros((1, 1, H, W), device=device)

    # Particle swarm
    swarm = ParticleSwarm(max_particles, H, W, device)

    for t in range(steps):
        # Emit particles from terminals
        swarm.emit(start, emit_rate, speed_range, g)
        swarm.emit(goal, emit_rate, speed_range, g)

        # Move particles and deposit energy
        swarm.step(maze01, trail, deposit)

        # Trail decay
        trail = trail * (1.0 - decay)
        trail = trail.clamp(0, 1)

        # Visualization
        if visualize_fn:
            visualize_fn(t, trail, swarm)

    return None  # Pure emergence - no path extraction!


def render_terminal(t, trail, swarm, maze01, start, goal, H, W):
    """Render particle trails."""

    excitement = trail[0, 0].cpu().numpy()

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

    # Stats
    total = float(excitement.sum())
    max_exc = float(excitement.max())
    active_particles = int(swarm.active.sum())

    stats = f"Step {t:4d} | Particles: {active_particles:3d} | Trail Energy: {total:6.1f} | Max: {max_exc:.2f}"

    content = "\n".join(lines)

    return Panel(
        content,
        subtitle=stats,
        border_style="bright_blue",
        expand=False
    )


def main():
    """Main entry."""
    parser = argparse.ArgumentParser(
        description="⚛️  Particle Ionization Pathfinder - Particles leave trails!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--size', type=int, default=25, help='Maze size (default: 25)')
    parser.add_argument('--obstacles', type=float, default=0.10, help='Obstacle density (default: 0.10)')
    parser.add_argument('--seed', type=int, default=42, help='Maze seed')

    parser.add_argument('--steps', type=int, default=1000, help='Simulation steps')
    parser.add_argument('--max-particles', type=int, default=2000, help='Max particles (default: 2000)')
    parser.add_argument('--emit', type=int, default=2, help='Particles emitted per frame (default: 2)')
    parser.add_argument('--speed-min', type=float, default=0.3, help='Min particle speed (default: 0.3)')
    parser.add_argument('--speed-max', type=float, default=1.5, help='Max particle speed (default: 1.5)')
    parser.add_argument('--deposit', type=float, default=0.15, help='Trail deposit amount (default: 0.15)')
    parser.add_argument('--decay', type=float, default=0.03, help='Trail decay rate (default: 0.03)')

    parser.add_argument('--fps', type=int, default=10, help='Steps per second (default: 10)')
    parser.add_argument('--debugger', action='store_true', help='Step-through mode')

    args = parser.parse_args()

    console.clear()
    console.print(f"\n[bold cyan]⚛️  Particle Ionization Pathfinder {'[DEBUG]' if args.debugger else ''} ⚛️[/bold cyan]\n")

    # Create maze
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

    console.print(f"Maze: {H}×{W} | Obstacles: {int(maze.sum())} | Device: {device}")
    console.print(f"Emit: {args.emit} particles/frame | Speed: {args.speed_min}-{args.speed_max}")
    console.print(f"Deposit: {args.deposit} | Decay: {args.decay}\n")
    console.print("[dim]Watch particles leave ionization trails![/dim]\n")

    time.sleep(1)

    if args.debugger:
        def viz(t, trail, swarm):
            panel = render_terminal(t, trail, swarm, maze, start, goal, H, W)
            console.clear()
            console.print(panel)
            wait_for_space(console)

        try:
            particle_pathfinder(
                maze, start, goal,
                steps=args.steps,
                max_particles=args.max_particles,
                emit_rate=args.emit,
                speed_range=(args.speed_min, args.speed_max),
                deposit=args.deposit,
                decay=args.decay,
                seed=int(time.time()) % 1000,
                visualize_fn=viz
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Exited[/yellow]")
    else:
        step_delay = 1.0 / args.fps

        with Live(console=console, refresh_per_second=30) as live:
            def viz(t, trail, swarm):
                panel = render_terminal(t, trail, swarm, maze, start, goal, H, W)
                live.update(panel)
                time.sleep(step_delay)

            try:
                particle_pathfinder(
                    maze, start, goal,
                    steps=args.steps,
                    max_particles=args.max_particles,
                    emit_rate=args.emit,
                    speed_range=(args.speed_min, args.speed_max),
                    deposit=args.deposit,
                    decay=args.decay,
                    seed=int(time.time()) % 1000,
                    visualize_fn=viz
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")

    console.print(f"\n[dim]Ionization trails show the explored space[/dim]")


if __name__ == "__main__":
    main()

