#!/usr/bin/env python3
"""GPU-accelerated ionization pathfinder with terminal visualization.

Like a cloud chamber or scintillation detector:
- Random ionization events (cosmic rays) excite cells
- Excited regions spread and reinforce each other
- Start/goal act as charged terminals creating directional fields
- Stable ionization trails emerge connecting the terminals

Watch particle physics solve mazes! ⚛️
"""
import torch as T
import torch.nn.functional as F
import networkx as nx
import time
import sys
import argparse
import warnings
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Suppress PyTorch padding warning for 2x2 kernels
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

    # Clear the prompt line
    console.print("\r" + " " * 60 + "\r", end="")


def build_kernels(device):
    """Build convolution kernels for neighbor counts and motif detection."""
    # 4-neighbor count
    K4 = T.tensor([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]], dtype=T.float32, device=device)[None, None]

    # I-shape (straight 3) - horizontal and vertical
    Kh = T.tensor([[1, 1, 1]], dtype=T.float32, device=device)[None, None]
    Kv = Kh.transpose(-1, -2)

    # L-shapes - four 2x2 corners
    L1 = T.tensor([[1, 0], [0, 1]], dtype=T.float32, device=device)[None, None]
    L2 = T.tensor([[0, 1], [1, 0]], dtype=T.float32, device=device)[None, None]

    return K4, Kh, Kv, L1, L2


def conv2(x, k):
    """2D convolution helper."""
    return F.conv2d(x, k, padding="same")


def excite_pathfinder(
    maze01,  # HxW tensor (0=open, 1=wall)
    start,   # (i, j)
    goal,    # (i, j)
    steps=600,
    alpha=0.06,      # Base decay rate
    kappa=1.6,       # Decay slowdown from reinforcement
    beta1=0.9,       # Neighbor reinforcement
    beta2=0.7,       # I-shape reinforcement
    beta3=0.7,       # L-shape reinforcement
    gamma=0.18,      # Excitation spread
    theta=0.35,      # Activation threshold
    eps_seed=1e-6,   # Only seed if excitement < this
    noise_rate=0.02, # Sparkle probability
    refractory_dec=0.02,  # Refractory decay
    fire_thr=0.6,    # Threshold to trigger refractory
    seed=0,
    visualize_fn=None,  # Optional callback for visualization
):
    """
    PURE emergent excitable medium - no classical algorithms!

    The path emerges naturally as a stable excited chain connecting start to goal.
    You SEE the solution as the bright connected cells - no extraction needed!
    """
    g = T.Generator(device=device).manual_seed(seed)
    H, W = maze01.shape

    open_mask = (1 - maze01).to(device).float()  # 1=open, 0=wall
    x = T.zeros((1, 1, H, W), device=device)
    r = T.zeros_like(x)

    # Source field: start and goal emit energy to excited neighbors ONLY
    # Like magnetic poles - they pull excited cells toward each other
    S = T.zeros_like(x)

    K4, Kh, Kv, L1, L2 = build_kernels(device)

    # Main loop
    for t in range(steps):
        # Binary activity map (excited = above threshold)
        a = (x > theta).float()

        # Compute reinforcement from neighbors and motifs
        N = conv2(a, K4) * open_mask  # Neighbor count
        I = T.maximum(conv2(a, Kh), conv2(a, Kv)) * open_mask  # I-shapes
        Lsum = T.maximum(conv2(a, L1), conv2(a, L2)) * open_mask  # L-shapes

        reinforce = beta1 * N + beta2 * I + beta3 * Lsum

        # Directional attraction: start and goal charge their excited neighbors
        # This creates magnetic-like pull between terminals through excited cells
        S.zero_()
        if a[0, 0, start[0], start[1]] > 0.5 or t < 10:
            # Start charges excited neighbors
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = start[0] + di, start[1] + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if a[0, 0, ni, nj] > 0.5:  # Only charge if neighbor is excited
                        S[0, 0, ni, nj] += 0.3

        if a[0, 0, goal[0], goal[1]] > 0.5 or t < 10:
            # Goal charges excited neighbors
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = goal[0] + di, goal[1] + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if a[0, 0, ni, nj] > 0.5:  # Only charge if neighbor is excited
                        S[0, 0, ni, nj] += 0.3

        # Keep terminals themselves alive
        S[0, 0, start[0], start[1]] = 0.2
        S[0, 0, goal[0], goal[1]] = 0.2

        # Fixed decay: always decay, reinforcement just slows it
        # Cells without reinforcement will decay to zero!
        leak = alpha / (1.0 + kappa * reinforce)
        # But ensure minimum decay even with max reinforcement
        leak = T.maximum(leak, T.tensor(alpha * 0.1, device=device))

        # Update excitement
        x = (1 - leak) * x + gamma * reinforce + S

        # Sparkles: seed only where fully collapsed and not refractory
        can_seed = ((x < eps_seed) & (r < 0.5) & (open_mask == 1)).float()

        if noise_rate > 0:
            noise = T.bernoulli(T.full_like(x, noise_rate), generator=g) * can_seed
            x = x + noise * 0.9  # Inject spark

        # Clip and enforce walls
        x = x.clamp(0, 1) * open_mask

        # Refractory update
        r = T.clamp(r - refractory_dec, 0, 1)
        r = T.where(x > fire_thr, T.ones_like(r), r)
        r = r * open_mask

        # Visualization callback - just show the raw state!
        if visualize_fn:
            visualize_fn(t, x, r, a)

    # No path extraction - the solution IS the visible excited chain!
    return None


def render_terminal(t, x, r, a, maze01, start, goal, H, W):
    """Render current state - PURE emergent visualization!

    No path extraction - just show the raw energy field.
    The solution emerges as bright connected cells like lightning!
    """

    # Get excitement values
    excitement = x[0, 0].cpu().numpy()
    active_mask = a[0, 0].cpu().numpy()

    # Build display
    lines = []

    for i in range(H):
        row = ""
        for j in range(W):
            if (i, j) == start:
                # Start - green square (charging terminal)
                row += "[green]██[/green]"
            elif (i, j) == goal:
                # Goal - red square (charging terminal)
                row += "[red]██[/red]"
            elif maze01[i, j] == 1:
                # Wall - stippled pattern
                row += "░░"
            else:
                # Free cell - smooth color gradient by excitement level
                # Dark gray → very dark blue → dark blue → blue → cyan → bright cyan → white
                exc = excitement[i, j]

                if exc < 0.02:
                    # No energy - dark gray
                    row += "[dim]██[/dim]"
                elif exc < 0.08:
                    # Very very low - very dark blue
                    row += "[rgb(0,0,80)]██[/rgb(0,0,80)]"
                elif exc < 0.15:
                    # Very low - dark blue
                    row += "[rgb(0,0,120)]██[/rgb(0,0,120)]"
                elif exc < 0.3:
                    # Low - blue
                    row += "[blue]██[/blue]"
                elif exc < 0.5:
                    # Medium low - cyan
                    row += "[cyan]██[/cyan]"
                elif exc < 0.7:
                    # Medium - bright cyan
                    row += "[bright_cyan]██[/bright_cyan]"
                elif exc < 0.9:
                    # High - white
                    row += "[white]██[/white]"
                else:
                    # Maximum sparkle - bright white
                    row += "[bright_white]██[/bright_white]"

        lines.append(row)

    # Stats
    total_excitement = float(excitement.sum())
    max_excitement = float(excitement.max())
    active_cells = int(active_mask.sum())

    stats = f"Step {t:4d} | Active: {active_cells:3d} | Total Energy: {total_excitement:6.1f} | Max: {max_excitement:.2f}"

    content = "\n".join(lines)

    return Panel(
        content,
        subtitle=stats,
        border_style="bright_blue",
        expand=False
    )


def demo_terminal_ui():
    """Demo with live terminal visualization."""
    console.clear()
    console.print("\n[bold cyan]🎇 GPU-Accelerated Sparkle Pathfinder 🎇[/bold cyan]\n")

    # Create maze
    H = W = 30
    maze = T.zeros((H, W))

    # Add some obstacles
    rng = T.Generator().manual_seed(42)
    mask = T.rand((H, W), generator=rng) < 0.12
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    maze[mask] = 1

    start = (2, 2)
    goal = (H - 3, W - 3)
    maze[start] = 0
    maze[goal] = 0

    console.print(f"Maze: {H}×{W} | Obstacles: {int(maze.sum())} | Device: {device}")
    console.print(f"Start: {start} → Goal: {goal}")
    console.print("\n[dim]Legend: ·· calm | ░░ low | ▒▒ medium | ▓▓ excited | ◆◆ sparkle | ▓▓ path[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")
    time.sleep(1)

    # State for visualization
    current_path = [None]
    step_counter = [0]

    def viz_callback(t, x, r, a, path):
        """Called each step - update visualization."""
        step_counter[0] = t
        if path:
            current_path[0] = path

    # Run with live display
    with Live(console=console, refresh_per_second=10) as live:
        def viz_with_live(t, x, r, a, path):
            viz_callback(t, x, r, a, path)
            if t % 3 == 0:  # Update display every 3 steps for smoother rendering
                panel = render_terminal(t, x, r, a, current_path[0], maze, start, goal, H, W)
                live.update(panel)

        try:
            path = excite_pathfinder(
                maze, start, goal,
                steps=1000,
                snap_every=25,
                alpha=0.35,      # Strong decay - prevents saturation
                kappa=3.0,       # Strong reinforcement for chains (compensates decay)
                beta1=0.5,       # Moderate neighbor coupling
                beta2=0.3,       # I-shape bonus
                beta3=0.3,       # L-shape bonus
                gamma=0.08,      # Slow spread - makes waves visible!
                theta=0.25,      # Lower threshold - easier to activate
                noise_rate=0.008,  # Fewer sparkles - more controlled
                refractory_dec=0.04,  # Faster refractory recovery
                fire_thr=0.5,
                seed=int(time.time()) % 1000,
                visualize_fn=viz_with_live
            )
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted by user[/yellow]")
            path = current_path[0]

    console.print("\n[bold cyan]🎇 COMPLETE 🎇[/bold cyan]")
    if path:
        console.print(f"[bold green]✓ PATH FOUND![/bold green] Length: {len(path)} (Manhattan: {abs(start[0] - goal[0]) + abs(start[1] - goal[1])})")
    else:
        console.print(f"[bold yellow]✗ No path found[/bold yellow] - try: --steps 2000 --noise 0.02")


def main():
    """Main entry with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="⚛️  Ionization Pathfinder - Watch particle physics solve mazes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ionization_pathfinder.py
  python ionization_pathfinder.py --size 20 --noise 0.0005 --fps 3
  python ionization_pathfinder.py --debugger --size 16
  python ionization_pathfinder.py --alpha 0.6 --gamma 0.03 --kappa 2.0
        """
    )

    # Maze parameters
    parser.add_argument('--size', type=int, default=None, help='Maze size (HxW, default: auto-fit to terminal)')
    parser.add_argument('--obstacles', type=float, default=0.12, help='Obstacle density (default: 0.12)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for maze generation')

    # Simulation parameters
    parser.add_argument('--steps', type=int, default=1000, help='Simulation steps (default: 1000)')
    parser.add_argument('--alpha', type=float, default=0.4, help='Decay rate (default: 0.4)')
    parser.add_argument('--kappa', type=float, default=4.0, help='Reinforcement strength (default: 4.0)')
    parser.add_argument('--beta1', type=float, default=0.4, help='Neighbor coupling (default: 0.4)')
    parser.add_argument('--beta2', type=float, default=0.2, help='I-shape bonus (default: 0.2)')
    parser.add_argument('--beta3', type=float, default=0.2, help='L-shape bonus (default: 0.2)')
    parser.add_argument('--gamma', type=float, default=0.05, help='Spread rate (default: 0.05)')
    parser.add_argument('--theta', type=float, default=0.25, help='Activation threshold (default: 0.25)')
    parser.add_argument('--noise', type=float, default=0.0003, help='Ionization rate per cell (default: 0.0003, ~1 event per 5 frames on 30x30)')

    # Display parameters
    parser.add_argument('--fps', type=int, default=10, help='Simulation speed: steps per second (default: 10, use 1 for slow motion)')
    parser.add_argument('--debugger', action='store_true', help='Step-through mode (SPACE to advance)')

    args = parser.parse_args()

    # Setup
    console.clear()
    console.print(f"\n[bold cyan]⚛️  Ionization Pathfinder {'[DEBUGGER MODE]' if args.debugger else ''} ⚛️[/bold cyan]\n")

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
    console.print(f"Start: {start} → Goal: {goal}")
    console.print(f"Steps: {args.steps} | FPS: {args.fps}")
    console.print(f"\n[dim]α={args.alpha} κ={args.kappa} γ={args.gamma} noise={args.noise}[/dim]")
    console.print("[dim]Legend: [/dim][green]██[/green][dim]=start [/dim][red]██[/red][dim]=goal [/dim]░░[dim]=wall | Energy: [/dim][dim]██[/dim] [rgb(0,0,100)]██[/rgb(0,0,100)] [blue]██[/blue] [cyan]██[/cyan] [bright_cyan]██[/bright_cyan] [white]██[/white]")
    if args.debugger:
        console.print("[yellow]DEBUGGER: Press SPACE to step, Ctrl+C to exit[/yellow]")
    console.print()

    time.sleep(0.5 if args.debugger else 1)

    # Run with live display
    if args.debugger:
        # Debugger mode: manual stepping
        def viz_debugger(t, x, r, a):
            panel = render_terminal(t, x, r, a, maze, start, goal, H, W)
            console.clear()
            console.print(panel)
            wait_for_space(console)

        try:
            excite_pathfinder(
                maze, start, goal,
                steps=args.steps,
                alpha=args.alpha,
                kappa=args.kappa,
                beta1=args.beta1,
                beta2=args.beta2,
                beta3=args.beta3,
                gamma=args.gamma,
                theta=args.theta,
                noise_rate=args.noise,
                refractory_dec=0.04,
                fire_thr=0.5,
                seed=int(time.time()) % 1000,
                visualize_fn=viz_debugger
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Exited by user[/yellow]")
    else:
        # Normal mode: controlled speed
        # fps controls simulation speed - each step pauses for 1/fps seconds
        step_delay = 1.0 / args.fps

        with Live(console=console, refresh_per_second=30) as live:
            def viz_live(t, x, r, a):
                # Show every step
                panel = render_terminal(t, x, r, a, maze, start, goal, H, W)
                live.update(panel)

                # Pause to control simulation speed
                time.sleep(step_delay)

            try:
                excite_pathfinder(
                    maze, start, goal,
                    steps=args.steps,
                    alpha=args.alpha,
                    kappa=args.kappa,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    beta3=args.beta3,
                    gamma=args.gamma,
                    theta=args.theta,
                    noise_rate=args.noise,
                    refractory_dec=0.04,
                    fire_thr=0.5,
                    seed=int(time.time()) % 1000,
                    visualize_fn=viz_live
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")

    console.print(f"\n[dim]Watch for bright connected cells forming a 'lightning bolt' from green to red![/dim]")


if __name__ == "__main__":
    main()

