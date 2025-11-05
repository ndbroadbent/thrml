# ⚛️ GPU Ionization Pathfinder

**An excitable medium pathfinder inspired by particle physics**

Watch ionization events spread like cosmic rays in a cloud chamber, forming trails that connect through mazes in real-time!

## The Concept

Instead of traditional pathfinding or static energy minimization, this uses **excitable medium dynamics** inspired by particle physics:

- ⚛️ **Ionization Events**: Random cosmic-ray-like events excite calm cells
- 🌊 **Cascade Propagation**: Excited regions spread to neighbors like particle showers
- 🔗 **Trail Reinforcement**: Connected ionization trails persist (slower decay)
- 💀 **Dead ends fade**: Isolated excited regions eventually decay
- ⚡ **Lightning emergence**: Stable trails naturally connect start and goal terminals!

## How It Works

Each cell has **excitement** `x` ∈ [0,1] and evolves via:

```python
neighbors = count_excited_neighbors(x)
motifs = detect_I_shapes(x) + detect_L_shapes(x)  # Chain patterns

reinforcement = β₁·neighbors + β₂·I_shapes + β₃·L_shapes

# Decay slows where chains exist!
decay_rate = α / (1 + κ·reinforcement)

x(t+1) = (1 - decay_rate)·x(t) + γ·reinforcement + sources + sparkles
```

**Key properties:**
- Lone excited cells decay fast
- Chains of 3-4 cells reinforce each other → decay slow
- Dead ends eventually die (only 1 neighbor)
- Valid paths persist (2 neighbors each)
- Start/goal are constant sources

## Installation

```bash
cd path_finding_gpu
uv pip install -r requirements.txt
```

Requires:
- PyTorch (CUDA optional but recommended for speed)
- networkx (for path extraction)
- rich (for terminal UI)

## Run It!

```bash
python ionization_pathfinder.py
```

Watch particle physics in your terminal! ⚛️

## Terminal Visualization

**Colors:**
- `··` (dim) = Calm cells (excitement ≈ 0)
- `░░` (blue) = Low energy (0.05-0.2)
- `▒▒` (cyan) = Medium energy (0.2-0.4)
- `▓▓` (bright cyan) = Excited (0.4-0.6)
- `▓▓` (magenta) = Very excited (0.6-0.8)
- `◆◆` (bright magenta) = **SPARKLE!** (0.8-1.0)
- `◉◉` (green) = Start (constant source)
- `◉◉` (red) = Goal (constant source)
- `▓▓` (yellow) = Solution path (when found)
- `██` (black) = Walls

## What You'll See

1. **Random sparkles** appearing and fading
2. **Waves** emanating from start and goal
3. **Chains forming** as nearby excited cells reinforce each other
4. **Dead ends** slowly fading away
5. **Connection!** When chains meet, path emerges
6. **Yellow path** highlighting the solution

## Tuning Parameters

In `sparkle_pathfinder.py`, adjust:

```python
steps=800,        # Total time steps
noise_rate=0.015, # Sparkle frequency (higher = more random exploration)
alpha=0.06,       # Base decay (higher = faster decay)
kappa=1.6,        # Reinforcement effect (higher = chains live longer)
beta1/2/3,        # Neighbor/I-shape/L-shape weights
gamma=0.18,       # Spread rate (higher = waves propagate faster)
theta=0.35,       # Activation threshold
```

**More sparkles:** Increase `noise_rate`
**Longer-lived chains:** Increase `kappa`
**Faster waves:** Increase `gamma`
**Slower decay:** Decrease `alpha`

## Why GPU?

Each step is just:
- 5 convolutions (4-neighbor, 2 I-shapes, 2 L-shapes)
- Element-wise operations
- All massively parallel!

On CUDA: Can run 800 steps on 64×64 grid in ~1 second

## The Physics

This is an **excitable medium** like:
- Neural activity spreading in brain tissue
- Chemical waves in Belousov-Zhabotinsky reactions
- Forest fires spreading
- Ant colony pheromone trails

But with controlled randomness (sparkles) and motif detection (I/L shapes favor path formation).

## Not an EBM!

This is **non-equilibrium dynamics**, not energy-based model sampling:
- Time-evolution with history dependence
- No detailed balance
- Purpose-built for search, not probability distributions

But conceptually related to how excitable p-bit networks could work on thermodynamic hardware!

## Key Insight

Your intuition was perfect: **Let physics do the work**
- Don't prescribe the path
- Create local dynamics that favor connectivity
- Emergent behavior finds solutions
- GPU makes it fast enough to watch!

Enjoy the sparkles! 🎇✨

