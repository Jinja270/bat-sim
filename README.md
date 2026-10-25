# Autonomous Bat Echolocation & Terminal Buzz Neuroevolution

A 60 FPS real-time bio-inspired neuroevolution simulation in Python/Pygame. A swarm of 100 autonomous drones evolves acoustic obstacle evasion, dynamic Doppler-closing velocity detection, and biological terminal buzz behavior driven by a pure NumPy Multi-Layer Perceptron (1,314 parameters).

Includes a dedicated live neural network visualizer that streams synaptic activations, firing pulses, and membrane potentials in real time via shared memory IPC (`multiprocessing.shared_memory`).

---

## Key Features

- **Bio-Acoustic Sensing Model:**
  - 3-Ray Acoustic Cones: Left (-30°), Front (0°), Right (+30°).
  - Time-of-Flight (ToF) range measurement and acoustic Doppler shift closing velocity $v_c$.
  - Dynamic Pulse Frequency Modulation: Automatically scales from a **10 Hz search phase** up to a high-frequency **150 Hz terminal buzz attack phase** as obstacle proximity increases.

- **Pure NumPy Neural Architecture:**
  - Fast vectorized MLP: $5 \to 32 \to 32 \to 2$ (1,314 trainable parameters).
  - Inputs: $[d_L, d_R, d_F, v, v_c]$ (normalized distance rays, cruise speed, Doppler closing velocity).
  - Outputs: Steering rate $\Delta\theta \in [-1.0, 1.0]$ ($\tanh$) and Buzz urgency $\lambda \in [0.0, 1.0]$ ($\text{sigmoid}$).

- **Neuroevolutionary Engine:**
  - **Truncation Selection:** Top 20% fittest individuals form the parent breeding pool.
  - **Elitism:** Top 2 champions preserved with zero mutation.
  - **Uniform Crossover & Decaying Mutation:** 50/50 gene splice with geometric schedule $\sigma(g) = \max(0.01, 0.10 \times 0.98^g)$.
  - Dual-mode stepping: 60 FPS Pygame interactive loop or accelerated CPU batch execution (50 generations in <10 seconds).

- **Dual-Window Live Neural Visualizer:**
  - Docked in-HUD overlay or separate dedicated OS pop-out window.
  - Sub-millisecond synchronization via OS Shared Memory (`multiprocessing.shared_memory`).
  - Synaptic connection mesh illuminating with active signal currents (amber for proximity evasion, cyan/gold for bilateral steering, pulsing crimson for terminal buzz).
  - Traveling action potential particles flowing at the exact acoustic pulse frequency.

- **Interactive Physics Arena:**
  - Dynamic, drifting circular and rectangular obstacles.
  - Real-time mouse drag-and-drop to reposition obstacles on the fly.
  - Left click to spawn circular obstacles; Right click to remove or spawn rectangular baffles.
  - Traditional 10 Hz baseline comparison mode to benchmark neuroevolved agents against fixed-rate threshold drones.

---

## Interactive Controls

| Key / Action | Function |
| :--- | :--- |
| **`Space`** | Pause / Resume simulation |
| **`F`** | Toggle Fast-Forward mode (maximum CPU execution speed) |
| **`E`** | Toggle between **Evolving Population** and **Champion Demo** |
| **`S`** | Manually save current champion weights to `best_weights.npz` |
| **`C`** | Toggle Comparison mode (evolved bio-drones vs traditional 10 Hz drones) |
| **`D`** | Cycle sensor ray rendering (`best` $\to$ `all` $\to$ `none`) |
| **`W`** | Toggle acoustic wavefront rings |
| **`N` / `T`** | Cycle in-game neural network display mode (docked $\to$ centered $\to$ hidden) |
| **`P`** | Pop out dedicated neural network visualizer window |
| **`Up` / `Down`** | Adjust flight cruise speed (40 to 130 px/s) |
| **`R`** | Reset arena to default layout |
| **Left Click** | Drag existing obstacle, or click empty space to spawn circular pylon |
| **Right Click** | Delete clicked obstacle, or click empty space to spawn rectangular baffle |
| **`Esc`** | Graceful exit and telemetry report |

---

## Installation & Quickstart

### Prerequisites
- Python 3.9+
- pip

```bash
# Clone the repository
git clone https://github.com/Jinja270/bat-sim.git
cd bat-sim

# Install dependencies
pip install -r requirements.txt
```

### Launch Interactive Simulation
```bash
python main.py
```

### Launch 3D Overhead Maze Swarm Simulation
```bash
python maze.py
# or: python main.py --maze
```

### Launch with Dedicated Neural Window
```bash
python main.py --neural-window
```
*(Or launch `python neural_window.py` in a separate terminal while `main.py` is running).*

### Headless Training CLI
Train a population headlessly at maximum CPU speed:
```bash
python train.py --generations 50 --pop-size 100 --output best_weights.npz
```

---

## Architecture Overview

```
bat-sim/
├── main.py                  # 60 FPS Pygame interactive loop & telemetry HUD
├── maze.py                  # 3D overhead labyrinth & autonomous green swarm neuroevolution
├── neural_window.py         # Dedicated second window for neural architecture visualizer
├── neural_ipc.py            # High-speed shared-memory IPC (multiprocessing.shared_memory)
├── neural_visualizer.py     # Real-time synaptic mesh, glowing nodes, and particle renderer
├── drone.py                 # Bio-inspired drone kinematics & acoustic sensor raycasting
├── environment.py           # 2D arena, obstacle collision detection, and mouse interaction
├── evolution.py             # Genetic algorithm, reproduction, elitism, and telemetry tracking
├── neural_net.py            # 5->32->32->2 NumPy MLP inference and parameter views
├── physics.py               # Vectorized ToF ray-casting, Doppler calculations, geometry
├── train.py                 # Headless batch training script with progress bar
├── best_weights.npz         # Saved pre-trained 98.3% survival champion weights
├── requirements.txt         # Core dependencies (numpy, pygame)
└── README.md                # Project documentation
```

---

## License

MIT License. Free for academic, educational, and open-source use.
