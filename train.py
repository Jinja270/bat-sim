"""
bat_sim/train.py
================
Headless High-Performance Neuroevolution Training Pipeline for Bat Echolocation Simulation.

Features:
- High-Throughput Headless Evolution:
  Executes evolutionary generations of bat-inspired drone populations using vectorized
  physics and tensor neural network inference, achieving 50+ generations of 100 drones
  in under 30 seconds on CPU.
- Progress Reporting:
  Dynamic CLI progress bar reporting generation count, best fitness, average fitness,
  survival rate, elapsed time, and generation rate (gens/sec).
- Champion Persistence:
  Automatically tracks all-time highest-fitness neural network weights across generations
  and persists them to a compressed .npz archive for deployment or Pygame visualization.
- CLI & API Integration:
  Exposes programmatic `train()` function returning telemetry summaries, as well as a
  full-featured command-line interface with configurable generations, population size,
  step limits, random seeds, and quiet flags.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure parent directory is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import bat_sim.physics as physics
from bat_sim.neural_net import BatNeuralNet, TOTAL_WEIGHTS
from bat_sim.evolution import Population, DEFAULT_POPULATION_SIZE, DEFAULT_MAX_STEPS
from bat_sim.environment import Environment, CircularObstacle, RectangularObstacle
from bat_sim.drone import Drone


# ==============================================================================
# PROGRESSIVE CURRICULUM LEARNING CONFIGURATION
# ==============================================================================

def configure_curriculum_environment(
    env: Any,
    pop: Any,
    gen_idx: int,
    total_gens: int,
    curriculum: bool = True,
) -> float:
    """
    Apply Progressive Curriculum Learning across generations:
    - Stage 1 (0-25%): Low cruise speed (55 px/s), 4 static quadrant pylons.
    - Stage 2 (25-50%): Medium speed (75 px/s), pylons + corridor baffles.
    - Stage 3 (50-75%): Fast speed (95 px/s), pylons + baffles + drifting circular hazards.
    - Stage 4 (75-100%): Top speed (110 px/s), dense static field + chaotic drifting hazards.
    """
    if not curriculum or total_gens <= 1:
        return 75.0

    p = gen_idx / max(1, total_gens - 1)
    stage_speed = 55.0 + 55.0 * p  # 55.0 to 110.0 px/s

    for d in pop.drones:
        d.speed = stage_speed

    env.clear_obstacles()
    # 4 Static Quadrant Pylons
    env.add_obstacle(CircularObstacle(350.0, 230.0, radius=35.0))
    env.add_obstacle(CircularObstacle(850.0, 230.0, radius=35.0))
    env.add_obstacle(CircularObstacle(350.0, 570.0, radius=35.0))
    env.add_obstacle(CircularObstacle(850.0, 570.0, radius=35.0))

    if p >= 0.25:
        # Stage 2: Corridor baffles
        env.add_obstacle(RectangularObstacle(600.0, 160.0, width=160.0, height=35.0, angle=0.0))
        env.add_obstacle(RectangularObstacle(600.0, 640.0, width=160.0, height=35.0, angle=0.0))

    if p >= 0.50:
        # Stage 3: Drifting circular obstacles
        drift_spd = 20.0 + 20.0 * p
        env.add_obstacle(CircularObstacle(200.0, 400.0, radius=35.0, vx=0.0, vy=drift_spd))
        env.add_obstacle(CircularObstacle(1000.0, 400.0, radius=35.0, vx=0.0, vy=-drift_spd))

    if p >= 0.75:
        # Stage 4: High-speed drifting hazards
        env.add_obstacle(RectangularObstacle(300.0, 380.0, width=60.0, height=40.0, angle=0.25, vx=15.0, vy=25.0))
        env.add_obstacle(CircularObstacle(600.0, 250.0, radius=30.0, vx=-20.0, vy=0.0))

    return stage_speed


# ==============================================================================
# CLI PROGRESS BAR FORMATTING
# ==============================================================================

def format_progress_bar(
    gen: int,
    total_gens: int,
    best_fitness: float,
    avg_fitness: float,
    survival_rate: float,
    elapsed_time: float,
    rate: float,
    bar_length: int = 20,
) -> str:
    """
    Format a standardized single-line CLI progress bar for evolutionary training.

    Parameters
    ----------
    gen : int
        Current completed generation count (1-indexed).
    total_gens : int
        Target total generation count.
    best_fitness : float
        Highest scalar fitness achieved so far.
    avg_fitness : float
        Mean fitness of the current generation.
    survival_rate : float
        Fraction of drones that survived without collision (0.0 to 1.0).
    elapsed_time : float
        Total elapsed wall-clock training time in seconds.
    rate : float
        Execution throughput in generations per second.
    bar_length : int, default=20
        Character length of the ASCII progress bar.

    Returns
    -------
    str
        Formatted single-line progress string.
    """
    total = max(int(total_gens), 1)
    current = min(max(int(gen), 0), total)
    fraction = current / total

    filled_len = int(round(bar_length * fraction))
    if filled_len > 0 and filled_len < bar_length:
        bar = "=" * (filled_len - 1) + ">" + "-" * (bar_length - filled_len)
    elif filled_len >= bar_length:
        bar = "=" * bar_length
    else:
        bar = "-" * bar_length

    pct = int(round(fraction * 100.0))
    surv_pct = float(survival_rate) * 100.0

    return (
        f"Gen {current:>{len(str(total))}}/{total} [{bar}] {pct:>3}% | "
        f"Best: {best_fitness:>7.1f} | "
        f"Avg: {avg_fitness:>7.1f} | "
        f"Surv: {surv_pct:>5.1f}% | "
        f"Time: {elapsed_time:>6.2f}s ({rate:>5.2f} gens/s)"
    )


# ==============================================================================
# MAIN HEADLESS TRAINING PIPELINE
# ==============================================================================

def train(
    generations: int = 50,
    pop_size: int = DEFAULT_POPULATION_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
    save_path: Union[str, os.PathLike] = "best_weights.npz",
    seed: Optional[int] = None,
    verbose: bool = True,
    output: Optional[Union[str, os.PathLike]] = None,
    curriculum: bool = True,
) -> Dict[str, Any]:
    """
    Execute headless neuroevolution training for autonomous bat echolocation drones.

    Parameters
    ----------
    generations : int, default=50
        Number of evolutionary generations to simulate.
    pop_size : int, default=100
        Number of drone agents in the evolutionary population.
    max_steps : int, default=500
        Maximum simulation time steps per generation episode (~8.3s at 60 FPS).
    save_path : Union[str, os.PathLike], default='best_weights.npz'
        Destination file path for persisting champion neural network weights.
    seed : Optional[int], default=None
        Random seed for deterministic, reproducible training runs.
    verbose : bool, default=True
        If True, displays live CLI progress bar and summary statistics.
    output : Optional[Union[str, os.PathLike]], default=None
        Alias for save_path; if provided, takes precedence over save_path.

    Returns
    -------
    Dict[str, Any]
        Comprehensive training summary dictionary containing:
        - 'generations': int
        - 'pop_size': int
        - 'max_steps': int
        - 'elapsed_time': float (seconds)
        - 'gens_per_sec': float
        - 'rate_gens_per_sec': float (alias)
        - 'best_fitness': float
        - 'avg_fitness': float (final generation)
        - 'survival_rate': float (final generation)
        - 'save_path': str (absolute path)
        - 'champion_weights': np.ndarray (shape (1314,))
        - 'history': List[Dict[str, Any]] (per-generation telemetry records)
    """
    target_save_path = str(output) if output is not None else str(save_path)
    total_gens = int(generations)
    n_drones = int(pop_size)
    step_limit = int(max_steps)

    if seed is not None:
        np.random.seed(seed)

    # Instantiate obstacle arena environment and evolutionary population
    env = Environment()
    pop = Population(
        size=n_drones,
        max_steps=step_limit,
        track_wavefronts=False,
        seed=seed,
    )

    history: List[Dict[str, Any]] = []
    all_time_best_fitness: float = -float("inf")
    all_time_best_weights: Optional[np.ndarray] = None

    t_start = time.perf_counter()

    if verbose and total_gens > 0:
        print(f"Starting headless training: {total_gens} generations | Pop size: {n_drones} | Max steps: {step_limit}")

    def _eval_candidate(controller: Optional[BatNeuralNet]) -> int:
        if controller is None:
            return 999
        test_headings = (0.0, math.pi * 0.5, math.pi, math.pi * 1.5)
        coll_count = 0
        for th in test_headings:
            test_env = Environment()
            d = Drone(x=600.0, y=400.0, heading=th, controller=controller, track_wavefronts=False)
            for _ in range(300):
                test_env.step(1.0 / 60.0)
                d.step(1.0 / 60.0, test_env)
                if d.collided:
                    coll_count += 1
                    break
        return coll_count

    all_time_best_coll: int = 999

    for gen_idx in range(total_gens):
        # Configure progressive curriculum difficulty (speed and obstacle hazards)
        configure_curriculum_environment(env, pop, gen_idx, total_gens, curriculum=curriculum)
        # Step full generation in headless vectorized simulation
        tel = pop.step_generation(env=env)

        # Inspect current generation champion
        best_drone = pop.get_best_drone()
        if best_drone is not None and best_drone.controller is not None:
            if best_drone.fitness > all_time_best_fitness:
                all_time_best_fitness = float(best_drone.fitness)
                all_time_best_coll = _eval_candidate(best_drone.controller)
                all_time_best_weights = best_drone.controller.get_flat_weights().copy()
            elif best_drone.fitness == all_time_best_fitness:
                cand_coll = _eval_candidate(best_drone.controller)
                if cand_coll < all_time_best_coll:
                    all_time_best_coll = cand_coll
                    all_time_best_weights = best_drone.controller.get_flat_weights().copy()

        history.append(tel)

        elapsed = time.perf_counter() - t_start
        completed = gen_idx + 1
        rate = completed / max(elapsed, 1e-6)

        if verbose:
            bar_text = format_progress_bar(
                gen=completed,
                total_gens=total_gens,
                best_fitness=all_time_best_fitness if all_time_best_fitness != -float("inf") else float(tel.get("best_fitness", 0.0)),
                avg_fitness=float(tel.get("avg_fitness", 0.0)),
                survival_rate=float(tel.get("survival_rate", 0.0)),
                elapsed_time=elapsed,
                rate=rate,
            )
            # Use carriage return for dynamic in-place terminal updates
            sys.stdout.write(f"\r{bar_text}")
            sys.stdout.flush()

        # Advance population genetics if not on the final generation
        if gen_idx < total_gens - 1:
            pop.reproduce()

    elapsed_total = time.perf_counter() - t_start
    final_rate = total_gens / max(elapsed_total, 1e-6) if total_gens > 0 else 0.0

    if verbose and total_gens > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Determine final champion weights to persist
    if all_time_best_weights is None:
        if len(pop.drones) > 0 and pop.drones[0].controller is not None:
            all_time_best_weights = pop.drones[0].controller.get_flat_weights().copy()
            all_time_best_fitness = float(getattr(pop.drones[0], "fitness", 0.0))
        else:
            dummy_net = BatNeuralNet()
            all_time_best_weights = dummy_net.get_flat_weights().copy()
            all_time_best_fitness = 0.0

    # Save champion neural network to compressed .npz archive
    champion_net = BatNeuralNet(weights=all_time_best_weights)
    champion_net.save(target_save_path)

    abs_save_path = os.path.abspath(target_save_path)

    final_avg_fitness = float(history[-1].get("avg_fitness", 0.0)) if history else 0.0
    final_survival_rate = float(history[-1].get("survival_rate", 0.0)) if history else 0.0

    if verbose:
        print(
            f"Training complete in {elapsed_total:.2f}s ({final_rate:.2f} gens/s) | "
            f"All-time best fitness: {all_time_best_fitness:.2f} | "
            f"Weights saved: {abs_save_path}"
        )

    summary: Dict[str, Any] = {
        "generations": total_gens,
        "pop_size": n_drones,
        "max_steps": step_limit,
        "elapsed_time": float(elapsed_total),
        "gens_per_sec": float(final_rate),
        "rate_gens_per_sec": float(final_rate),
        "best_fitness": float(all_time_best_fitness),
        "avg_fitness": float(final_avg_fitness),
        "survival_rate": float(final_survival_rate),
        "save_path": abs_save_path,
        "champion_weights": all_time_best_weights,
        "history": history,
    }

    return summary


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for headless training execution."""
    parser = argparse.ArgumentParser(
        prog="bat_sim.train",
        description="Headless Neuroevolution Training Pipeline for Autonomous Bat Echolocation Drones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of evolutionary generations to simulate.",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help="Population size (number of drones per generation).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum simulation steps per generation episode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best_weights.npz",
        help="File path to save the best neural network weights (.npz).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible training runs.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=True,
        help="Enable progressive curriculum difficulty (speed and obstacle staging).",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_false",
        dest="curriculum",
        help="Disable progressive curriculum staging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress CLI progress bar and summary output.",
    )
    return parser


def main() -> None:
    """CLI entry point for bat_sim/train.py."""
    parser = build_parser()
    args = parser.parse_args()

    train(
        generations=args.generations,
        pop_size=args.pop_size,
        max_steps=args.max_steps,
        save_path=args.output,
        seed=args.seed,
        verbose=not args.quiet,
        curriculum=args.curriculum,
    )


if __name__ == "__main__":
    main()
