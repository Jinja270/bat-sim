"""
Headless Swarm Training & Optimization Pipeline
Evolves and refines weights from best_weights_maze.npz to achieve maximum
swarm exit convergence (>60/80 finishers) and high-speed trajectory lines.
"""
import os
import sys
import time
import numpy as np

# Ensure bat_sim package path
sys.path.insert(0, os.path.dirname(__file__))

from maze import Maze, MazePopulation, MazeDrone, BatNeuralNet

def run_headless_training(
    target_weights_path: str = "best_weights_maze.npz",
    num_generations: int = 50,
    pop_size: int = 80,
    max_steps: int = 8000,
    seed: int = 42,
) -> None:
    print("=" * 75)
    print("  AUTONOMOUS SWARM HEADLESS TRAINING & WEIGHT OPTIMIZATION")
    print(f"  Target File: {target_weights_path} | Generations: {num_generations} | Swarm: {pop_size}")
    print("=" * 75)

    maze = Maze(cols=23, rows=23, cell_size=38.0, seed=seed)
    pop = MazePopulation(size=pop_size, max_steps=max_steps, seed=seed, scratch=False)

    # Load existing weights if present
    if os.path.exists(target_weights_path):
        try:
            champ = BatNeuralNet.load(target_weights_path, rng=pop.rng)
            pop.all_time_best_controller = champ.copy()
            for d in pop.drones:
                d.controller = champ.copy()
                d.corridor_bias = 0.0
                d.speed_multiplier = 1.0
            pop._sync_vectors()
            print(f"[INIT] Loaded existing weights from {target_weights_path} (norm = {np.linalg.norm(champ.get_flat_weights()):.2f})")
        except Exception as e:
            print(f"[INIT] Could not load existing weights: {e}, starting from population defaults.")

    pop.reset_at_maze(maze)
    dt = 1.0 / 60.0

    best_finishers_seen = 0
    fastest_finish_time = float("inf")

    t_start = time.perf_counter()

    for gen in range(num_generations):
        step_count = 0
        while True:
            step_count += 1
            done = pop.step(dt, maze)
            if done or step_count >= max_steps:
                break

        # Check generation metrics
        pop.evaluate_fitness()
        finishers = [d for d in pop.drones if d.reached_goal]
        num_finishers = len(finishers)

        if num_finishers > 0:
            fastest_this_gen = min(d.time_alive_steps for d in finishers)
            if fastest_this_gen < fastest_finish_time:
                fastest_finish_time = fastest_this_gen

            if num_finishers >= best_finishers_seen:
                best_finishers_seen = num_finishers
                # Save checkpoint immediately
                pop.save_best(target_weights_path)

        elapsed = time.perf_counter() - t_start
        status_tag = f"EXIT REACHED ({num_finishers}/{pop_size} drones)" if num_finishers > 0 else f"Peak Prog: {pop.all_time_best_progress*100:5.1f}%"
        print(f"Gen {gen:03d} | Steps: {step_count:4d} | {status_tag} | Fast Time: {fastest_finish_time if fastest_finish_time < 9999 else '---'} | Total Elapsed: {elapsed:5.1f}s")

        # Reproduce for next generation
        pop.reproduce(maze)

        # Early exit if we reach near-perfect convergence (>65 finishers and fast time)
        if best_finishers_seen >= 65 and gen >= 25:
            print(f"\n[TARGET CONVERGENCE ACHIEVED] {best_finishers_seen}/{pop_size} drones reaching target! Stopping early.")
            break

    # Final save
    pop.save_best(target_weights_path)
    total_elapsed = time.perf_counter() - t_start
    print("=" * 75)
    print(f"TRAINING COMPLETE IN {total_elapsed:.2f}s!")
    print(f"Peak Swarm Convergence: {best_finishers_seen}/{pop_size} drones reached target")
    print(f"Fastest Goal Arrival Time: {fastest_finish_time} steps ({fastest_finish_time/60.0:.2f}s)")
    print(f"Saved perfected weights to -> {target_weights_path}")
    print("=" * 75)

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "best_weights_maze.npz")
    gens = 40
    if len(sys.argv) > 1:
        try:
            gens = int(sys.argv[1])
        except ValueError:
            pass
    run_headless_training(target_weights_path=target, num_generations=gens)
