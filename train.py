from __future__ import annotations
import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import bat_sim.physics as physics
from bat_sim.neural_net import BatNeuralNet, TOTAL_WEIGHTS
from bat_sim.evolution import Population, DEFAULT_POPULATION_SIZE, DEFAULT_MAX_STEPS
from bat_sim.environment import Environment, CircularObstacle, RectangularObstacle
from bat_sim.drone import Drone

def configure_curriculum_environment(env: Any, pop: Any, gen_idx: int, total_gens: int, curriculum: bool=True) -> float:
    if not curriculum or total_gens <= 1:
        return 75.0
    p = gen_idx / max(1, total_gens - 1)
    stage_speed = 55.0 + 55.0 * p
    for d in pop.drones:
        d.speed = stage_speed
    env.clear_obstacles()
    env.add_obstacle(CircularObstacle(350.0, 230.0, radius=35.0))
    env.add_obstacle(CircularObstacle(850.0, 230.0, radius=35.0))
    env.add_obstacle(CircularObstacle(350.0, 570.0, radius=35.0))
    env.add_obstacle(CircularObstacle(850.0, 570.0, radius=35.0))
    if p >= 0.25:
        env.add_obstacle(RectangularObstacle(600.0, 160.0, width=160.0, height=35.0, angle=0.0))
        env.add_obstacle(RectangularObstacle(600.0, 640.0, width=160.0, height=35.0, angle=0.0))
    if p >= 0.5:
        drift_spd = 20.0 + 20.0 * p
        env.add_obstacle(CircularObstacle(200.0, 400.0, radius=35.0, vx=0.0, vy=drift_spd))
        env.add_obstacle(CircularObstacle(1000.0, 400.0, radius=35.0, vx=0.0, vy=-drift_spd))
    if p >= 0.75:
        env.add_obstacle(RectangularObstacle(300.0, 380.0, width=60.0, height=40.0, angle=0.25, vx=15.0, vy=25.0))
        env.add_obstacle(CircularObstacle(600.0, 250.0, radius=30.0, vx=-20.0, vy=0.0))
    return stage_speed

def format_progress_bar(gen: int, total_gens: int, best_fitness: float, avg_fitness: float, survival_rate: float, elapsed_time: float, rate: float, bar_length: int=20) -> str:
    total = max(int(total_gens), 1)
    current = min(max(int(gen), 0), total)
    fraction = current / total
    filled_len = int(round(bar_length * fraction))
    if filled_len > 0 and filled_len < bar_length:
        bar = '=' * (filled_len - 1) + '>' + '-' * (bar_length - filled_len)
    elif filled_len >= bar_length:
        bar = '=' * bar_length
    else:
        bar = '-' * bar_length
    pct = int(round(fraction * 100.0))
    surv_pct = float(survival_rate) * 100.0
    return f'Gen {current:>{len(str(total))}}/{total} [{bar}] {pct:>3}% | Best: {best_fitness:>7.1f} | Avg: {avg_fitness:>7.1f} | Surv: {surv_pct:>5.1f}% | Time: {elapsed_time:>6.2f}s ({rate:>5.2f} gens/s)'

def train(generations: int=50, pop_size: int=DEFAULT_POPULATION_SIZE, max_steps: int=DEFAULT_MAX_STEPS, save_path: Union[str, os.PathLike]='best_weights.npz', seed: Optional[int]=None, verbose: bool=True, output: Optional[Union[str, os.PathLike]]=None, curriculum: bool=True) -> Dict[str, Any]:
    target_save_path = str(output) if output is not None else str(save_path)
    total_gens = int(generations)
    n_drones = int(pop_size)
    step_limit = int(max_steps)
    if seed is not None:
        np.random.seed(seed)
    env = Environment()
    pop = Population(size=n_drones, max_steps=step_limit, track_wavefronts=False, seed=seed)
    history: List[Dict[str, Any]] = []
    all_time_best_fitness: float = -float('inf')
    all_time_best_weights: Optional[np.ndarray] = None
    t_start = time.perf_counter()
    if verbose and total_gens > 0:
        print(f'Starting headless training: {total_gens} generations | Pop size: {n_drones} | Max steps: {step_limit}')

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
        configure_curriculum_environment(env, pop, gen_idx, total_gens, curriculum=curriculum)
        tel = pop.step_generation(env=env)
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
        rate = completed / max(elapsed, 1e-06)
        if verbose:
            bar_text = format_progress_bar(gen=completed, total_gens=total_gens, best_fitness=all_time_best_fitness if all_time_best_fitness != -float('inf') else float(tel.get('best_fitness', 0.0)), avg_fitness=float(tel.get('avg_fitness', 0.0)), survival_rate=float(tel.get('survival_rate', 0.0)), elapsed_time=elapsed, rate=rate)
            sys.stdout.write(f'\r{bar_text}')
            sys.stdout.flush()
        if gen_idx < total_gens - 1:
            pop.reproduce()
    elapsed_total = time.perf_counter() - t_start
    final_rate = total_gens / max(elapsed_total, 1e-06) if total_gens > 0 else 0.0
    if verbose and total_gens > 0:
        sys.stdout.write('\n')
        sys.stdout.flush()
    if all_time_best_weights is None:
        if len(pop.drones) > 0 and pop.drones[0].controller is not None:
            all_time_best_weights = pop.drones[0].controller.get_flat_weights().copy()
            all_time_best_fitness = float(getattr(pop.drones[0], 'fitness', 0.0))
        else:
            dummy_net = BatNeuralNet()
            all_time_best_weights = dummy_net.get_flat_weights().copy()
            all_time_best_fitness = 0.0
    champion_net = BatNeuralNet(weights=all_time_best_weights)
    champion_net.save(target_save_path)
    abs_save_path = os.path.abspath(target_save_path)
    final_avg_fitness = float(history[-1].get('avg_fitness', 0.0)) if history else 0.0
    final_survival_rate = float(history[-1].get('survival_rate', 0.0)) if history else 0.0
    if verbose:
        print(f'Training complete in {elapsed_total:.2f}s ({final_rate:.2f} gens/s) | All-time best fitness: {all_time_best_fitness:.2f} | Weights saved: {abs_save_path}')
    summary: Dict[str, Any] = {'generations': total_gens, 'pop_size': n_drones, 'max_steps': step_limit, 'elapsed_time': float(elapsed_total), 'gens_per_sec': float(final_rate), 'rate_gens_per_sec': float(final_rate), 'best_fitness': float(all_time_best_fitness), 'avg_fitness': float(final_avg_fitness), 'survival_rate': float(final_survival_rate), 'save_path': abs_save_path, 'champion_weights': all_time_best_weights, 'history': history}
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='bat_sim.train', description='Headless Neuroevolution Training Pipeline for Autonomous Bat Echolocation Drones.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--generations', type=int, default=50, help='Number of evolutionary generations to simulate.')
    parser.add_argument('--pop-size', type=int, default=DEFAULT_POPULATION_SIZE, help='Population size (number of drones per generation).')
    parser.add_argument('--max-steps', type=int, default=DEFAULT_MAX_STEPS, help='Maximum simulation steps per generation episode.')
    parser.add_argument('--output', type=str, default='best_weights.npz', help='File path to save the best neural network weights (.npz).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible training runs.')
    parser.add_argument('--curriculum', action='store_true', default=True, help='Enable progressive curriculum difficulty (speed and obstacle staging).')
    parser.add_argument('--no-curriculum', action='store_false', dest='curriculum', help='Disable progressive curriculum staging.')
    parser.add_argument('--quiet', action='store_true', default=False, help='Suppress CLI progress bar and summary output.')
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(generations=args.generations, pop_size=args.pop_size, max_steps=args.max_steps, save_path=args.output, seed=args.seed, verbose=not args.quiet, curriculum=args.curriculum)
if __name__ == '__main__':
    main()
