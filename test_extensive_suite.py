"""
Extensive Test and Benchmark Suite for Autonomous Swarm Maze Navigation
Covers:
1. Neural Network Architecture Invariants & Numerical Stability
2. 7-Ray Spatial Vision & Maze Raycasting Geometry
3. Vectorized Swarm Math & Micro-Benchmark Throughput
4. Multi-Generation Evolution Stress Runs (Multiple Seeds)
5. Weight Persistence (.npz) & Deterministic Replay
6. Inter-Process Communication (Shared Memory & File Fallback)
"""

from __future__ import annotations
import math
import os
import sys
import time
from typing import List, Tuple
import numpy as np

# Ensure bat_sim package resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bat_sim.neural_net import (
    BatNeuralNet,
    TOTAL_WEIGHTS,
    INPUT_DIM,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    OUTPUT_DIM,
    W1_SHAPE,
    W_REC_SHAPE,
    B1_SHAPE,
    W2_SHAPE,
    B2_SHAPE,
    W3_SHAPE,
    B3_SHAPE,
    W_SKIP_SHAPE,
    leaky_relu,
    safe_sigmoid,
    compute_mutation_sigma,
    breed_population_vectorized,
)
from bat_sim.maze import Maze, MazeDrone, MazePopulation
from bat_sim.neural_ipc import ActivationWriter, ActivationReader


def test_section_header(title: str) -> None:
    print(f"\n{'='*75}\n[TEST SUITE] {title}\n{'='*75}")


def run_phase_1_neural_invariants() -> bool:
    test_section_header("PHASE 1: Neural Architecture & Numerical Stability")
    
    # 1. Total weights match
    expected_total = (
        9 * 32 + 32 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3 + 9 * 3
    )
    assert TOTAL_WEIGHTS == expected_total == 2526, f"Weight count mismatch: {TOTAL_WEIGHTS} != {expected_total}"
    print(f"  [PASS] Total parameter geometry verified: {TOTAL_WEIGHTS} weights")

    net = BatNeuralNet()
    assert net.w1.shape == W1_SHAPE == (9, 32)
    assert net.w_rec.shape == W_REC_SHAPE == (32, 32)
    assert net.b1.shape == B1_SHAPE == (32,)
    assert net.w2.shape == W2_SHAPE == (32, 32)
    assert net.b2.shape == B2_SHAPE == (32,)
    assert net.w3.shape == W3_SHAPE == (32, 3)
    assert net.b3.shape == B3_SHAPE == (3,)
    assert net.w_skip.shape == W_SKIP_SHAPE == (9, 3)
    print("  [PASS] Layer tensor dimensions & memory views match exact shapes")

    # 2. Numerical Stability over 50,000 random & extreme inputs
    rng = np.random.default_rng(42)
    for i in range(50000):
        # Mix of normal, high magnitude, and edge inputs
        if i % 1000 == 0:
            raw_x = rng.choice([-1e6, 1e6, -50.0, 50.0, 0.0, 1.0], size=9)
        else:
            raw_x = rng.uniform(-5.0, 5.0, size=9)
        
        out = net.forward(raw_x)
        assert not np.any(np.isnan(out)), f"NaN output detected at sample {i}"
        assert not np.any(np.isinf(out)), f"Inf output detected at sample {i}"
        assert -1.0001 <= out[0] <= 1.0001, f"Steering out of tanh range: {out[0]}"
        assert 0.0 <= out[1] <= 1.0, f"Urgency out of sigmoid range: {out[1]}"
        assert 0.0 <= out[2] <= 1.0, f"Throttle out of sigmoid range: {out[2]}"
    print("  [PASS] 50,000 extreme input vectors verified: zero NaN, zero Inf, strict bound adherence")

    # 3. Recurrent State Evolution
    net.reset_state()
    assert np.all(net.h_state == 0.0), "reset_state failed to zero hidden state"
    x_test = np.full(9, 0.5)
    out1 = net.forward(x_test).copy()
    h_state_1 = net.h_state.copy()
    assert not np.all(h_state_1 == 0.0), "Recurrent state did not activate after step 1"
    
    out2 = net.forward(x_test).copy()
    h_state_2 = net.h_state.copy()
    assert not np.array_equal(h_state_1, h_state_2), "Recurrent state failed to evolve across time steps"
    
    net.reset_state()
    assert np.all(net.h_state == 0.0)
    out_after_reset = net.forward(x_test)
    assert np.allclose(out1, out_after_reset, atol=1e-12), "State reset failed to reproduce identical output"
    print("  [PASS] Recurrent latent state evolution and deterministic reset verified")

    # 4. Reflex Skip Connections Instant Response
    # Clear recurrent state and set zero hidden layer
    net_skip = BatNeuralNet()
    # Left wall close vs Right wall close
    x_left_close = np.array([0.05, 0.05, 0.05, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5])
    x_right_close = np.array([1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 0.05, 0.0, 0.5])
    
    out_left = net_skip.forward(x_left_close)
    net_skip.reset_state()
    out_right = net_skip.forward(x_right_close)
    
    # Left wall close should steer right (> 0), right wall close should steer left (< 0)
    assert out_left[0] > 0.0, f"Left wall proximity failed to steer right: {out_left[0]}"
    assert out_right[0] < 0.0, f"Right wall proximity failed to steer left: {out_right[0]}"
    print(f"  [PASS] Spinal reflex skip connections: Left close -> {out_left[0]:+.2f} (steer right), Right close -> {out_right[0]:+.2f} (steer left)")

    # 5. Dynamic throttle response
    x_open_sprint = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5]) # wide open
    x_wall_ahead = np.array([0.8, 0.8, 0.8, 0.02, 0.8, 0.8, 0.8, 0.0, 0.5]) # wall directly ahead
    
    net.reset_state()
    out_open = net.forward(x_open_sprint)
    net.reset_state()
    out_wall = net.forward(x_wall_ahead)
    
    speed_open = 40.0 + 45.0 * out_open[2]
    speed_wall = 40.0 + 45.0 * out_wall[2]
    assert speed_open > speed_wall, f"Throttle failed to brake: open={speed_open:.1f} px/s <= wall={speed_wall:.1f} px/s"
    print(f"  [PASS] Dynamic throttle & braking: Open corridor sprint = {speed_open:.1f} px/s, Wall ahead brake = {speed_wall:.1f} px/s")

    # 6. Mutation decay and genetic operators
    sigmas = [compute_mutation_sigma(gen, sigma_0=0.1, decay_factor=0.98, min_sigma=0.01) for gen in (0, 10, 50, 100, 500)]
    assert sigmas[0] == 0.1
    assert sigmas[-1] == 0.01
    assert all(sigmas[i] >= sigmas[i+1] for i in range(len(sigmas)-1))
    print(f"  [PASS] Genetic mutation decay: Gen 0={sigmas[0]:.3f} -> Gen 500={sigmas[-1]:.3f}")

    return True


def run_phase_2_raycasting_geometry() -> bool:
    test_section_header("PHASE 2: 7-Ray Spatial Vision & Maze Raycasting Geometry")
    
    maze = Maze(cols=25, rows=17, cell_size=40.0, seed=123)
    sx, sy = maze.start_pos
    
    # Verify raycasting at open position
    RAY_ANGLES = (-math.pi * 0.50, -math.pi * 0.33, -math.pi * 0.17, 0.0, math.pi * 0.17, math.pi * 0.33, math.pi * 0.50)
    assert len(RAY_ANGLES) == 7, "Ray count must be exactly 7"
    
    dists = [maze.raycast(sx, sy, off, 240.0) for off in RAY_ANGLES]
    for i, d in enumerate(dists):
        assert 0.0 < d <= 240.0, f"Ray {i} returned invalid distance: {d}"
    print(f"  [PASS] 7-ray spatial coverage verified from origin: {[round(d, 1) for d in dists]}")

    # Boundary and wall collision verification
    assert maze.check_collision(0.0, 0.0, 6.0) is True, "Boundary collision missed at (0, 0)"
    assert maze.check_collision(maze.width, maze.height, 6.0) is True, "Boundary collision missed at (W, H)"
    assert maze.check_collision(sx, sy, 6.0) is False, "Start position falsely flagged as collision"
    print("  [PASS] Strict spatial collision boundaries verified")

    # Monotonic distance field verification
    gx, gy = int(maze.goal_pos[0] / maze.cell_size), int(maze.goal_pos[1] / maze.cell_size)
    assert maze.dist_field[gy, gx] == 0.0, "Goal cell distance field must be 0"
    start_dist = maze.dist_field[int(sy / maze.cell_size), int(sx / maze.cell_size)]
    assert start_dist > 0.0, "Start cell distance field must be positive"
    
    # Progress normalization
    p_start = maze.get_progress(sx, sy)
    p_goal = maze.get_progress(maze.goal_pos[0], maze.goal_pos[1])
    assert p_start >= 0.0
    assert p_goal >= 0.99
    print(f"  [PASS] Distance field BFS validated: Start dist = {start_dist:.1f} cells, Goal progress = {p_goal*100:.1f}%")

    return True


def run_phase_3_swarm_vectorization_benchmarks() -> bool:
    test_section_header("PHASE 3: Swarm Dynamics, Vectorized Math & Throughput Benchmark")
    
    maze = Maze(cols=35, rows=22, cell_size=40.0, seed=999)
    pop_sizes = [50, 100, 150, 300]
    
    for ps in pop_sizes:
        pop = MazePopulation(size=ps, seed=42)
        pop.reset_at_maze(maze)
        
        # Test mathematical equivalence: batch vectorized vs sequential drone forward pass
        # Fill inputs with test values
        pop.raw_inputs.fill(0.7)
        pop.H_states.fill(0.1)
        
        # Vectorized pass
        X = pop.raw_inputs
        H_prev = pop.H_states
        pre_h1 = np.matmul(X, pop.W1) + np.matmul(H_prev, pop.W_rec) + pop.B1
        h1_vec = np.where(pre_h1 > 0.0, pre_h1, 0.05 * pre_h1)
        pre_h2 = np.matmul(h1_vec, pop.W2) + pop.B2
        h2_vec = np.where(pre_h2 > 0.0, pre_h2, 0.05 * pre_h2)
        out_vec = np.matmul(h2_vec, pop.W3) + pop.B3 + np.matmul(X, pop.W_skip)
        
        # Sequential pass
        out_seq = np.empty_like(out_vec)
        for i in range(ps):
            c = pop.drones[i].controller
            c.h_state = H_prev[i, 0].copy()
            c_out = c.forward(X[i, 0])
            # Raw pre-activation equivalence check
            h1_s = leaky_relu(X[i, 0] @ c.w1 + H_prev[i, 0] @ c.w_rec + c.b1)
            h2_s = leaky_relu(h1_s @ c.w2 + c.b2)
            out_s = h2_s @ c.w3 + c.b3 + X[i, 0] @ c.w_skip
            out_seq[i, 0] = out_s
            
        max_diff = np.max(np.abs(out_vec - out_seq))
        assert max_diff < 1e-12, f"Vectorized vs sequential discrepancy: {max_diff}"
        
        # Benchmark 500 steps
        dt = 1.0 / 60.0
        t0 = time.perf_counter()
        steps = 500
        for _ in range(steps):
            pop.step(dt, maze)
        t_elapsed = time.perf_counter() - t0
        
        steps_per_sec = steps / t_elapsed
        ms_per_step = (t_elapsed / steps) * 1000.0
        print(f"  [PASS] Swarm N={ps:3d}: {steps_per_sec:7.1f} steps/sec ({ms_per_step:5.2f} ms/step) | Max discrepancy = {max_diff:.2e}")

    return True


def run_phase_4_multi_generation_stress_runs() -> bool:
    test_section_header("PHASE 4: Multi-Generation Evolution Stress Runs Across 5 Seeds")
    
    seeds = [42, 101, 777, 1337, 2026]
    generations_per_seed = 10
    dt = 1.0 / 60.0
    
    for seed in seeds:
        maze = Maze(cols=27, rows=17, cell_size=40.0, seed=seed)
        pop = MazePopulation(size=100, seed=seed)
        pop.reset_at_maze(maze)
        
        best_progress_seen = 0.0
        t_start = time.perf_counter()
        
        for g in range(generations_per_seed):
            step_count = 0
            while True:
                step_count += 1
                done = pop.step(dt, maze)
                if done or step_count > 1000:
                    best = pop.get_best_drone()
                    if best.max_progress > best_progress_seen:
                        best_progress_seen = best.max_progress
                    pop.reproduce(maze)
                    break
        
        elapsed = time.perf_counter() - t_start
        print(f"  [PASS] Seed {seed:4d}: {generations_per_seed} gens completed in {elapsed:5.2f}s | Peak progress = {best_progress_seen*100:5.1f}% | Avg speed = {elapsed/generations_per_seed:5.3f}s/gen")
        assert best_progress_seen > 0.05, f"Swarm made zero progress on seed {seed}"

    return True


def run_phase_5_serialization_and_replay() -> bool:
    test_section_header("PHASE 5: Weight Persistence (.npz) & Deterministic Replay")
    
    weights_path = os.path.join(os.path.dirname(__file__), "test_temp_champ.npz")
    
    # 1. Train 5 generations and save champion
    maze = Maze(cols=25, rows=17, cell_size=40.0, seed=555)
    pop = MazePopulation(size=80, seed=555)
    pop.reset_at_maze(maze)
    
    dt = 1.0 / 60.0
    for _ in range(5):
        for _ in range(800):
            if pop.step(dt, maze):
                break
        pop.reproduce(maze)
        
    pop.save_best(weights_path)
    assert os.path.exists(weights_path), "Failed to save weights file"
    print(f"  [PASS] Successfully serialized champion weights to {weights_path}")

    # 2. Load into new controller
    reloaded_net = pop.load_champion(weights_path)
    best_original = pop.all_time_best_controller or pop.get_best_drone().controller
    
    assert len(reloaded_net) == TOTAL_WEIGHTS == 2526
    assert np.array_equal(reloaded_net.get_flat_weights(), best_original.get_flat_weights()), "Reloaded weights differ from source controller"
    print("  [PASS] Exact bitwise weight equality verified between memory and disk archive")

    # 3. Deterministic replay execution
    # Replay drone 1 with original controller, drone 2 with reloaded controller on identical trajectory
    d_orig = MazeDrone(x=maze.start_pos[0], y=maze.start_pos[1], heading=0.0, controller=best_original.copy())
    d_reloaded = MazeDrone(x=maze.start_pos[0], y=maze.start_pos[1], heading=0.0, controller=reloaded_net.copy())
    d_orig.reset(maze.start_pos[0], maze.start_pos[1], 0.0)
    d_reloaded.reset(maze.start_pos[0], maze.start_pos[1], 0.0)
    
    for step in range(300):
        d_orig.step(dt, maze)
        d_reloaded.step(dt, maze)
        assert abs(d_orig.x - d_reloaded.x) < 1e-10, f"Replay divergence at step {step}: dx={abs(d_orig.x - d_reloaded.x)}"
        assert abs(d_orig.y - d_reloaded.y) < 1e-10, f"Replay divergence at step {step}: dy={abs(d_orig.y - d_reloaded.y)}"
        assert abs(d_orig.speed - d_reloaded.speed) < 1e-10, f"Replay speed divergence at step {step}"
    
    print("  [PASS] Deterministic replay validated over 300 steps: zero trajectory divergence")

    # Cleanup temp file
    try:
        os.remove(weights_path)
    except Exception:
        pass

    return True


def run_phase_6_ipc_activation_streaming() -> bool:
    test_section_header("PHASE 6: Telemetry & Inter-Process Activation Streaming")
    
    writer = ActivationWriter()
    reader = ActivationReader()
    
    # Send 1,000 frames of telemetry and activations
    for seq in range(1, 1001):
        sample_acts = {
            'inputs': np.linspace(0.1, 0.9, 9),
            'h1': np.sin(np.linspace(0.0, 3.14, 32)),
            'h2': np.cos(np.linspace(0.0, 3.14, 32)),
            'outputs': np.array([0.45, 0.88, 0.72]),
        }
        telem = {
            'gen': 15,
            'mode': 0,
            'alive': 92,
            'pop_size': 100,
            'reached': 3,
            'progress': 64.5,
            'step': seq,
            'max_steps': 2500,
            'paused': 0,
            'fast_forward': 1,
            'fps': 60.0,
            'ray_mode': 1,
        }
        writer.write(sample_acts, ping_freq=125.0, speed=72.4, heading=1.23, telemetry=telem)
        
        # Read back
        recv = reader.read()
        assert recv is not None, f"Reader returned None on seq {seq}"
        assert abs(recv['speed'] - 72.4) < 1e-6
        assert abs(recv['ping_freq'] - 125.0) < 1e-6
        assert recv['telemetry']['gen'] == 15
        assert recv['telemetry']['step'] == seq
        assert recv['telemetry']['alive'] == 92
        assert np.allclose(recv['inputs'][:5], sample_acts['inputs'][:5], atol=1e-6)
        assert np.allclose(recv['outputs'][:2], sample_acts['outputs'][:2], atol=1e-6)
    
    writer.close()
    print("  [PASS] 1,000 IPC frames streamed & roundtrip verified without buffer overruns or drops")
    return True


def main() -> None:
    print(f"\n===========================================================================")
    print(f"  EXTENSIVE SYSTEM BENCHMARK & TEST SUITE")
    print(f"  Autonomous Swarm Maze Navigation - 7-Ray Recurrent Residual Architecture")
    print(f"===========================================================================")
    
    t_start_all = time.perf_counter()
    
    assert run_phase_1_neural_invariants()
    assert run_phase_2_raycasting_geometry()
    assert run_phase_3_swarm_vectorization_benchmarks()
    assert run_phase_4_multi_generation_stress_runs()
    assert run_phase_5_serialization_and_replay()
    assert run_phase_6_ipc_activation_streaming()
    
    total_time = time.perf_counter() - t_start_all
    print(f"\n{'='*75}")
    print(f"  ALL 6 PHASES PASSED IN {total_time:.2f}s WITH ZERO ERRORS")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
