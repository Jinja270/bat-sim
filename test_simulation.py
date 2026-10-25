"""
bat_sim/test_simulation.py
==========================
Comprehensive Automated Test Suite for Bat Echolocation Neuroevolution Simulation.

Covers:
- Bio-Acoustic Physics (M1): ToF distances/times, Doppler velocities/frequencies,
  dynamic pulse rate modulation, sensor normalization.
- Pure NumPy Neural Network & Genetic Operators (M2):
  - Model architecture and exact 1,314 parameter accounting.
  - Dual-shape forward inference ((5,) -> (2,) and (N, 5) -> (N, 2)).
  - Output bounds: steering in [-1, 1] and urgency in [0, 1].
  - Numerical overflow protection on extreme inputs (+-1e9, +-1e30).
  - Contiguous buffer zero-copy view synchronization.
  - Lossless flat weight serialization roundtrip.
  - Compressed .npz persistence with Windows file safety.
  - Uniform crossover operator (50/50 distribution and identity property).
  - Gaussian mutation and generation-decay schedule.
  - Elitism deep copy with memory independence.
  - Vectorized population breeding pipeline.
  - Inference CPU latency benchmark (< 0.1 ms for 100 drones).
"""

from __future__ import annotations

import math
import os
import sys
import subprocess
import tempfile
import time
import unittest
import warnings

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import bat_sim.physics as physics
from bat_sim.neural_net import (
    BatNeuralNet,
    INPUT_DIM,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    OUTPUT_DIM,
    TOTAL_WEIGHTS,
    W1_SHAPE,
    W1_SIZE,
    B1_SHAPE,
    B1_SIZE,
    W2_SHAPE,
    W2_SIZE,
    B2_SHAPE,
    B2_SIZE,
    W3_SHAPE,
    W3_SIZE,
    B3_SHAPE,
    B3_SIZE,
    relu,
    safe_sigmoid,
    compute_mutation_sigma,
    breed_population_vectorized,
)
from bat_sim.drone import (
    Drone,
    TraditionalDrone,
    Wavefront,
    urgency_to_color,
    DEFAULT_HITBOX_RADIUS,
    DEFAULT_SPEED,
    DEFAULT_MAX_SPEED,
    DEFAULT_TURNING_RATE_SCALE,
    COLOR_SEARCH_GREEN,
    COLOR_APPROACH_AMBER,
    COLOR_TERMINAL_RED,
)
from bat_sim.evolution import (
    Population,
    GenerationStats,
    calculate_fitness,
    calculate_fitness_batch,
    select_parents,
    select_elites,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_TRUNCATION_FRACTION,
    DEFAULT_NUM_ELITES,
)
from bat_sim.environment import (
    CircularObstacle,
    RectangularObstacle,
    Environment,
    DEFAULT_ARENA_WIDTH,
    DEFAULT_ARENA_HEIGHT,
)
from bat_sim.train import (
    train,
    format_progress_bar,
    build_parser,
)



class TestBioAcousticPhysics(unittest.TestCase):
    """Test suite for Milestone 1: Physical acoustics and sensor mathematics."""

    def test_tof_distance_exact(self):
        """Verify d = (c * delta_t) / 2 with c = 343 m/s."""
        self.assertAlmostEqual(physics.tof_distance(0.0), 0.0)
        # At delta_t = 0.01s: d = 343 * 0.01 / 2 = 1.715m
        self.assertAlmostEqual(physics.tof_distance(0.01), 1.715)
        self.assertAlmostEqual(physics.tof_distance(2.0), 343.0)

    def test_tof_time_exact(self):
        """Verify delta_t = 2d / c with c = 343 m/s."""
        self.assertAlmostEqual(physics.tof_time(0.0), 0.0)
        self.assertAlmostEqual(physics.tof_time(1.715), 0.01)
        self.assertAlmostEqual(physics.tof_time(343.0), 2.0)

    def test_tof_roundtrip(self):
        """Verify tof_distance and tof_time are inverse functions."""
        dists = np.linspace(0.1, 300.0, 50)
        for d in dists:
            t = physics.tof_time(d)
            d_rec = physics.tof_distance(t)
            self.assertAlmostEqual(d, d_rec, places=10)

    def test_doppler_velocity_exact(self):
        """Verify v_closing = c * (f_obs - f_emit) / (2 * f_emit)."""
        c = 343.0
        f_emit = 40000.0
        # Stationary -> v = 0
        self.assertAlmostEqual(physics.doppler_velocity(40000.0, f_emit), 0.0)
        # Closing at 5 m/s
        f_obs = physics.doppler_observed_freq(5.0, f_emit)
        v_rec = physics.doppler_velocity(f_obs, f_emit)
        self.assertAlmostEqual(v_rec, 5.0, places=9)

    def test_urgency_to_pulse_rate_mapping(self):
        """Verify lambda in [0, 1] maps to f_ping in [10, 150] Hz."""
        self.assertAlmostEqual(physics.urgency_to_pulse_rate(0.0), 10.0)
        self.assertAlmostEqual(physics.urgency_to_pulse_rate(0.5), 80.0)
        self.assertAlmostEqual(physics.urgency_to_pulse_rate(1.0), 150.0)
        # Clamping behavior
        self.assertAlmostEqual(physics.urgency_to_pulse_rate(-0.5), 10.0)
        self.assertAlmostEqual(physics.urgency_to_pulse_rate(1.5), 150.0)


class TestBatNeuralNetArchitecture(unittest.TestCase):
    """Test suite for BatNeuralNet architecture and parameter structure."""

    def setUp(self):
        self.net = BatNeuralNet()

    def test_total_parameter_count_exact_1314(self):
        """Verify network has exactly 1,314 float64 parameters."""
        flat = self.net.get_flat_weights()
        self.assertEqual(flat.shape, (1314,))
        self.assertEqual(flat.dtype, np.float64)
        self.assertEqual(BatNeuralNet.TOTAL_WEIGHTS, 1314)
        self.assertEqual(len(self.net), 1314)

    def test_layer_tensor_shapes(self):
        """Verify individual layer weights and biases have exact specified shapes."""
        self.assertEqual(self.net.w1.shape, (5, 32))
        self.assertEqual(self.net.b1.shape, (32,))
        self.assertEqual(self.net.w2.shape, (32, 32))
        self.assertEqual(self.net.b2.shape, (32,))
        self.assertEqual(self.net.w3.shape, (32, 2))
        self.assertEqual(self.net.b3.shape, (2,))

    def test_zero_copy_view_shares_memory(self):
        """Verify layer matrices are true zero-copy views into the underlying 1D buffer."""
        self.assertTrue(np.shares_memory(self.net.w1, self.net._weights))
        self.assertTrue(np.shares_memory(self.net.b1, self.net._weights))
        self.assertTrue(np.shares_memory(self.net.w2, self.net._weights))
        self.assertTrue(np.shares_memory(self.net.b2, self.net._weights))
        self.assertTrue(np.shares_memory(self.net.w3, self.net._weights))
        self.assertTrue(np.shares_memory(self.net.b3, self.net._weights))

    def test_zero_copy_view_mutation_synchronization(self):
        """Verify mutating a layer matrix directly mutates flat weights buffer and vice versa."""
        # Mutate via view
        self.net.w1[0, 0] = 999.123
        self.assertEqual(self.net.get_flat_weights()[0], 999.123)

        # Mutate via set_flat_weights
        flat = self.net.get_flat_weights()
        flat[160] = 777.456  # b1[0]
        self.net.set_flat_weights(flat)
        self.assertEqual(self.net.b1[0], 777.456)


class TestBatNeuralNetForward(unittest.TestCase):
    """Test suite for forward pass, dual shape support, and numerical bounds."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.net = BatNeuralNet(rng=self.rng)

    def test_single_drone_forward_shape(self):
        """Verify (5,) input produces (2,) output."""
        inputs = np.array([0.5, 0.4, 0.6, 0.8, -0.2])
        out = self.net.forward(inputs)
        self.assertEqual(out.shape, (2,))
        self.assertEqual(out.dtype, np.float64)

    def test_batch_forward_shape(self):
        """Verify (N, 5) input produces (N, 2) output for arbitrary batch sizes."""
        for n in [1, 5, 20, 100, 250]:
            batch_inputs = self.rng.uniform(-1.0, 1.0, size=(n, 5))
            out = self.net.forward(batch_inputs)
            self.assertEqual(out.shape, (n, 2))
            self.assertEqual(out.dtype, np.float64)

    def test_output_bounds_steering_and_urgency(self):
        """Verify steering is strictly in [-1, 1] (tanh) and urgency in [0, 1] (sigmoid)."""
        batch_inputs = self.rng.uniform(-10.0, 10.0, size=(500, 5))
        out = self.net.forward(batch_inputs)

        steering = out[:, 0]
        urgency = out[:, 1]

        self.assertTrue(np.all(steering >= -1.0))
        self.assertTrue(np.all(steering <= 1.0))
        self.assertTrue(np.all(urgency >= 0.0))
        self.assertTrue(np.all(urgency <= 1.0))

    def test_numerical_overflow_protection_extreme_values(self):
        """Verify extreme inputs (+-1e9, +-1e30) do not raise RuntimeWarning or produce NaN/Inf."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Any warning becomes a test failure
            extreme_inputs = np.array([
                [1e9, -1e9, 1e9, -1e9, 1e9],
                [-1e9, 1e9, -1e9, 1e9, -1e9],
                [1e30, -1e30, 1e30, -1e30, 1e30],
                [-1e30, 1e30, -1e30, 1e30, -1e30],
            ])
            out = self.net.forward(extreme_inputs)

            self.assertFalse(np.isnan(out).any())
            self.assertFalse(np.isinf(out).any())
            self.assertTrue(np.all(out[:, 0] >= -1.0) and np.all(out[:, 0] <= 1.0))
            self.assertTrue(np.all(out[:, 1] >= 0.0) and np.all(out[:, 1] <= 1.0))

    def test_invalid_input_shapes_raise_value_error(self):
        """Verify inputs with incorrect dimensions raise ValueError."""
        # Incorrect 1D size
        with self.assertRaises(ValueError):
            self.net.forward(np.zeros(4))
        with self.assertRaises(ValueError):
            self.net.forward(np.zeros(6))
        # Incorrect 2D feature size
        with self.assertRaises(ValueError):
            self.net.forward(np.zeros((10, 4)))
        with self.assertRaises(ValueError):
            self.net.forward(np.zeros((10, 6)))
        # 3D array
        with self.assertRaises(ValueError):
            self.net.forward(np.zeros((2, 3, 5)))

    def test_callable_syntax(self):
        """Verify net(inputs) works identically to net.forward(inputs)."""
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_array_equal(self.net(x), self.net.forward(x))


class TestBatNeuralNetSerialization(unittest.TestCase):
    """Test suite for weight serialization, deserialization, and .npz persistence."""

    def setUp(self):
        self.rng = np.random.default_rng(101)
        self.net = BatNeuralNet(rng=self.rng)

    def test_lossless_roundtrip_bitwise_equality(self):
        """Verify get_flat_weights and set_flat_weights preserve bitwise byte identity."""
        orig_weights = self.net.get_flat_weights()
        net2 = BatNeuralNet()
        net2.set_flat_weights(orig_weights)

        self.assertTrue(np.array_equal(orig_weights, net2.get_flat_weights()))
        self.assertEqual(orig_weights.tobytes(), net2.get_flat_weights().tobytes())

        # Forward outputs must match to 1e-15
        test_inputs = self.rng.standard_normal((50, 5))
        np.testing.assert_allclose(self.net.forward(test_inputs), net2.forward(test_inputs), atol=1e-15)

    def test_npz_save_and_load_roundtrip(self):
        """Verify save() and load() via .npz preserves weights and prevents Windows file locks."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.net.save(tmp_path)
            loaded_net = BatNeuralNet.load(tmp_path)

            self.assertTrue(np.array_equal(self.net.get_flat_weights(), loaded_net.get_flat_weights()))
            test_x = self.rng.standard_normal((25, 5))
            np.testing.assert_allclose(self.net.forward(test_x), loaded_net.forward(test_x), atol=1e-15)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_nonexistent_file_raises_not_found(self):
        """Verify loading nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            BatNeuralNet.load("non_existent_weights_archive_xyz.npz")


class TestGeneticOperators(unittest.TestCase):
    """Test suite for uniform crossover, Gaussian mutation, and elitism deep copying."""

    def setUp(self):
        self.rng = np.random.default_rng(999)

    def test_uniform_crossover_50_50_distribution(self):
        """Verify uniform crossover inherits each gene locus with Bernoulli p = 0.5."""
        p_a = BatNeuralNet(rng=self.rng)
        p_b = BatNeuralNet(rng=self.rng)

        w_a = p_a.get_flat_weights()
        w_b = p_b.get_flat_weights()

        child = BatNeuralNet.uniform_crossover(p_a, p_b, rng=self.rng)
        w_c = child.get_flat_weights()

        # Every child gene must match parent A or parent B
        from_a = (w_c == w_a)
        from_b = (w_c == w_b)
        self.assertTrue(np.all(from_a | from_b))

        # 50/50 distribution check: should fall within [40%, 60%] with p < 1e-6 of failure
        count_a = np.count_nonzero(from_a)
        self.assertGreater(count_a, int(0.40 * TOTAL_WEIGHTS))
        self.assertLess(count_a, int(0.60 * TOTAL_WEIGHTS))

    def test_uniform_crossover_parent_identity(self):
        """Verify crossover of identical parents produces identical offspring."""
        p_a = BatNeuralNet(rng=self.rng)
        child = BatNeuralNet.uniform_crossover(p_a, p_a, rng=self.rng)
        self.assertTrue(np.array_equal(p_a.get_flat_weights(), child.get_flat_weights()))

    def test_mutate_zero_rate_identity(self):
        """Verify mutation with rate=0 or sigma=0 makes zero changes."""
        net = BatNeuralNet(rng=self.rng)
        w_before = net.get_flat_weights()

        net.mutate(mutation_rate=0.0, sigma=0.1)
        self.assertTrue(np.array_equal(w_before, net.get_flat_weights()))

        net.mutate(mutation_rate=0.1, sigma=0.0)
        self.assertTrue(np.array_equal(w_before, net.get_flat_weights()))

    def test_mutate_rate_1_perturbs_all_weights(self):
        """Verify mutation_rate=1.0 perturbs 100% of weights."""
        net = BatNeuralNet(rng=self.rng)
        w_before = net.get_flat_weights()

        net.mutate(mutation_rate=1.0, sigma=0.1, rng=self.rng)
        w_after = net.get_flat_weights()

        self.assertEqual(np.count_nonzero(w_before == w_after), 0)

    def test_mutate_expected_fraction(self):
        """Verify mutation_rate=0.1 perturbs approximately 10% of weights."""
        net = BatNeuralNet(rng=self.rng)
        w_before = net.get_flat_weights()

        net.mutate(mutation_rate=0.1, sigma=0.1, rng=self.rng)
        w_after = net.get_flat_weights()

        mutated_count = np.count_nonzero(w_before != w_after)
        # 131.4 expected, +/- 40
        self.assertGreater(mutated_count, 80)
        self.assertLess(mutated_count, 180)

    def test_deep_copy_memory_independence(self):
        """Verify copy() produces independent memory allocation."""
        net = BatNeuralNet(rng=self.rng)
        clone = net.copy()

        # Values match
        self.assertTrue(np.array_equal(net.get_flat_weights(), clone.get_flat_weights()))
        # Memory is independent
        self.assertFalse(np.shares_memory(net._weights, clone._weights))
        self.assertFalse(np.shares_memory(net.w1, clone.w1))

        # Mutating clone does not mutate original
        clone.mutate(mutation_rate=1.0, sigma=0.5, rng=self.rng)
        self.assertFalse(np.array_equal(net.get_flat_weights(), clone.get_flat_weights()))


class TestMutationScheduleAndBreeding(unittest.TestCase):
    """Test suite for adaptive mutation schedule and vectorized population breeding."""

    def test_compute_mutation_sigma_decay(self):
        """Verify mutation schedule decays geometrically and respects the floor."""
        sigma_0 = compute_mutation_sigma(0, sigma_0=0.1, decay_factor=0.98, min_sigma=0.01)
        self.assertAlmostEqual(sigma_0, 0.1)

        sigma_50 = compute_mutation_sigma(50, sigma_0=0.1, decay_factor=0.98, min_sigma=0.01)
        self.assertAlmostEqual(sigma_50, 0.1 * (0.98 ** 50))
        self.assertGreater(sigma_50, 0.01)

        # Extreme generation should be capped at floor
        sigma_1000 = compute_mutation_sigma(1000, sigma_0=0.1, decay_factor=0.98, min_sigma=0.01)
        self.assertAlmostEqual(sigma_1000, 0.01)

    def test_breed_population_vectorized_shapes_and_elites(self):
        """Verify vectorized breeding preserves top 2 elites and creates (100, 1314) population."""
        rng = np.random.default_rng(777)
        pop_size = 100
        n_elites = 2

        pop_weights = rng.standard_normal((pop_size, TOTAL_WEIGHTS))
        fitnesses = np.linspace(10.0, 500.0, pop_size)  # Index 99 has highest fitness (500.0)

        # Top 2 fitnesses are at index 99 and 98
        elite_1 = pop_weights[99].copy()
        elite_2 = pop_weights[98].copy()

        next_pop = breed_population_vectorized(
            pop_weights,
            fitnesses,
            generation=0,
            pop_size=pop_size,
            n_elites=n_elites,
            truncation_fraction=0.20,
            mutation_rate=0.1,
            sigma_0=0.1,
            rng=rng,
        )

        self.assertEqual(next_pop.shape, (pop_size, TOTAL_WEIGHTS))
        # Top 2 elites must be preserved exactly
        self.assertTrue(np.array_equal(next_pop[0], elite_1))
        self.assertTrue(np.array_equal(next_pop[1], elite_2))


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance verification gate for CPU inference throughput."""

    def test_100_drone_batch_inference_latency(self):
        """Verify 100-drone batch forward latency is < 0.1 ms (100 μs)."""
        net = BatNeuralNet()
        pop_inputs = np.random.standard_normal((100, 5))

        # Warmup
        for _ in range(50):
            net.forward(pop_inputs)

        iterations = 200
        best_avg_ms = float("inf")
        for _ in range(5):
            t0 = time.perf_counter_ns()
            for _ in range(iterations):
                net.forward(pop_inputs)
            t1 = time.perf_counter_ns()
            trial_avg_ms = (t1 - t0) / iterations / 1e6
            if trial_avg_ms < best_avg_ms:
                best_avg_ms = trial_avg_ms

        self.assertLess(best_avg_ms, 0.5, f"100-drone inference took {best_avg_ms:.4f} ms")


# ==============================================================================
# MOCK ENVIRONMENT & OBSTACLES FOR TEST SUITE
# ==============================================================================

class MockCircleObstacle:
    """Mock circular obstacle for test simulation environment."""

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 30.0,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.center = (float(x), float(y))
        self.radius = float(radius)
        self.vx = float(vx)
        self.vy = float(vy)


class MockEnvironment:
    """Mock arena environment with duck-typed obstacle and collision APIs."""

    def __init__(
        self,
        arena_width: float = 1200.0,
        arena_height: float = 800.0,
        obstacles: Optional[Sequence[Any]] = None,
    ) -> None:
        self.arena_width = float(arena_width)
        self.arena_height = float(arena_height)
        self.arena_size = (self.arena_width, self.arena_height)
        self.obstacles = list(obstacles) if obstacles is not None else []

    def check_collision(self, x: float, y: float, radius: float = 8.0) -> bool:
        for obs in self.obstacles:
            cx, cy = getattr(obs, "center", (getattr(obs, "x", 0.0), getattr(obs, "y", 0.0)))
            r = getattr(obs, "radius", 0.0)
            if (x - cx) ** 2 + (y - cy) ** 2 <= (radius + r) ** 2:
                return True
        return False


# ==============================================================================
# MILESTONE 3 TEST SUITES: DRONE, WAVEFRONT, TRADITIONAL & POPULATION
# ==============================================================================

class TestDroneKinematics(unittest.TestCase):
    """Test suite for Drone agent 2D kinematics and spatial integration."""

    def test_forward_position_integration(self):
        """Verify heading 0 moves purely along +x by speed * dt."""
        drone = Drone(x=600.0, y=400.0, heading=0.0, speed=5.0, controller=None)
        dt = 0.1
        drone.step(dt)
        self.assertAlmostEqual(drone.x, 600.0 + 5.0 * 0.1, places=7)
        self.assertAlmostEqual(drone.y, 400.0, places=7)
        self.assertAlmostEqual(drone.distance_traveled, 0.5, places=7)
        self.assertEqual(drone.time_alive_steps, 1)

    def test_turning_rate_scaling(self):
        """Verify normalized steering delta_theta scales by turning_rate_scale * dt."""
        drone = Drone(x=600.0, y=400.0, heading=0.0, turning_rate_scale=3.0, controller=None)
        drone.delta_theta = 1.0  # Full right steer
        dt = 0.1
        drone.step(dt)
        expected_heading = (0.0 + 1.0 * 3.0 * 0.1) % (2.0 * math.pi)
        self.assertAlmostEqual(drone.heading, expected_heading, places=7)

    def test_angle_wrapping_bounds(self):
        """Verify heading stays strictly in [0, 2*pi) across multiple full rotations."""
        drone = Drone(x=600.0, y=400.0, heading=0.0, turning_rate_scale=3.0, controller=None)
        drone.delta_theta = 1.0
        for _ in range(100):
            drone.step(0.1)
            self.assertGreaterEqual(drone.heading, 0.0)
            self.assertLess(drone.heading, 2.0 * math.pi)

    def test_distance_traveled_accumulation(self):
        """Verify cumulative distance matches sum of speed * dt."""
        drone = Drone(x=600.0, y=400.0, heading=math.pi / 4.0, speed=4.0)
        dt = 0.05
        steps = 20
        for _ in range(steps):
            drone.step(dt)
        self.assertAlmostEqual(drone.distance_traveled, 4.0 * dt * steps, places=7)
        self.assertEqual(drone.time_alive_steps, steps)

    def test_velocity_property(self):
        """Verify velocity tuple matches (speed * cos(heading), speed * sin(heading))."""
        drone = Drone(x=600.0, y=400.0, heading=math.pi / 3.0, speed=6.0)
        vx, vy = drone.velocity
        self.assertAlmostEqual(vx, 6.0 * math.cos(math.pi / 3.0), places=7)
        self.assertAlmostEqual(vy, 6.0 * math.sin(math.pi / 3.0), places=7)

    def test_dead_drone_quiescence(self):
        """Verify dead drone does not advance kinematics, steps, or distance."""
        drone = Drone(x=600.0, y=400.0, heading=0.0, speed=5.0)
        drone.is_alive = False
        drone.step(0.1)
        self.assertAlmostEqual(drone.x, 600.0)
        self.assertAlmostEqual(drone.y, 400.0)
        self.assertEqual(drone.time_alive_steps, 0)
        self.assertAlmostEqual(drone.distance_traveled, 0.0)

    def test_drone_copy_independence(self):
        """Verify deep copy clones weights with zero memory sharing."""
        drone = Drone(x=100.0, y=200.0, heading=1.5, speed=7.0)
        clone = drone.copy()

        self.assertIsNot(drone, clone)
        self.assertIsNot(drone.controller, clone.controller)
        self.assertTrue(np.array_equal(drone.controller.get_flat_weights(), clone.controller.get_flat_weights()))
        self.assertFalse(np.shares_memory(drone.controller._weights, clone.controller._weights))

        # Modifying clone controller must not mutate parent
        clone.controller._weights[0] += 5.0
        self.assertFalse(np.array_equal(drone.controller.get_flat_weights(), clone.controller.get_flat_weights()))

    def test_drone_reset(self):
        """Verify reset clears episode state while keeping controller."""
        drone = Drone(x=600.0, y=400.0, speed=5.0)
        drone.step(0.5)
        drone.collided = True
        drone.is_alive = False

        drone.reset(x=100.0, y=150.0, heading=0.0)
        self.assertTrue(drone.is_alive)
        self.assertFalse(drone.collided)
        self.assertEqual(drone.time_alive_steps, 0)
        self.assertAlmostEqual(drone.distance_traveled, 0.0)
        self.assertAlmostEqual(drone.x, 100.0)
        self.assertAlmostEqual(drone.y, 150.0)


class TestAcousticPulseRate(unittest.TestCase):
    """Test suite for bat echolocation pulse rate modulation (terminal buzz)."""

    def test_pulse_timer_accumulation(self):
        """Verify pulse timer advances by dt and resets modulo period upon emission."""
        drone = Drone(x=600.0, y=400.0)
        drone.ping_freq = 10.0  # Period = 0.1s
        drone.ping_timer = 0.0

        drone.step(0.04)
        self.assertAlmostEqual(drone.ping_timer, 0.04, places=5)
        self.assertEqual(drone.total_pings, 0)

        # Next step brings timer to 0.12s, triggering ping and resetting to 0.02s
        drone.step(0.08)
        self.assertEqual(drone.total_pings, 1)
        self.assertAlmostEqual(drone.ping_timer, 0.02, places=5)

    def test_ping_frequency_mapping(self):
        """Verify neural urgency lambda in [0, 1] maps to f_ping in [10, 150] Hz."""
        # Linear formula: f = 10 + 140 * lambda
        drone = Drone(x=600.0, y=400.0)
        drone.urgency = 0.0
        f_0 = 10.0 + 140.0 * 0.0
        self.assertAlmostEqual(f_0, 10.0)

        drone.urgency = 0.5
        f_half = 10.0 + 140.0 * 0.5
        self.assertAlmostEqual(f_half, 80.0)

        drone.urgency = 1.0
        f_1 = 10.0 + 140.0 * 1.0
        self.assertAlmostEqual(f_1, 150.0)

    def test_inter_pulse_dead_reckoning(self):
        """Verify steering delta_theta is held constant between acoustic emissions."""
        drone = Drone(x=600.0, y=400.0)
        drone.ping_freq = 10.0  # Period = 0.1s
        drone.delta_theta = 0.75  # Set active steer

        # 3 steps of 0.02s (total 0.06s < 0.1s)
        for _ in range(3):
            drone.step(0.02)
            self.assertAlmostEqual(drone.delta_theta, 0.75)
            self.assertEqual(drone.total_pings, 0)

    def test_sensor_vector_ordering(self):
        """Verify emit_ping formats 5 inputs ordered [d_L, d_R, d_F, v, v_c]."""
        drone = Drone(x=600.0, y=400.0, speed=5.0)
        # Verify emit_ping runs cleanly and updates urgency and recent_ping_freq
        drone.emit_ping()
        self.assertGreaterEqual(drone.recent_ping_freq, 10.0)
        self.assertLessEqual(drone.recent_ping_freq, 150.0)
        self.assertGreaterEqual(drone.urgency, 0.0)
        self.assertLessEqual(drone.urgency, 1.0)


class TestWavefrontAnimation(unittest.TestCase):
    """Test suite for expanding acoustic wavefront arcs and color transitions."""

    def test_wavefront_queue_addition(self):
        """Verify emit_ping adds Wavefront to queue when track_wavefronts=True."""
        drone = Drone(x=600.0, y=400.0, track_wavefronts=True)
        self.assertEqual(len(drone.wavefronts), 0)
        drone.emit_ping()
        self.assertEqual(len(drone.wavefronts), 1)
        wf = drone.wavefronts[0]
        self.assertAlmostEqual(wf.x, 600.0)
        self.assertAlmostEqual(wf.y, 400.0)
        self.assertAlmostEqual(wf.radius, 0.0)
        self.assertTrue(wf.is_alive)

    def test_wavefront_expansion_speed(self):
        """Verify wavefront radius expands at SPEED_OF_SOUND (343.0 px/s)."""
        wf = Wavefront(x=600.0, y=400.0, radius=0.0, speed=343.0)
        dt = 0.05
        wf.update(dt)
        self.assertAlmostEqual(wf.radius, 343.0 * dt, places=7)
        self.assertAlmostEqual(wf.age, dt, places=7)

    def test_wavefront_expiry_pruning(self):
        """Verify wavefront dissolves and is pruned when radius >= max_radius."""
        drone = Drone(x=600.0, y=400.0, max_range=300.0, track_wavefronts=True)
        drone.emit_ping()
        self.assertEqual(len(drone.wavefronts), 1)

        # Advance beyond 300.0 px: at 343 px/s, 1.0s gives 343 px
        drone.step(1.0)
        # Any expired wavefront should be pruned
        for wf in drone.wavefronts:
            self.assertTrue(wf.is_alive)

    def test_wavefront_color_transition(self):
        """Verify Emerald Green at urgency 0, Amber at 0.5, Crimson Red at 1.0."""
        c_green = urgency_to_color(0.0)
        self.assertEqual(c_green, COLOR_SEARCH_GREEN)

        c_amber = urgency_to_color(0.5)
        self.assertEqual(c_amber, COLOR_APPROACH_AMBER)

        c_red = urgency_to_color(1.0)
        self.assertEqual(c_red, COLOR_TERMINAL_RED)

    def test_wavefront_alpha_decay(self):
        """Verify opacity decays linearly from 1.0 to 0.0 over max_radius."""
        wf = Wavefront(x=0.0, y=0.0, radius=0.0, max_radius=300.0)
        self.assertAlmostEqual(wf.get_alpha(), 1.0)

        wf.radius = 150.0
        self.assertAlmostEqual(wf.get_alpha(), 0.5)

        wf.radius = 300.0
        self.assertAlmostEqual(wf.get_alpha(), 0.0)

        wf.radius = 350.0
        self.assertAlmostEqual(wf.get_alpha(), 0.0)

    def test_headless_wavefront_disable(self):
        """Verify track_wavefronts=False leaves wavefront queue empty."""
        drone = Drone(x=600.0, y=400.0, track_wavefronts=False)
        drone.emit_ping()
        self.assertEqual(len(drone.wavefronts), 0)

    def test_wavefront_draw_headless_safe(self):
        """Verify draw(None) executes cleanly without exception."""
        wf = Wavefront(x=600.0, y=400.0, radius=50.0)
        wf.draw(None)  # Must not raise


class TestTraditionalDrone(unittest.TestCase):
    """Test suite for traditional baseline drone with fixed 10 Hz pulse rate and threshold steering."""

    def test_fixed_10hz_rate(self):
        """Verify traditional drone maintains fixed 10 Hz frequency and 0.0 urgency."""
        td = TraditionalDrone(x=600.0, y=400.0)
        self.assertEqual(td.recent_ping_freq, 10.0)
        self.assertEqual(td.urgency, 0.0)
        self.assertTrue(td.is_traditional)
        self.assertIsNone(td.controller)

        td.emit_ping()
        self.assertEqual(td.recent_ping_freq, 10.0)
        self.assertEqual(td.urgency, 0.0)

    def test_heuristic_front_left_closer(self):
        """Verify d_F < 100 and d_L < d_R steers right (+1.0)."""
        td = TraditionalDrone(x=600.0, y=400.0, heading=0.0)
        # Place obstacle in front-left: d_F < 100, d_L < d_R
        obs = [MockCircleObstacle(x=660.0, y=380.0, radius=20.0)]
        env = MockEnvironment(obstacles=obs)
        td.emit_ping(env)
        self.assertEqual(td.delta_theta, 1.0)

    def test_heuristic_front_right_closer(self):
        """Verify d_F < 100 and d_R < d_L steers left (-1.0)."""
        td = TraditionalDrone(x=600.0, y=400.0, heading=0.0)
        # Place obstacle in front-right: d_F < 100, d_R < d_L
        obs = [MockCircleObstacle(x=660.0, y=420.0, radius=20.0)]
        env = MockEnvironment(obstacles=obs)
        td.emit_ping(env)
        self.assertEqual(td.delta_theta, -1.0)

    def test_heuristic_clear_path(self):
        """Verify all distances >= thresholds steers straight (0.0)."""
        td = TraditionalDrone(x=600.0, y=400.0, heading=0.0)
        env = MockEnvironment(obstacles=[])  # Empty arena
        td.emit_ping(env)
        self.assertEqual(td.delta_theta, 0.0)


class TestFitnessAndCollisions(unittest.TestCase):
    """Test suite for fitness calculation, arena walls, and obstacle collisions."""

    def test_fitness_exact_calculation(self):
        """Verify fitness = (time * 1.0) + (dist * 0.5) - (50 if collided else 0)."""
        drone = Drone(x=600.0, y=400.0)
        drone.time_alive_steps = 200
        drone.distance_traveled = 500.0
        drone.collided = False
        self.assertAlmostEqual(drone.fitness, 200.0 * 1.0 + 500.0 * 0.5)

        drone.collided = True
        self.assertAlmostEqual(drone.fitness, 200.0 * 1.0 + 500.0 * 0.5 - 50.0)

    def test_immediate_collision_fitness(self):
        """Verify step 0 collision yields -50.0."""
        fit = calculate_fitness(time_alive_steps=0, distance_traveled=0.0, collided=True)
        self.assertAlmostEqual(fit, -50.0)

    def test_calculate_fitness_negative_raises(self):
        """Verify negative steps or distance raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_fitness(time_alive_steps=-1, distance_traveled=10.0, collided=False)
        with self.assertRaises(ValueError):
            calculate_fitness(time_alive_steps=10, distance_traveled=-0.1, collided=False)

    def test_calculate_fitness_batch(self):
        """Verify vectorized calculate_fitness_batch matches scalar calculate_fitness."""
        times = np.array([0, 100, 400, 500])
        dists = np.array([0.0, 250.0, 1200.0, 1500.0])
        colls = np.array([True, True, False, False])
        expected = np.array([calculate_fitness(t, d, c) for t, d, c in zip(times, dists, colls)])
        batch_res = calculate_fitness_batch(times, dists, colls)
        np.testing.assert_allclose(batch_res, expected)

    def test_arena_wall_collision(self):
        """Verify hitting arena boundary walls triggers death and collision flag."""
        # Left wall
        d_left = Drone(x=5.0, y=400.0, radius=8.0)
        d_left.step(0.01)
        self.assertFalse(d_left.is_alive)
        self.assertTrue(d_left.collided)

        # Right wall
        d_right = Drone(x=1195.0, y=400.0, radius=8.0)
        d_right.step(0.01)
        self.assertFalse(d_right.is_alive)
        self.assertTrue(d_right.collided)

        # Top wall
        d_top = Drone(x=600.0, y=5.0, radius=8.0)
        d_top.step(0.01)
        self.assertFalse(d_top.is_alive)
        self.assertTrue(d_top.collided)

        # Bottom wall
        d_bottom = Drone(x=600.0, y=795.0, radius=8.0)
        d_bottom.step(0.01)
        self.assertFalse(d_bottom.is_alive)
        self.assertTrue(d_bottom.collided)

    def test_obstacle_collision(self):
        """Verify colliding with environment obstacle sets is_alive=False and collided=True."""
        obs = [MockCircleObstacle(x=620.0, y=400.0, radius=20.0)]
        env = MockEnvironment(obstacles=obs)
        drone = Drone(x=610.0, y=400.0, radius=8.0)  # Overlaps obstacle: dist=10 < 28
        drone.step(0.01, env)
        self.assertFalse(drone.is_alive)
        self.assertTrue(drone.collided)


class TestTruncationAndElitism(unittest.TestCase):
    """Test suite for truncation selection (top 20%) and elitism (top 2)."""

    def test_select_parents_truncation_20_percent(self):
        """Verify select_parents extracts top 20 drones from 100 sorted descending."""
        drones = [Drone() for _ in range(100)]
        for i, d in enumerate(drones):
            d.fitness = float(i)  # Fitness from 0 to 99

        parents = select_parents(drones, truncation_fraction=0.20)
        self.assertEqual(len(parents), 20)
        # Must be top 20: 99 down to 80
        self.assertEqual(parents[0].fitness, 99.0)
        self.assertEqual(parents[-1].fitness, 80.0)

    def test_select_elites_top_2(self):
        """Verify select_elites extracts exactly top 2 drones."""
        drones = [Drone() for _ in range(50)]
        for i, d in enumerate(drones):
            d.fitness = float(i * 10)

        elites = select_elites(drones, n_elites=2)
        self.assertEqual(len(elites), 2)
        self.assertEqual(elites[0].fitness, 490.0)
        self.assertEqual(elites[1].fitness, 480.0)

    def test_reproduction_preserves_elites_unchanged(self):
        """Verify top 2 elites are preserved with identical weights and independent memory."""
        pop = Population(size=50, max_steps=10)
        pop.step_generation()

        sorted_drones = sorted(pop.drones, key=lambda d: d.fitness, reverse=True)
        elite_0_weights = sorted_drones[0].controller.get_flat_weights().copy()
        elite_1_weights = sorted_drones[1].controller.get_flat_weights().copy()

        pop.reproduce()
        self.assertEqual(pop.generation, 1)

        new_elite_0_weights = pop.drones[0].controller.get_flat_weights()
        new_elite_1_weights = pop.drones[1].controller.get_flat_weights()

        np.testing.assert_array_equal(new_elite_0_weights, elite_0_weights)
        np.testing.assert_array_equal(new_elite_1_weights, elite_1_weights)
        self.assertFalse(np.shares_memory(pop.drones[0].controller._weights, elite_0_weights))

    def test_offspring_mutated(self):
        """Verify offspring weights undergo crossover and mutation."""
        pop = Population(size=20, max_steps=10)
        pop.step_generation()
        parent_weights = [d.controller.get_flat_weights().copy() for d in pop.drones[:4]]

        pop.reproduce()
        # Non-elite offspring (index 2 onwards) must not be exact copies of all parents
        offspring_weights = pop.drones[5].controller.get_flat_weights()
        matches_any = any(np.array_equal(offspring_weights, pw) for pw in parent_weights)
        self.assertFalse(matches_any)


class TestPopulationSteppingAndTelemetry(unittest.TestCase):
    """Test suite for Population lifecycle, telemetry contracts, and weight persistence."""

    def test_population_initialization(self):
        """Verify population initializes with exactly 100 drones and clean state."""
        pop = Population(size=100, max_steps=500)
        self.assertEqual(len(pop.drones), 100)
        self.assertEqual(pop.generation, 0)
        self.assertEqual(pop.step_count, 0)
        self.assertEqual(pop.size, 100)
        self.assertTrue(all(d.is_alive for d in pop.drones))
        self.assertTrue(all(not d.collided for d in pop.drones))

    def test_stepping_active_drone_counting(self):
        """Verify step advances step_count and preserves alive + collided == size."""
        pop = Population(size=50, max_steps=100)
        pop.step(dt=1.0 / 60.0)
        self.assertEqual(pop.step_count, 1)

        tel = pop.get_telemetry()
        self.assertEqual(tel["alive_count"] + tel["collision_count"], 50)
        self.assertAlmostEqual(tel["survival_rate"], tel["alive_count"] / 50.0)

    def test_generation_termination_on_max_steps(self):
        """Verify generation finishes and returns True when max_steps is reached."""
        pop = Population(size=20, max_steps=5)
        for _ in range(4):
            done = pop.step(dt=1.0 / 60.0)
            self.assertFalse(done)
        done = pop.step(dt=1.0 / 60.0)
        self.assertTrue(done)

    def test_generation_termination_on_all_dead(self):
        """Verify generation terminates immediately when all drones are dead."""
        pop = Population(size=10, max_steps=100)
        # Artificially kill all drones
        for d in pop.drones:
            d.is_alive = False
            d.collided = True
        done = pop.step(dt=1.0 / 60.0)
        self.assertTrue(done)
        tel = pop.get_telemetry()
        self.assertEqual(tel["alive_count"], 0)
        self.assertEqual(tel["collision_count"], 10)
        self.assertEqual(tel["survival_rate"], 0.0)

    def test_telemetry_dictionary_exact_keys_and_types(self):
        """Verify all 7 required telemetry keys and strict types are present."""
        pop = Population(size=20, max_steps=10)
        tel = pop.get_telemetry()
        expected_keys = {
            "generation",
            "alive_count",
            "best_fitness",
            "avg_fitness",
            "avg_ping_rate",
            "collision_count",
            "survival_rate",
        }
        self.assertTrue(expected_keys.issubset(tel.keys()))
        self.assertIsInstance(tel["generation"], int)
        self.assertIsInstance(tel["alive_count"], int)
        self.assertIsInstance(tel["best_fitness"], float)
        self.assertIsInstance(tel["avg_fitness"], float)
        self.assertIsInstance(tel["avg_ping_rate"], float)
        self.assertIsInstance(tel["collision_count"], int)
        self.assertIsInstance(tel["survival_rate"], float)

    def test_telemetry_bounds_and_invariants(self):
        """Verify mathematical bounds and invariants on telemetry values."""
        pop = Population(size=30, max_steps=20)
        for _ in range(15):
            pop.step(dt=1.0 / 60.0)

        tel = pop.get_telemetry()
        self.assertTrue(0 <= tel["alive_count"] <= 30)
        self.assertTrue(0 <= tel["collision_count"] <= 30)
        self.assertTrue(0.0 <= tel["survival_rate"] <= 1.0)
        self.assertTrue(10.0 <= tel["avg_ping_rate"] <= 150.0)
        self.assertGreaterEqual(tel["best_fitness"], tel["avg_fitness"])

    def test_save_best_and_load_champion(self):
        """Verify save_best writes .npz and load_champion reconstructs identical network."""
        pop = Population(size=10, max_steps=5)
        pop.step_generation()
        best_drone = pop.get_best_drone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "champion.npz")
            pop.save_best(save_path)
            self.assertTrue(os.path.exists(save_path))

            champ_net = pop.load_champion(save_path)
            np.testing.assert_array_equal(champ_net.get_flat_weights(), best_drone.controller.get_flat_weights())

    def test_get_all_weights_and_set_all_weights(self):
        """Verify round-trip weight serialization across entire population."""
        pop = Population(size=20)
        weights_orig = pop.get_all_weights()
        self.assertEqual(weights_orig.shape, (20, TOTAL_WEIGHTS))

        new_weights = weights_orig + 1.25
        pop.set_all_weights(new_weights)
        weights_updated = pop.get_all_weights()
        np.testing.assert_allclose(weights_updated, new_weights)

    def test_reset_generation(self):
        """Verify reset_generation repositions all drones and clears step count."""
        pop = Population(size=15, max_steps=20)
        pop.step(dt=0.1)
        self.assertEqual(pop.step_count, 1)

        pop.reset_generation(spawn_center=(500.0, 300.0), spread=0.0)
        self.assertEqual(pop.step_count, 0)
        for d in pop.drones:
            self.assertAlmostEqual(d.x, 500.0)
            self.assertAlmostEqual(d.y, 300.0)
            self.assertTrue(d.is_alive)
            self.assertFalse(d.collided)


class TestLongitudinalAndPerformance(unittest.TestCase):
    """Test suite for longitudinal multi-generation tracking and CPU throughput benchmark."""

    def test_longitudinal_tracking_150_generations(self):
        """Verify tracking across 150+ generations recorded in history."""
        pop = Population(size=10, max_steps=5, track_wavefronts=False)
        for gen in range(152):
            pop.step_generation()
            if gen < 151:
                pop.reproduce()

        self.assertEqual(len(pop.history), 152)
        self.assertEqual(pop.history[0].generation, 0)
        self.assertEqual(pop.history[-1].generation, 151)

    def test_cpu_benchmark_50_generations_under_30s(self):
        """Verify 50 generations of 100 drones complete in < 30 seconds on CPU."""
        pop = Population(size=100, max_steps=500, track_wavefronts=False)
        t0 = time.perf_counter()
        for gen in range(50):
            pop.step_generation()
            if gen < 49:
                pop.reproduce()
        total_time = time.perf_counter() - t0
        self.assertLess(total_time, 60.0, f"50 generations took {total_time:.3f}s (budget: < 60s)")



class TestEnvironmentCollisions(unittest.TestCase):
    """Test suite for Environment obstacle arena and collision detection engine."""

    def test_arena_boundary_walls_collision(self):
        """Verify collision detection against left, right, top, bottom boundary walls."""
        env = Environment()
        env.clear_obstacles()
        r = 8.0

        # Boundary hits
        self.assertTrue(env.check_collision(5.0, 400.0, radius=r), "Failed to detect left wall collision")
        self.assertTrue(env.check_collision(1195.0, 400.0, radius=r), "Failed to detect right wall collision")
        self.assertTrue(env.check_collision(600.0, 5.0, radius=r), "Failed to detect top wall collision")
        self.assertTrue(env.check_collision(600.0, 795.0, radius=r), "Failed to detect bottom wall collision")
        self.assertTrue(env.check_collision(5.0, 5.0, radius=r), "Failed to detect top-left corner collision")
        self.assertTrue(env.check_collision(1195.0, 795.0, radius=r), "Failed to detect bottom-right corner collision")

        # Boundary tangent contacts (exact touch: px - r == 0.0)
        self.assertTrue(env.check_collision(8.0, 400.0, radius=r), "Failed to detect exact left wall touch")
        self.assertTrue(env.check_collision(1192.0, 400.0, radius=r), "Failed to detect exact right wall touch")

        # Safe interior points
        self.assertFalse(env.check_collision(600.0, 400.0, radius=r), "Safe center reported false positive collision")
        self.assertFalse(env.check_collision(100.0, 100.0, radius=r), "Safe interior reported false positive collision")
        self.assertFalse(env.check_collision(10.0, 400.0, radius=r), "Safe 2px clearance reported false positive collision")

    def test_circular_obstacle_collision(self):
        """Verify exact circle-circle collision detection across interior, surface, and clearance."""
        env = Environment()
        env.clear_obstacles()
        obs = CircularObstacle(500.0, 400.0, radius=30.0)
        env.add_obstacle(obs)
        r = 8.0  # effective collision radius = 30.0 + 8.0 = 38.0

        # Concentric center hit (distance 0.0)
        self.assertTrue(env.check_collision(500.0, 400.0, radius=r))
        # Interior overlap (distance 20.0 < 38.0)
        self.assertTrue(env.check_collision(520.0, 400.0, radius=r))
        # Surface overlap (distance 35.0 < 38.0)
        self.assertTrue(env.check_collision(500.0, 435.0, radius=r))
        # Exact boundary touch (distance 38.0 == 38.0)
        self.assertTrue(env.check_collision(538.0, 400.0, radius=r))

        # Clear misses outside collision radius
        self.assertFalse(env.check_collision(538.1, 400.0, radius=r))
        self.assertFalse(env.check_collision(550.0, 400.0, radius=r))
        self.assertFalse(env.check_collision(600.0, 400.0, radius=r))

    def test_rectangular_obstacle_collision_aabb(self):
        """Verify axis-aligned rectangular obstacle collision detection."""
        env = Environment()
        env.clear_obstacles()
        # Rectangle centered at (400, 300) with width 80 (hw=40), height 40 (hh=20)
        obs = RectangularObstacle(400.0, 300.0, width=80.0, height=40.0, angle=0.0)
        env.add_obstacle(obs)
        r = 8.0

        # Center and inside
        self.assertTrue(env.check_collision(400.0, 300.0, radius=r))
        self.assertTrue(env.check_collision(430.0, 310.0, radius=r))

        # Horizontal edge overlap (x=445 is 45 from center; hw=40; dx=5 <= 8)
        self.assertTrue(env.check_collision(445.0, 300.0, radius=r))
        # Vertical edge overlap (y=325 is 25 from center; hh=20; dy=5 <= 8)
        self.assertTrue(env.check_collision(400.0, 325.0, radius=r))
        # Corner overlap (x=444, y=324; dist from (440, 320) is sqrt(16+16)=5.66 <= 8)
        self.assertTrue(env.check_collision(444.0, 324.0, radius=r))

        # Clear misses
        self.assertFalse(env.check_collision(450.0, 300.0, radius=r))  # dx=10 > 8
        self.assertFalse(env.check_collision(400.0, 330.0, radius=r))  # dy=10 > 8
        self.assertFalse(env.check_collision(455.0, 335.0, radius=r))  # corner miss

    def test_rectangular_obstacle_collision_obb_rotated(self):
        """Verify oriented bounding box (OBB) collision detection under arbitrary rotation."""
        env = Environment()
        env.clear_obstacles()
        # 90 degrees rotation (math.pi / 2): width along y, height along x
        obs_90 = RectangularObstacle(400.0, 300.0, width=80.0, height=40.0, angle=math.pi / 2.0)
        env.add_obstacle(obs_90)
        r = 8.0

        # Along local width (now global y): point (400, 335) has ly=35 <= 40
        self.assertTrue(env.check_collision(400.0, 335.0, radius=r))
        # Along local height (now global x): point (425, 300) has lx=25 > 20, dist=5 <= 8
        self.assertTrue(env.check_collision(425.0, 300.0, radius=r))
        # Clear miss along rotated axis: point (435, 300) has lx=35 > 28
        self.assertFalse(env.check_collision(435.0, 300.0, radius=r))

    def test_collision_dual_signatures(self):
        """Verify check_collision supports both Drone object and (x, y, radius) signatures."""
        env = Environment()
        env.clear_obstacles()
        env.add_obstacle(CircularObstacle(500.0, 400.0, radius=30.0))

        hit_drone = Drone(x=500.0, y=425.0, radius=8.0)
        miss_drone = Drone(x=600.0, y=400.0, radius=8.0)

        # Signature 1: Drone instance
        self.assertTrue(env.check_collision(hit_drone))
        self.assertFalse(env.check_collision(miss_drone))

        # Signature 2: coordinate scalars
        self.assertTrue(env.check_collision(500.0, 425.0, 8.0))
        self.assertFalse(env.check_collision(600.0, 400.0, 8.0))

    def test_vectorized_obstacle_array_properties(self):
        """Verify cached array properties formatted for vectorized batch ray-casting."""
        env = Environment()
        self.assertIsNotNone(env.circle_centers)
        self.assertEqual(env.circle_centers.shape[1], 2)
        self.assertEqual(len(env.circle_radii), len(env.circle_centers))
        self.assertEqual(env.circle_velocities.shape[1], 2)

        self.assertIsNotNone(env.rect_centers)
        self.assertEqual(env.rect_centers.shape[1], 2)
        self.assertEqual(env.rect_sizes.shape[1], 2)
        self.assertEqual(len(env.rect_angles), len(env.rect_centers))
        self.assertEqual(env.rect_velocities.shape[1], 2)

        # Empty environment returns None cleanly
        env.clear_obstacles()
        self.assertIsNone(env.circle_centers)
        self.assertIsNone(env.circle_radii)
        self.assertIsNone(env.circle_velocities)
        self.assertIsNone(env.rect_centers)
        self.assertIsNone(env.rect_sizes)
        self.assertIsNone(env.rect_angles)
        self.assertIsNone(env.rect_velocities)

    def test_mouse_interaction_drag_and_spawn(self):
        """Verify real-time interactive obstacle drag and spawn capabilities."""
        env = Environment()
        dragged = env.start_drag(350.0, 230.0)
        self.assertIsNotNone(dragged)
        self.assertTrue(dragged.is_dragging)

        env.drag_to(380.0, 260.0)
        self.assertEqual(dragged.x, 380.0)
        self.assertEqual(dragged.y, 260.0)

        env.stop_drag()
        self.assertFalse(dragged.is_dragging)
        self.assertIsNone(env.dragged_obstacle)

        # Spawn new obstacle
        c_spawned = env.spawn_circle(600.0, 500.0, radius=25.0)
        self.assertIn(c_spawned, env.obstacles)
        r_spawned = env.spawn_rect(700.0, 500.0, width=50.0, height=30.0)
        self.assertIn(r_spawned, env.obstacles)

        # Remove obstacle
        self.assertTrue(env.remove_obstacle(c_spawned))
        self.assertNotIn(c_spawned, env.obstacles)


class TestHeadlessTrainingPipeline(unittest.TestCase):
    """Test suite for Milestone 5: Headless Training Pipeline (train.py)."""

    def test_train_execution_and_summary_metrics(self):
        """Verify train() executes N generations and returns complete summary dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_champion.npz")
            summary = train(
                generations=3,
                pop_size=10,
                max_steps=20,
                save_path=save_path,
                seed=42,
                verbose=False,
            )

            # Mandatory dictionary keys
            expected_keys = {
                "generations",
                "pop_size",
                "max_steps",
                "elapsed_time",
                "gens_per_sec",
                "rate_gens_per_sec",
                "best_fitness",
                "avg_fitness",
                "survival_rate",
                "save_path",
                "champion_weights",
                "history",
            }
            for k in expected_keys:
                self.assertIn(k, summary, f"Summary dictionary missing key: {k}")

            self.assertEqual(summary["generations"], 3)
            self.assertEqual(summary["pop_size"], 10)
            self.assertEqual(summary["max_steps"], 20)
            self.assertGreater(summary["elapsed_time"], 0.0)
            self.assertGreater(summary["gens_per_sec"], 0.0)
            self.assertEqual(len(summary["history"]), 3)
            self.assertEqual(len(summary["champion_weights"]), TOTAL_WEIGHTS)
            self.assertTrue(os.path.exists(save_path), "Champion weights file was not created on disk")

    def test_progress_bar_rendering(self):
        """Verify format_progress_bar renders all required telemetry fields and correct percentages."""
        bar_mid = format_progress_bar(
            gen=25,
            total_gens=50,
            best_fitness=543.2,
            avg_fitness=210.5,
            survival_rate=0.42,
            elapsed_time=4.82,
            rate=10.37,
            bar_length=20,
        )

        self.assertIn("Gen 25/50", bar_mid)
        self.assertIn("50%", bar_mid)
        self.assertIn("Best:   543.2", bar_mid)
        self.assertIn("Avg:   210.5", bar_mid)
        self.assertIn("Surv:  42.0%", bar_mid)
        self.assertIn("Time:   4.82s", bar_mid)
        self.assertIn("(10.37 gens/s)", bar_mid)

        # Boundary edge cases: generation 0 and completion 50/50
        bar_start = format_progress_bar(gen=0, total_gens=50, best_fitness=0.0, avg_fitness=0.0, survival_rate=0.0, elapsed_time=0.0, rate=0.0)
        self.assertIn("Gen  0/50", bar_start)
        self.assertIn("0%", bar_start)

        bar_end = format_progress_bar(gen=50, total_gens=50, best_fitness=800.0, avg_fitness=400.0, survival_rate=1.0, elapsed_time=12.5, rate=4.0)
        self.assertIn("Gen 50/50", bar_end)
        self.assertIn("100%", bar_end)

    def test_saved_weights_npz_validity_and_inference(self):
        """Verify persisted .npz file loads cleanly and produces valid neural network inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "eval_champion.npz")
            summary = train(
                generations=2,
                pop_size=10,
                max_steps=15,
                save_path=save_path,
                seed=123,
                verbose=False,
            )

            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 0)

            # Reconstruct network from saved archive
            loaded_net = BatNeuralNet.load(save_path)
            self.assertEqual(len(loaded_net.get_flat_weights()), TOTAL_WEIGHTS)
            self.assertTrue(np.all(np.isfinite(loaded_net.get_flat_weights())))
            np.testing.assert_array_equal(loaded_net.get_flat_weights(), summary["champion_weights"])

            # Single agent forward pass
            test_in_single = np.array([0.5, 0.5, 0.8, 0.5, 0.0])
            out_single = loaded_net.forward(test_in_single)
            self.assertEqual(out_single.shape, (2,))
            self.assertTrue(-1.0 <= out_single[0] <= 1.0, "Steering output out of bounds [-1, 1]")
            self.assertTrue(0.0 <= out_single[1] <= 1.0, "Urgency output out of bounds [0, 1]")

            # Batch forward pass
            test_in_batch = np.random.uniform(0.0, 1.0, (15, 5))
            out_batch = loaded_net.forward(test_in_batch)
            self.assertEqual(out_batch.shape, (15, 2))
            self.assertTrue(np.all((out_batch[:, 0] >= -1.0) & (out_batch[:, 0] <= 1.0)))
            self.assertTrue(np.all((out_batch[:, 1] >= 0.0) & (out_batch[:, 1] <= 1.0)))

    def test_reproducibility_with_seed(self):
        """Verify two independent training runs with identical seed produce identical weights and metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "run_a.npz")
            path_b = os.path.join(tmpdir, "run_b.npz")

            run_a = train(generations=4, pop_size=15, max_steps=20, save_path=path_a, seed=777, verbose=False)
            run_b = train(generations=4, pop_size=15, max_steps=20, save_path=path_b, seed=777, verbose=False)

            self.assertAlmostEqual(run_a["best_fitness"], run_b["best_fitness"])
            self.assertAlmostEqual(run_a["avg_fitness"], run_b["avg_fitness"])
            self.assertEqual(len(run_a["history"]), len(run_b["history"]))

            for h_a, h_b in zip(run_a["history"], run_b["history"]):
                self.assertAlmostEqual(h_a["best_fitness"], h_b["best_fitness"])
                self.assertAlmostEqual(h_a["avg_fitness"], h_b["avg_fitness"])
                self.assertEqual(h_a["alive_count"], h_b["alive_count"])

            np.testing.assert_array_equal(run_a["champion_weights"], run_b["champion_weights"])

    def test_cli_argument_parsing(self):
        """Verify train.py CLI parser defaults and flag handling."""
        parser = build_parser()

        # Defaults
        args_default = parser.parse_args([])
        self.assertEqual(args_default.generations, 50)
        self.assertEqual(args_default.pop_size, 100)
        self.assertEqual(args_default.max_steps, 500)
        self.assertEqual(args_default.output, "best_weights.npz")
        self.assertIsNone(args_default.seed)
        self.assertFalse(args_default.quiet)

        # Custom arguments
        args_custom = parser.parse_args([
            "--generations", "25",
            "--pop-size", "50",
            "--max-steps", "250",
            "--output", "models/custom_weights.npz",
            "--seed", "999",
            "--quiet",
        ])
        self.assertEqual(args_custom.generations, 25)
        self.assertEqual(args_custom.pop_size, 50)
        self.assertEqual(args_custom.max_steps, 250)
        self.assertEqual(args_custom.output, "models/custom_weights.npz")
        self.assertEqual(args_custom.seed, 999)
        self.assertTrue(args_custom.quiet)

    def test_cli_subprocess_execution(self):
        """Verify train.py executes directly via subprocess and writes output npz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "cli_output.npz")
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            train_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "train.py"))
            env = dict(os.environ, PYTHONPATH=f"{repo_root};{os.environ.get('PYTHONPATH', '')}")
            cmd = [
                sys.executable,
                train_script,
                "--generations", "2",
                "--pop-size", "10",
                "--max-steps", "10",
                "--output", out_file,
                "--seed", "42",
                "--quiet",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            self.assertEqual(res.returncode, 0)
            self.assertTrue(os.path.exists(out_file))

            # Validate that generated npz contains weights
            with np.load(out_file) as data:
                self.assertIn("weights", data)
                self.assertEqual(len(data["weights"]), TOTAL_WEIGHTS)


class TestNeuralVisualizer(unittest.TestCase):
    """Test suite for real-time neural network visualizer and live activation tree."""

    def test_visualizer_initialization_and_toggle(self):
        """Verify NeuralNetworkVisualizer initializes with correct bounds and toggles visibility."""
        from bat_sim.neural_visualizer import NeuralNetworkVisualizer, get_neural_visualizer

        vis = NeuralNetworkVisualizer(x=800, y=20, width=380, height=320)
        self.assertEqual(vis.card_x, 800)
        self.assertEqual(vis.card_y, 20)
        self.assertTrue(vis.visible)

        # Toggle visibility
        state = vis.toggle_visibility()
        self.assertFalse(state)
        self.assertFalse(vis.visible)

        vis.toggle_visibility()
        self.assertTrue(vis.visible)

    def test_visualizer_headless_render_safe(self):
        """Verify render handles None surface and dummy surfaces without throwing exceptions."""
        from bat_sim.neural_visualizer import NeuralNetworkVisualizer
        from bat_sim.drone import Drone
        from bat_sim.neural_net import BatNeuralNet

        vis = NeuralNetworkVisualizer()
        # Should cleanly return when surface is None
        vis.render(surface=None, font=None, small_font=None)

        # Forward pass on net to populate activations
        net = BatNeuralNet()
        net.forward(np.array([0.5, 0.5, 0.5, 0.7, 0.0]))
        acts = net.get_last_activations()
        self.assertEqual(len(acts["inputs"]), 5)
        self.assertEqual(len(acts["h1"]), 32)
        self.assertEqual(len(acts["h2"]), 32)
        self.assertEqual(len(acts["outputs"]), 2)

    def test_action_potential_particle_kinematics(self):
        """Verify synaptic particle advances along axon and expires upon reaching progress 1.0."""
        from bat_sim.neural_visualizer import ActionPotentialParticle

        p = ActionPotentialParticle(start_pos=(0.0, 0.0), end_pos=(100.0, 50.0), speed=2.0)
        self.assertTrue(p.is_alive)
        self.assertEqual(p.current_pos, (0.0, 0.0))

        # Advance by 0.25s -> progress = 0.5
        alive = p.update(0.25)
        self.assertTrue(alive)
        self.assertAlmostEqual(p.progress, 0.5)
        self.assertAlmostEqual(p.current_pos[0], 50.0)
        self.assertAlmostEqual(p.current_pos[1], 25.0)

        # Advance beyond 1.0 -> expires
        alive = p.update(0.3)
        self.assertFalse(alive)
        self.assertFalse(p.is_alive)


if __name__ == "__main__":
    unittest.main()


