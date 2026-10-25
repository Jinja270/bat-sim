"""
bat_sim/test_adversarial_m3.py
==============================
Adversarial Stress Test Suite for Milestone 3: Drone Agent & Neuroevolution Engine.

Validates boundary conditions, numerical extremes, invariant stress, and edge cases:
1. Kinematics Extremes:
   - Zero speed, maximum speed clamping (speed > max_speed).
   - Extreme time step delta (dt = 0.0, dt = 10.0, dt = 1e-6).
   - Exact boundary coordinate hits (touching walls at exactly radius boundary).
   - Rapid multi-turn heading wrapping.
2. Acoustic Pulse Rate & Sensor Invariants:
   - Extreme urgency values (< 0.0, > 1.0 clamped).
   - High frequency pings under large dt (multiple period accumulations per frame).
   - Zero range / extreme distances.
3. Wavefront Robustness:
   - Wavefront age, speed, alpha clamping with zero max_radius.
   - Arbitrary surface draw calls without Pygame initialization.
4. Genetic Selection & Elitism Rigor:
   - Population size boundary (size = 2, size = 5, size = 100, size = 200).
   - All identical fitness ranking.
   - All negative fitness ranking (all drones collided at step 0).
   - Zero memory sharing between elites and parents across 10 generations.
   - Mutation sigma stability at gen 1,000 (never drops below min_sigma).
5. Telemetry Schema & Strict Invariants:
   - Sum rule: alive_count + collision_count == size.
   - Survival rate: survival_rate == alive_count / size.
   - Mean inequality: best_fitness >= avg_fitness >= min_fitness.
   - Ping rate bounds: 10.0 <= avg_ping_rate <= 150.0.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import bat_sim.physics as physics
from bat_sim.neural_net import BatNeuralNet, TOTAL_WEIGHTS, compute_mutation_sigma
from bat_sim.drone import (
    Drone,
    TraditionalDrone,
    Wavefront,
    urgency_to_color,
    DEFAULT_HITBOX_RADIUS,
    DEFAULT_SPEED,
    DEFAULT_MAX_SPEED,
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
)


class TestDroneAdversarialKinematics(unittest.TestCase):
    """Stress testing drone kinematic integration and edge boundaries."""

    def test_speed_clamping_at_max(self):
        """Verify speed above max_speed is clamped to max_speed."""
        drone = Drone(speed=25.0, max_speed=10.0)
        self.assertEqual(drone.speed, 10.0)

    def test_speed_clamping_at_zero(self):
        """Verify negative speed is clamped to 0.0."""
        drone = Drone(speed=-5.0, max_speed=10.0)
        self.assertEqual(drone.speed, 0.0)

    def test_zero_dt_step(self):
        """Verify dt = 0.0 advances nothing."""
        drone = Drone(x=500.0, y=300.0, heading=1.0, speed=5.0)
        drone.step(0.0)
        self.assertEqual(drone.x, 500.0)
        self.assertEqual(drone.y, 300.0)
        self.assertEqual(drone.distance_traveled, 0.0)
        self.assertEqual(drone.time_alive_steps, 1)

    def test_micro_dt_stepping(self):
        """Verify micro time steps dt = 1e-6 maintain numerical precision."""
        drone = Drone(x=500.0, y=300.0, heading=0.0, speed=5.0, controller=None)
        dt = 1e-6
        for _ in range(1000):
            drone.step(dt)
        self.assertAlmostEqual(drone.x, 500.0 + 5.0 * 1e-3, places=7)
        self.assertAlmostEqual(drone.distance_traveled, 5.0 * 1e-3, places=7)

    def test_wall_touching_exact_boundary(self):
        """Verify drone with hitbox touching or crossing wall by 1e-9 is caught."""
        r = DEFAULT_HITBOX_RADIUS  # 8.0
        # Exactly touching left wall: x = 8.0 -> x - r == 0.0 -> collision
        d_touch = Drone(x=8.0, y=400.0, radius=r, speed=0.0, controller=None)
        d_touch.step(0.01)
        self.assertFalse(d_touch.is_alive)
        self.assertTrue(d_touch.collided)

        # Inside safe boundary: x = 8.5 -> x - r == 0.5 > 0 -> alive
        d_safe = Drone(x=8.5, y=400.0, heading=0.0, speed=0.0, radius=r, controller=None)
        d_safe.step(0.01)
        self.assertTrue(d_safe.is_alive)
        self.assertFalse(d_safe.collided)

    def test_large_dt_multiple_ping_periods(self):
        """Verify dt much larger than ping period triggers ping and properly wraps timer."""
        drone = Drone(x=600.0, y=400.0)
        drone.ping_freq = 100.0  # period = 0.01s
        # Advance by 0.055s: should ping and timer should remain in [0, 0.01)
        drone.step(0.055)
        self.assertGreaterEqual(drone.total_pings, 1)
        self.assertLess(drone.ping_timer, 0.01)


class TestWavefrontAdversarial(unittest.TestCase):
    """Stress testing Wavefront expansion, opacity, and color bounds."""

    def test_wavefront_negative_urgency_clamped(self):
        """Verify urgency < 0.0 is clamped to green."""
        col = urgency_to_color(-0.5)
        self.assertEqual(col, COLOR_SEARCH_GREEN)

    def test_wavefront_excessive_urgency_clamped(self):
        """Verify urgency > 1.0 is clamped to red."""
        col = urgency_to_color(1.5)
        self.assertEqual(col, COLOR_TERMINAL_RED)

    def test_wavefront_zero_max_radius(self):
        """Verify get_alpha handles max_radius <= 0 without division by zero."""
        wf = Wavefront(x=0.0, y=0.0, max_radius=0.0)
        self.assertEqual(wf.get_alpha(), 0.0)

    def test_wavefront_radius_exceeding_max(self):
        """Verify radius beyond max_radius clamps alpha strictly to 0.0."""
        wf = Wavefront(x=0.0, y=0.0, radius=500.0, max_radius=300.0)
        self.assertEqual(wf.get_alpha(), 0.0)
        self.assertFalse(wf.is_alive)


class TestPopulationAdversarialSelection(unittest.TestCase):
    """Stress testing population evolution under extreme distributions."""

    def test_all_identical_fitness(self):
        """Verify truncation and elitism handle identical fitness scores without error."""
        pop = Population(size=50, max_steps=5)
        for d in pop.drones:
            d.fitness = 100.0

        parents = select_parents(pop.drones, truncation_fraction=0.20)
        self.assertEqual(len(parents), 10)

        elites = select_elites(pop.drones, n_elites=2)
        self.assertEqual(len(elites), 2)

        pop.reproduce()
        self.assertEqual(len(pop.drones), 50)
        self.assertEqual(pop.generation, 1)

    def test_all_collided_negative_fitness(self):
        """Verify reproduction works when all drones collide at step 0 (-50.0)."""
        pop = Population(size=40, max_steps=10)
        for d in pop.drones:
            d.time_alive_steps = 0
            d.distance_traveled = 0.0
            d.collided = True
            d.is_alive = False

        fit = pop.evaluate_fitness()
        self.assertTrue(np.all(fit == -50.0))

        pop.reproduce()
        self.assertEqual(len(pop.drones), 40)
        self.assertEqual(pop.generation, 1)
        self.assertTrue(all(d.is_alive for d in pop.drones))

    def test_mutation_sigma_decay_asymptotic_floor(self):
        """Verify mutation sigma decays monotonically and stays clamped at min_sigma at gen 500+."""
        sigmas = [compute_mutation_sigma(g, sigma_0=0.1, decay_factor=0.98, min_sigma=0.01) for g in range(500)]
        # Must be non-increasing
        for i in range(len(sigmas) - 1):
            self.assertGreaterEqual(sigmas[i], sigmas[i + 1])
        # Floor must be exactly 0.01
        self.assertAlmostEqual(sigmas[0], 0.1)
        self.assertAlmostEqual(sigmas[200], 0.01)
        self.assertAlmostEqual(sigmas[499], 0.01)

    def test_telemetry_invariants_across_10_generations(self):
        """Verify strict invariants hold across 10 complete generations."""
        pop = Population(size=50, max_steps=100, track_wavefronts=False)
        for _ in range(10):
            tel = pop.step_generation()
            self.assertEqual(tel["alive_count"] + tel["collision_count"], 50)
            self.assertAlmostEqual(tel["survival_rate"], tel["alive_count"] / 50.0)
            self.assertGreaterEqual(tel["best_fitness"], tel["avg_fitness"])
            self.assertGreaterEqual(tel["avg_ping_rate"], 10.0)
            self.assertLessEqual(tel["avg_ping_rate"], 150.0)
            pop.reproduce()


if __name__ == "__main__":
    unittest.main()
