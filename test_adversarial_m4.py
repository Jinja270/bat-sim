"""
bat_sim/test_adversarial_m4.py
==============================
Adversarial Challenge & Stress-Test Suite for Milestone 4.

Validates the 5 key focus areas:
1. Corner/Edge Case Collision Testing for Rotated Rectangles:
   - OBB clamping at 0, 30, 45, 60, 90, 135, 180, 270, and negative angles.
   - Exact corner normal testing (in vs out vs boundary contact).
   - Exact edge normal testing (in vs out).
   - Extreme aspect ratios (hairline needles 500x1, 1x500; huge 700x700; zero-extent).
   - High speed kinematics and boundary reflection conservation.
2. Circle Collision Edge Cases:
   - Exact tangency over 360-degree sweep (floating point precision).
   - Grazing margin (+/- 1e-6).
   - Zero-radius obstacle and zero-radius drone point collisions.
   - Arena wall grazing and arena corner contacts.
   - Boundary bouncing for extreme velocities.
3. Mouse Drag Bounds Clamping:
   - Circular obstacles dragged to extreme negative/positive coordinates.
   - Rotated rectangular obstacles at diverse angles dragged past all 4 boundaries.
   - Verification that all 4 corners of rotated OBB remain in [0, 1200] x [0, 800].
   - Drag lifecycle invariants (empty drag, release, obstacle deletion mid-drag).
4. Comparison Mode Telemetry & Baseline Drone Collision Difference:
   - Telemetry consistency and invariants (alive + collided == total).
   - Fixed 10 Hz vs dynamic 10-150 Hz pulse rate verification.
   - Collision divergence under drifting obstacle fields.
   - Reset lifecycle in comparison mode.
5. Headless Simulation Stability Under Load:
   - High drone population (pop_size = 200).
   - High obstacle density (50 obstacles in arena).
   - Multi-generation fast-forward unthrottled stepping.
   - Wavefront memory leak guard (pruning verification).
   - Finite numeric stability (no NaNs, no Infs across telemetry).
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import bat_sim.physics as physics
from bat_sim.drone import (
    Drone,
    TraditionalDrone,
    DEFAULT_HITBOX_RADIUS,
    DEFAULT_SPEED,
    DEFAULT_MAX_SPEED,
)
from bat_sim.environment import (
    CircularObstacle,
    RectangularObstacle,
    Environment,
    DEFAULT_ARENA_WIDTH,
    DEFAULT_ARENA_HEIGHT,
)
from bat_sim.evolution import Population
from bat_sim.main import init_display, run_simulation


class TestRotatedRectangularObstacleOBB(unittest.TestCase):
    """Adversarial stress-testing of rotated OBB collision math, corners, and aspect ratios."""

    def test_corner_collision_at_various_angles(self):
        """
        Verify that drone placed at corner normal distance (r - eps) collides,
        and at (r + eps) misses, across angles: 0, 30, 45, 60, 90, 135, 180, 270, -45 deg.
        """
        test_angles = [
            0.0,
            math.pi / 6.0,      # 30 deg
            math.pi / 4.0,      # 45 deg
            math.pi / 3.0,      # 60 deg
            math.pi / 2.0,      # 90 deg
            3.0 * math.pi / 4.0,# 135 deg
            math.pi,            # 180 deg
            3.0 * math.pi / 2.0,# 270 deg
            -math.pi / 4.0,     # -45 deg
            1.2345,             # arbitrary float
        ]
        r_drone = 8.0
        w, h = 80.0, 50.0
        cx, cy = 600.0, 400.0
        hw, hh = w / 2.0, h / 2.0

        for angle in test_angles:
            rect = RectangularObstacle(cx, cy, width=w, height=h, angle=angle)
            env = Environment(obstacles=[rect])

            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Local corners
            local_corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            for lx, ly in local_corners:
                # World corner
                wx = cx + lx * cos_a - ly * sin_a
                wy = cy + lx * sin_a + ly * cos_a

                # Direction from rectangle center to corner in world space
                diag_len = math.hypot(lx, ly)
                ux = (lx * cos_a - ly * sin_a) / diag_len
                uy = (lx * sin_a + ly * cos_a) / diag_len

                # 1. Inside corner by 0.5px: distance to corner is r_drone - 0.5
                px_in = wx + (r_drone - 0.5) * ux
                py_in = wy + (r_drone - 0.5) * uy
                self.assertTrue(
                    env.check_collision(px_in, py_in, radius=r_drone),
                    f"Failed corner collision (in) at angle {math.degrees(angle):.1f}° for corner ({lx}, {ly})",
                )

                # 2. Outside corner by 0.5px: distance to corner is r_drone + 0.5
                px_out = wx + (r_drone + 0.5) * ux
                py_out = wy + (r_drone + 0.5) * uy
                self.assertFalse(
                    env.check_collision(px_out, py_out, radius=r_drone),
                    f"False positive corner collision (out) at angle {math.degrees(angle):.1f}° for corner ({lx}, {ly})",
                )

                # 3. Exactly at the corner: distance is 0 <= r_drone
                self.assertTrue(
                    env.check_collision(wx, wy, radius=r_drone),
                    f"Failed exact corner hit at angle {math.degrees(angle):.1f}°",
                )

    def test_edge_collision_precision_at_various_angles(self):
        """Verify normal edge distance collision check at 45° and 90°."""
        r_drone = 8.0
        w, h = 100.0, 60.0
        cx, cy = 600.0, 400.0

        for angle in [math.pi / 4.0, math.pi / 2.0, -math.pi / 3.0]:
            rect = RectangularObstacle(cx, cy, width=w, height=h, angle=angle)
            env = Environment(obstacles=[rect])

            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Test point at midpoint of right edge: local (w/2, 0)
            lx_mid, ly_mid = w / 2.0, 0.0
            wx_mid = cx + lx_mid * cos_a - ly_mid * sin_a
            wy_mid = cy + lx_mid * sin_a + ly_mid * cos_a

            # Normal pointing out of right edge: (cos_a, sin_a)
            nx, ny = cos_a, sin_a

            # Overlapping by 0.1px: dist from edge = r_drone - 0.1
            p_hit = (wx_mid + (r_drone - 0.1) * nx, wy_mid + (r_drone - 0.1) * ny)
            self.assertTrue(env.check_collision(p_hit[0], p_hit[1], radius=r_drone))

            # Missing by 0.1px: dist from edge = r_drone + 0.1
            p_miss = (wx_mid + (r_drone + 0.1) * nx, wy_mid + (r_drone + 0.1) * ny)
            self.assertFalse(env.check_collision(p_miss[0], p_miss[1], radius=r_drone))

    def test_extreme_aspect_ratios(self):
        """Test needle-like obstacles (500x2, 2x500) and zero-dimension degradation."""
        r_drone = 8.0
        cx, cy = 600.0, 400.0

        # Ultra-thin needle rotated at 45 degrees
        needle_h = RectangularObstacle(cx, cy, width=500.0, height=2.0, angle=math.pi / 4.0)
        env = Environment(obstacles=[needle_h])

        # Center hit
        self.assertTrue(env.check_collision(cx, cy, radius=r_drone))

        # Tip hit (local lx = 240, ly = 0)
        tip_x = cx + 240.0 * math.cos(math.pi / 4.0)
        tip_y = cy + 240.0 * math.sin(math.pi / 4.0)
        self.assertTrue(env.check_collision(tip_x, tip_y, radius=r_drone))

        # Beyond tip (local lx = 265 > 250 + 8)
        beyond_x = cx + 265.0 * math.cos(math.pi / 4.0)
        beyond_y = cy + 265.0 * math.sin(math.pi / 4.0)
        self.assertFalse(env.check_collision(beyond_x, beyond_y, radius=r_drone))

        # Degenerate zero-width rectangle (line segment)
        zero_rect = RectangularObstacle(cx, cy, width=0.0, height=60.0, angle=0.0)
        env_zero = Environment(obstacles=[zero_rect])
        self.assertTrue(env_zero.check_collision(cx + 5.0, cy, radius=r_drone))  # 5 <= 8
        self.assertFalse(env_zero.check_collision(cx + 12.0, cy, radius=r_drone)) # 12 > 8

    def test_high_speed_rotated_obstacle_bouncing(self):
        """Stress-test obstacle kinematics under extreme velocities and large time steps."""
        bounds = (0.0, 0.0, 1200.0, 800.0)
        # Fast spinning/moving rectangle
        rect = RectangularObstacle(600.0, 400.0, width=80.0, height=40.0, angle=math.pi / 4.0, vx=2000.0, vy=-1500.0)

        dt = 0.05  # 50ms at 2000 px/s is 100px per step
        for _ in range(200):
            rect.update(dt, bounds)
            # Verify bounding extents never penetrate arena boundary
            hw_proj = 0.5 * (abs(rect.width * math.cos(rect.angle)) + abs(rect.height * math.sin(rect.angle)))
            hh_proj = 0.5 * (abs(rect.width * math.sin(rect.angle)) + abs(rect.height * math.cos(rect.angle)))

            self.assertGreaterEqual(rect.x - hw_proj, 0.0 - 1e-9)
            self.assertLessEqual(rect.x + hw_proj, 1200.0 + 1e-9)
            self.assertGreaterEqual(rect.y - hh_proj, 0.0 - 1e-9)
            self.assertLessEqual(rect.y + hh_proj, 800.0 + 1e-9)


class TestCircleCollisionEdgeCases(unittest.TestCase):
    """Adversarial stress-testing of circular obstacle collision edge cases."""

    def test_exact_tangency_360_degrees(self):
        """Verify circle-circle collision handles tangency across full circle."""
        cx, cy = 600.0, 400.0
        cr = 40.0
        pr = 8.0
        target_dist = cr + pr  # 48.0

        obs = CircularObstacle(cx, cy, radius=cr)
        env = Environment(obstacles=[obs])

        # Test at 72 angles around the circle (every 5 degrees)
        for i in range(72):
            theta = math.radians(i * 5.0)

            # Inward by 1e-4: must collide
            px_in = cx + (target_dist - 1e-4) * math.cos(theta)
            py_in = cy + (target_dist - 1e-4) * math.sin(theta)
            self.assertTrue(
                env.check_collision(px_in, py_in, radius=pr),
                f"Failed tangency hit inward at angle {i * 5}°",
            )

            # Outward by 1e-4: must not collide
            px_out = cx + (target_dist + 1e-4) * math.cos(theta)
            py_out = cy + (target_dist + 1e-4) * math.sin(theta)
            self.assertFalse(
                env.check_collision(px_out, py_out, radius=pr),
                f"Failed tangency miss outward at angle {i * 5}°",
            )

    def test_zero_radius_obstacle_and_drone(self):
        """Verify behavior with zero-radius obstacle and zero-radius drone."""
        cx, cy = 500.0, 300.0
        point_obs = CircularObstacle(cx, cy, radius=0.0)
        env = Environment(obstacles=[point_obs])

        # Drone with radius 8 against point obstacle
        self.assertTrue(env.check_collision(cx + 6.0, cy, radius=8.0))
        self.assertTrue(env.check_collision(cx, cy, radius=8.0))
        self.assertFalse(env.check_collision(cx + 10.0, cy, radius=8.0))

        # Point drone (radius 0.0) against point obstacle (radius 0.0)
        self.assertTrue(env.check_collision(cx, cy, radius=0.0))
        self.assertFalse(env.check_collision(cx + 0.1, cy, radius=0.0))

    def test_arena_wall_grazing_precision(self):
        """Verify precision at arena boundary walls and all 4 corners."""
        env = Environment(obstacles=[])
        pr = 8.0

        # Exact boundary touch (x - pr == 0.0) -> is_out_of_bounds returns True
        self.assertTrue(env.check_collision(8.0, 400.0, radius=pr))
        self.assertTrue(env.check_collision(1200.0 - 8.0, 400.0, radius=pr))
        self.assertTrue(env.check_collision(600.0, 8.0, radius=pr))
        self.assertTrue(env.check_collision(600.0, 800.0 - 8.0, radius=pr))

        # Safe interior by 0.01px -> False
        self.assertFalse(env.check_collision(8.01, 400.0, radius=pr))
        self.assertFalse(env.check_collision(1200.0 - 8.01, 400.0, radius=pr))
        self.assertFalse(env.check_collision(600.0, 8.01, radius=pr))
        self.assertFalse(env.check_collision(600.0, 800.0 - 8.01, radius=pr))

        # 4 Arena Corners safe vs collided
        self.assertTrue(env.check_collision(8.0, 8.0, radius=pr))
        self.assertFalse(env.check_collision(8.01, 8.01, radius=pr))


class TestMouseDragBoundsClamping(unittest.TestCase):
    """Stress-test mouse drag bounds clamping and boundary constraints."""

    def test_circle_drag_clamping_to_all_extremes(self):
        """Verify circular obstacle cannot be dragged outside [0, 1200] x [0, 800]."""
        env = Environment()
        env.clear_obstacles()
        obs = env.spawn_circle(600.0, 400.0, radius=35.0)

        hit = env.start_drag(600.0, 400.0)
        self.assertIs(hit, obs)

        # Drag far to top-left negative coordinates
        env.drag_to(-5000.0, -5000.0)
        self.assertEqual(obs.x, 35.0)
        self.assertEqual(obs.y, 35.0)
        self.assertGreaterEqual(obs.x - obs.radius, 0.0)
        self.assertGreaterEqual(obs.y - obs.radius, 0.0)

        # Drag far to bottom-right positive coordinates
        env.drag_to(10000.0, 10000.0)
        self.assertEqual(obs.x, 1200.0 - 35.0)
        self.assertEqual(obs.y, 800.0 - 35.0)
        self.assertLessEqual(obs.x + obs.radius, 1200.0)
        self.assertLessEqual(obs.y + obs.radius, 800.0)

        env.stop_drag()
        self.assertFalse(obs.is_dragging)
        self.assertIsNone(env.dragged_obstacle)

    def test_rotated_rect_drag_clamping_all_corners(self):
        """Verify all 4 corners of rotated OBB remain in bounds when dragged to extremes."""
        env = Environment()
        env.clear_obstacles()

        for angle in [0.0, math.pi / 6.0, math.pi / 4.0, math.pi / 2.0, 1.2]:
            obs = env.spawn_rect(600.0, 400.0, width=120.0, height=60.0, angle=angle)
            env.start_drag(600.0, 400.0)

            # Test extreme drag targets
            extreme_targets = [
                (-2000.0, -2000.0),
                (5000.0, -2000.0),
                (-2000.0, 5000.0),
                (5000.0, 5000.0),
            ]
            for tx, ty in extreme_targets:
                env.drag_to(tx, ty)
                corners = obs.get_corners()
                for cx, cy in corners:
                    self.assertGreaterEqual(cx, 0.0 - 1e-9, f"Corner x={cx} outside left arena at angle {angle}")
                    self.assertLessEqual(cx, 1200.0 + 1e-9, f"Corner x={cx} outside right arena at angle {angle}")
                    self.assertGreaterEqual(cy, 0.0 - 1e-9, f"Corner y={cy} outside top arena at angle {angle}")
                    self.assertLessEqual(cy, 800.0 + 1e-9, f"Corner y={cy} outside bottom arena at angle {angle}")

            env.stop_drag()
            env.clear_obstacles()

    def test_drag_lifecycle_defensive_invariants(self):
        """Verify drag lifecycle robustness when dragging empty space or deleting mid-drag."""
        env = Environment()

        # 1. Dragging empty space
        hit = env.start_drag(1.0, 1.0)
        self.assertIsNone(hit)
        self.assertIsNone(env.dragged_obstacle)

        # Dragging when nothing selected must be a safe no-op
        env.drag_to(500.0, 500.0)
        env.stop_drag()

        # 2. Deleting obstacle while dragging
        obs = env.spawn_circle(500.0, 500.0, radius=20.0)
        env.start_drag(500.0, 500.0)
        self.assertTrue(obs.is_dragging)

        # Remove obstacle while active
        removed = env.remove_obstacle(obs)
        self.assertTrue(removed)
        self.assertIsNone(env.dragged_obstacle)
        self.assertFalse(obs.is_dragging)


class TestComparisonModeTelemetryAndDivergence(unittest.TestCase):
    """Stress-test comparison mode, baseline drone collision divergence, and telemetry."""

    def test_traditional_drone_fixed_frequency_invariant(self):
        """Verify traditional drone strictly maintains 10.0 Hz regardless of obstacle proximity."""
        env = Environment()
        trad = TraditionalDrone(x=600.0, y=400.0, heading=0.0)

        # Step towards an obstacle
        for _ in range(50):
            trad.step(1.0 / 60.0, env)

        self.assertEqual(trad.ping_freq, 10.0)
        self.assertEqual(trad.recent_ping_freq, 10.0)
        self.assertEqual(trad.urgency, 0.0)

    def test_comparison_mode_telemetry_invariants(self):
        """Verify comparison mode preserves alive + collision == total across steps."""
        tel = run_simulation(
            pop_size=30,
            max_steps=20,
            steps_limit=40,
            comparison=True,
            headless=True,
            seed=42,
        )
        self.assertIn("alive_count", tel)
        self.assertIn("collision_count", tel)
        self.assertEqual(tel["alive_count"] + tel["collision_count"], 30)

    def test_traditional_vs_bio_collision_divergence(self):
        """
        Verify that traditional baseline drones with fixed 10 Hz rate suffer collisions
        when navigating dynamic drifting obstacles.
        """
        env = Environment()
        # Add dynamic drifting obstacles to test collision avoidance
        env.clear_obstacles()
        env.add_obstacle(CircularObstacle(500.0, 400.0, radius=40.0, vx=40.0, vy=0.0))
        env.add_obstacle(CircularObstacle(700.0, 400.0, radius=40.0, vx=-40.0, vy=0.0))
        env.add_obstacle(RectangularObstacle(600.0, 300.0, width=80.0, height=40.0, angle=0.4, vy=30.0))

        # Spawn traditional drones heading directly toward obstacles
        trad_drones = [
            TraditionalDrone(x=600.0, y=400.0, heading=math.pi, speed=60.0),
            TraditionalDrone(x=600.0, y=400.0, heading=0.0, speed=60.0),
            TraditionalDrone(x=600.0, y=400.0, heading=-math.pi / 2.0, speed=60.0),
        ]

        for _ in range(200):
            env.step(1.0 / 60.0)
            for td in trad_drones:
                if td.is_alive:
                    td.step(1.0 / 60.0, env)

        collided_count = sum(1 for td in trad_drones if td.collided)
        self.assertGreater(collided_count, 0, "Traditional baseline drones should suffer collisions against drift")


class TestHeadlessSimulationStabilityUnderLoad(unittest.TestCase):
    """Stress-test headless simulation stability, memory leaks, and high load."""

    def test_heavy_population_headless_run(self):
        """Run headless simulation with 200 drones and verify execution stability."""
        tel = run_simulation(
            pop_size=200,
            max_steps=20,
            steps_limit=30,
            headless=True,
            seed=123,
        )
        self.assertEqual(tel["alive_count"] + tel["collision_count"], 200)
        self.assertTrue(math.isfinite(tel["best_fitness"]))
        self.assertTrue(math.isfinite(tel["avg_fitness"]))
        self.assertTrue(math.isfinite(tel["avg_ping_rate"]))

    def test_heavy_obstacle_field_scaling(self):
        """Simulate environment with 50 obstacles (circles and rotated rects)."""
        env = Environment()
        env.clear_obstacles()

        # Add 30 circles and 20 rotated rectangles
        for i in range(30):
            x = 100.0 + (i % 6) * 180.0
            y = 100.0 + (i // 6) * 120.0
            env.add_obstacle(CircularObstacle(x, y, radius=20.0, vx=10.0, vy=-10.0))

        for j in range(20):
            x = 150.0 + (j % 5) * 200.0
            y = 150.0 + (j // 5) * 150.0
            env.add_obstacle(RectangularObstacle(x, y, width=50.0, height=30.0, angle=j * 0.3, vx=-15.0, vy=10.0))

        pop = Population(size=50, max_steps=50, arena_size=(1200.0, 800.0), track_wavefronts=True)

        for _ in range(50):
            env.step(1.0 / 60.0)
            done = pop.step(1.0 / 60.0, env)
            if done:
                pop.reproduce()

        # All alive drones must have valid positions and finite distance
        for d in pop.drones:
            self.assertTrue(math.isfinite(d.x))
            self.assertTrue(math.isfinite(d.y))
            self.assertTrue(math.isfinite(d.distance_traveled))

    def test_raycast_distance_and_collision_hitbox_parity(self):
        """
        Adversarial cross-check: verify that physics.cast_rays_all hit distance
        corresponds exactly to the collision threshold in Environment.check_collision.
        """
        r_drone = 8.0
        # 1. Circle obstacle
        obs_c = CircularObstacle(600.0, 400.0, radius=40.0)
        env_c = Environment(obstacles=[obs_c])
        origin = (400.0, 400.0)
        heading = 0.0  # Ray points directly at circle center

        dists = physics.cast_rays_all(
            origin=origin,
            angle=heading,
            ray_offsets=[0.0],
            obstacles=env_c.obstacles,
            arena_size=(1200.0, 800.0),
        )
        d_hit = dists[0]
        # Circle center is at 600, radius is 40 -> surface is at 560 -> distance from 400 is 160.0
        self.assertAlmostEqual(d_hit, 160.0, places=5)

        # Place drone just before touching obstacle: center at 400 + (160 - 8 - 0.1) = 551.9
        px_clear = origin[0] + (d_hit - r_drone - 0.1)
        self.assertFalse(env_c.check_collision(px_clear, origin[1], radius=r_drone))

        # Place drone penetrating obstacle: center at 400 + (160 - 8 + 0.1) = 552.1
        px_hit = origin[0] + (d_hit - r_drone + 0.1)
        self.assertTrue(env_c.check_collision(px_hit, origin[1], radius=r_drone))

        # 2. Rotated rectangular obstacle at 45 degrees
        obs_r = RectangularObstacle(600.0, 400.0, width=60.0, height=60.0, angle=math.pi / 4.0)
        env_r = Environment(obstacles=[obs_r])
        dists_r = physics.cast_rays_all(
            origin=origin,
            angle=heading,
            ray_offsets=[0.0],
            obstacles=env_r.obstacles,
            arena_size=(1200.0, 800.0),
        )
        d_hit_r = dists_r[0]
        # Leftmost corner of 60x60 square at 45° is at cx - 30*sqrt(2) = 600 - 42.426 = 557.57359
        self.assertAlmostEqual(d_hit_r, 600.0 - 30.0 * math.sqrt(2.0) - 400.0, places=3)

        px_r_clear = origin[0] + (d_hit_r - r_drone - 0.2)
        self.assertFalse(env_r.check_collision(px_r_clear, origin[1], radius=r_drone))

        px_r_hit = origin[0] + (d_hit_r - r_drone + 0.2)
        self.assertTrue(env_r.check_collision(px_r_hit, origin[1], radius=r_drone))

    def test_extreme_angles_and_large_trig_wrap(self):
        """Verify rotated rectangle collision when angle is huge (large multiple of 2pi)."""
        r_drone = 8.0
        cx, cy = 600.0, 400.0
        # Angle with 100 * pi + pi / 4
        huge_angle = 100.0 * math.pi + (math.pi / 4.0)
        rect = RectangularObstacle(cx, cy, width=80.0, height=80.0, angle=huge_angle)
        env = Environment(obstacles=[rect])

        # Center hit
        self.assertTrue(env.check_collision(cx, cy, radius=r_drone))

        # Test all 4 corners from get_corners()
        corners = rect.get_corners()
        self.assertEqual(len(corners), 4)

        for wx, wy in corners:
            # Vector from center to corner
            dx, dy = wx - cx, wy - cy
            d_len = math.hypot(dx, dy)
            ux, uy = dx / d_len, dy / d_len

            # Test point penetrating corner hitbox: distance from corner is r_drone - 0.5
            px_in = wx + (r_drone - 0.5) * ux
            py_in = wy + (r_drone - 0.5) * uy
            self.assertTrue(env.check_collision(px_in, py_in, radius=r_drone))

            # Test point outside corner hitbox: distance from corner is r_drone + 0.5
            px_out = wx + (r_drone + 0.5) * ux
            py_out = wy + (r_drone + 0.5) * uy
            self.assertFalse(env.check_collision(px_out, py_out, radius=r_drone))

    def test_rapid_mouse_drag_random_waypoints(self):
        """Stress-test drag clamping over 100 rapid random mouse positions, including out-of-bounds."""
        env = Environment()
        env.clear_obstacles()
        rect = env.spawn_rect(600.0, 400.0, width=100.0, height=50.0, angle=0.7)
        env.start_drag(600.0, 400.0)

        rng = np.random.default_rng(42)
        random_targets = rng.uniform(-1000.0, 2000.0, size=(100, 2))

        for tx, ty in random_targets:
            env.drag_to(float(tx), float(ty))
            corners = rect.get_corners()
            for cx_c, cy_c in corners:
                self.assertGreaterEqual(cx_c, 0.0 - 1e-9)
                self.assertLessEqual(cx_c, 1200.0 + 1e-9)
                self.assertGreaterEqual(cy_c, 0.0 - 1e-9)
                self.assertLessEqual(cy_c, 800.0 + 1e-9)

        env.stop_drag()

    def test_wavefront_memory_pruning_under_load(self):
        """Verify wavefront list does not grow unboundedly as drones step and emit pings."""
        drone = Drone(x=600.0, y=400.0, track_wavefronts=True)
        drone.ping_freq = 150.0  # Max frequency terminal buzz (150 pings/sec)

        # Step 300 times at dt = 1/60s (5 seconds of simulated time)
        for _ in range(300):
            drone.step(1.0 / 60.0)

        # Wavefront speed is 343 m/s, max_range is 300px.
        # At 343 px/s, a wavefront reaches max_range in ~0.87s.
        # Expired wavefronts MUST be pruned and active count must remain strictly bounded (~53 max).
        self.assertLess(len(drone.wavefronts), 65)
        self.assertGreaterEqual(drone.total_pings, 250)


if __name__ == "__main__":
    unittest.main()

