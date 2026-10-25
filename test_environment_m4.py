"""
bat_sim/test_environment_m4.py
==============================
Unit Test Suite for Milestone 4: Interactive Pygame Environment & Main Simulation.

Covers:
- CircularObstacle: kinematics, containment, bouncing, property setters/getters.
- RectangularObstacle: OBB containment, corner projection, boundary bouncing, property setters/getters.
- Environment: default presets, obstacle management, mouse drag/spawn interactions,
  collision detection (circle-circle, circle-rect OBB, boundary walls, dual signature),
  and formatted vectorized raycast properties.
- Main Simulation: headless display initialization, rendering helpers, and simulation runner.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import bat_sim.physics as physics
from bat_sim.drone import Drone, TraditionalDrone
from bat_sim.evolution import Population
from bat_sim.environment import (
    CircularObstacle,
    Environment,
    RectangularObstacle,
    DEFAULT_ARENA_WIDTH,
    DEFAULT_ARENA_HEIGHT,
)
from bat_sim.main import (
    init_display,
    render_sensor_rays,
    render_hud,
    render_comparison_hud,
    run_simulation,
)


class TestCircularObstacle(unittest.TestCase):
    """Test suite for CircularObstacle kinematics and properties."""

    def test_initialization_and_properties(self):
        c = CircularObstacle(100.0, 200.0, radius=25.0, vx=10.0, vy=-15.0)
        self.assertEqual(c.center, (100.0, 200.0))
        self.assertEqual(c.pos, (100.0, 200.0))
        self.assertEqual(c.position, (100.0, 200.0))
        self.assertEqual(c.velocity, (10.0, -15.0))
        self.assertEqual(c.radius, 25.0)
        self.assertFalse(c.is_dragging)

        # Setters
        c.center = (300.0, 400.0)
        self.assertEqual(c.x, 300.0)
        self.assertEqual(c.y, 400.0)
        c.velocity = (5.0, -5.0)
        self.assertEqual(c.vx, 5.0)
        self.assertEqual(c.vy, -5.0)

    def test_contains_point(self):
        c = CircularObstacle(200.0, 200.0, radius=30.0)
        self.assertTrue(c.contains_point(200.0, 200.0))      # Center
        self.assertTrue(c.contains_point(220.0, 200.0))      # Inside (dist=20)
        self.assertTrue(c.contains_point(230.0, 200.0))      # On border (dist=30)
        self.assertFalse(c.contains_point(230.1, 200.0))     # Outside (dist=30.1)

    def test_boundary_bounce(self):
        bounds = (0.0, 0.0, 1200.0, 800.0)
        # Right wall bounce
        c_right = CircularObstacle(1190.0, 400.0, radius=20.0, vx=50.0, vy=0.0)
        c_right.update(0.5, bounds)
        self.assertEqual(c_right.vx, -50.0)
        self.assertEqual(c_right.x, 1180.0)

        # Left wall bounce
        c_left = CircularObstacle(10.0, 400.0, radius=20.0, vx=-50.0, vy=0.0)
        c_left.update(0.5, bounds)
        self.assertEqual(c_left.vx, 50.0)
        self.assertEqual(c_left.x, 20.0)

        # Top wall bounce
        c_top = CircularObstacle(600.0, 10.0, radius=20.0, vx=0.0, vy=-50.0)
        c_top.update(0.5, bounds)
        self.assertEqual(c_top.vy, 50.0)
        self.assertEqual(c_top.y, 20.0)

        # Bottom wall bounce
        c_bottom = CircularObstacle(600.0, 790.0, radius=20.0, vx=0.0, vy=50.0)
        c_bottom.update(0.5, bounds)
        self.assertEqual(c_bottom.vy, -50.0)
        self.assertEqual(c_bottom.y, 780.0)

    def test_draw_headless_safe(self):
        c = CircularObstacle(100.0, 100.0, radius=20.0, vx=10.0, vy=10.0)
        c.draw(None)  # Must execute cleanly without error


class TestRectangularObstacle(unittest.TestCase):
    """Test suite for RectangularObstacle kinematics and OBB math."""

    def test_initialization_and_properties(self):
        r = RectangularObstacle(400.0, 300.0, width=80.0, height=40.0, angle=0.5, vx=10.0, vy=-10.0)
        self.assertEqual(r.center, (400.0, 300.0))
        self.assertEqual(r.size, (80.0, 40.0))
        self.assertEqual(r.velocity, (10.0, -10.0))
        self.assertEqual(r.rect, (400.0, 300.0, 80.0, 40.0, 0.5))
        self.assertFalse(r.is_dragging)

        # Setters
        r.center = (500.0, 350.0)
        self.assertEqual(r.x, 500.0)
        self.assertEqual(r.y, 350.0)
        r.size = (100.0, 50.0)
        self.assertEqual(r.width, 100.0)
        self.assertEqual(r.height, 50.0)

    def test_contains_point_aabb(self):
        r = RectangularObstacle(400.0, 300.0, width=80.0, height=40.0, angle=0.0)
        self.assertTrue(r.contains_point(400.0, 300.0))      # Center
        self.assertTrue(r.contains_point(439.0, 319.0))      # Inside
        self.assertTrue(r.contains_point(440.0, 320.0))      # Corner
        self.assertFalse(r.contains_point(441.0, 300.0))     # Outside x
        self.assertFalse(r.contains_point(400.0, 321.0))     # Outside y

    def test_contains_point_obb_rotated(self):
        # 90 degrees rotation: width along y-axis, height along x-axis
        r_rot = RectangularObstacle(400.0, 300.0, width=80.0, height=40.0, angle=math.pi / 2.0)
        # Point (400, 335) is 35 along y, within width/2 = 40
        self.assertTrue(r_rot.contains_point(400.0, 335.0))
        # Point (430, 300) is 30 along x, outside height/2 = 20
        self.assertFalse(r_rot.contains_point(430.0, 300.0))

    def test_corners_count_and_shape(self):
        r = RectangularObstacle(400.0, 300.0, width=80.0, height=40.0, angle=0.25)
        corners = r.get_corners()
        self.assertEqual(len(corners), 4)
        for pt in corners:
            self.assertEqual(len(pt), 2)

    def test_projected_boundary_bounce(self):
        bounds = (0.0, 0.0, 1200.0, 800.0)
        # At angle=0, hw_proj=40, hh_proj=20
        r_right = RectangularObstacle(1180.0, 400.0, width=80.0, height=40.0, angle=0.0, vx=50.0)
        r_right.update(0.5, bounds)
        self.assertEqual(r_right.vx, -50.0)
        self.assertAlmostEqual(r_right.x, 1160.0)

    def test_draw_headless_safe(self):
        r = RectangularObstacle(100.0, 100.0, width=60.0, height=30.0, vx=10.0, vy=5.0)
        r.draw(None)  # Must execute cleanly without error


class TestEnvironment(unittest.TestCase):
    """Test suite for Environment management, mouse interactions, and collision checks."""

    def test_default_presets(self):
        env = Environment()
        self.assertEqual(env.arena_width, DEFAULT_ARENA_WIDTH)
        self.assertEqual(env.arena_height, DEFAULT_ARENA_HEIGHT)
        self.assertEqual(len(env.obstacles), 9)

        # Ensure spawn center (600, 400) is clear
        self.assertFalse(env.check_collision(600.0, 400.0, radius=8.0))

    def test_array_properties_for_vectorized_raycasting(self):
        env = Environment()
        self.assertIsNotNone(env.circle_centers)
        self.assertEqual(env.circle_centers.shape[1], 2)
        self.assertEqual(env.circle_radii.shape[0], env.circle_centers.shape[0])
        self.assertEqual(env.circle_velocities.shape[1], 2)

        self.assertIsNotNone(env.rect_centers)
        self.assertEqual(env.rect_centers.shape[1], 2)
        self.assertEqual(env.rect_sizes.shape[1], 2)
        self.assertEqual(env.rect_angles.shape[0], env.rect_centers.shape[0])
        self.assertEqual(env.rect_velocities.shape[1], 2)

    def test_empty_environment_array_properties(self):
        env = Environment()
        env.clear_obstacles()
        self.assertIsNone(env.circle_centers)
        self.assertIsNone(env.circle_radii)
        self.assertIsNone(env.circle_velocities)
        self.assertIsNone(env.rect_centers)
        self.assertIsNone(env.rect_sizes)
        self.assertIsNone(env.rect_angles)
        self.assertIsNone(env.rect_velocities)

    def test_collision_circle_circle(self):
        env = Environment()
        env.clear_obstacles()
        env.add_obstacle(CircularObstacle(500.0, 400.0, radius=30.0))

        # Overlapping: dist=25 <= 38
        self.assertTrue(env.check_collision(500.0, 425.0, radius=8.0))
        # Clear: dist=50 > 38
        self.assertFalse(env.check_collision(500.0, 450.0, radius=8.0))

    def test_collision_circle_rect_obb(self):
        env = Environment()
        env.clear_obstacles()
        env.add_obstacle(RectangularObstacle(700.0, 400.0, width=60.0, height=40.0, angle=0.0))

        # Inside rectangle
        self.assertTrue(env.check_collision(700.0, 400.0, radius=8.0))
        # Overlapping boundary
        self.assertTrue(env.check_collision(665.0, 400.0, radius=8.0))
        # Clear outside
        self.assertFalse(env.check_collision(650.0, 400.0, radius=8.0))

    def test_collision_boundary_walls(self):
        env = Environment()
        env.clear_obstacles()

        self.assertTrue(env.check_collision(5.0, 400.0, radius=8.0))      # Left wall
        self.assertTrue(env.check_collision(1195.0, 400.0, radius=8.0))   # Right wall
        self.assertTrue(env.check_collision(600.0, 5.0, radius=8.0))      # Top wall
        self.assertTrue(env.check_collision(600.0, 795.0, radius=8.0))    # Bottom wall

    def test_collision_dual_invocation_signatures(self):
        env = Environment()
        env.clear_obstacles()
        env.add_obstacle(CircularObstacle(500.0, 400.0, radius=30.0))

        d_hit = Drone(x=500.0, y=425.0, radius=8.0)
        d_miss = Drone(x=600.0, y=400.0, radius=8.0)

        # Drone object signature
        self.assertTrue(env.check_collision(d_hit))
        self.assertFalse(env.check_collision(d_miss))

        # Coordinate tuple signature
        self.assertTrue(env.check_collision(500.0, 425.0, 8.0))
        self.assertFalse(env.check_collision(600.0, 400.0, 8.0))

    def test_mouse_drag_and_spawn(self):
        env = Environment()
        obs = env.start_drag(350.0, 230.0)
        self.assertIsNotNone(obs)
        self.assertTrue(obs.is_dragging)

        env.drag_to(390.0, 270.0)
        self.assertEqual(obs.x, 390.0)
        self.assertEqual(obs.y, 270.0)

        env.stop_drag()
        self.assertFalse(obs.is_dragging)
        self.assertIsNone(env.dragged_obstacle)

        # Spawning
        c_new = env.spawn_circle(600.0, 500.0, radius=22.0)
        self.assertIn(c_new, env.obstacles)
        r_new = env.spawn_rect(650.0, 500.0, width=50.0, height=30.0)
        self.assertIn(r_new, env.obstacles)

        # Removal
        self.assertTrue(env.remove_obstacle(c_new))
        self.assertNotIn(c_new, env.obstacles)


class TestMainSimulation(unittest.TestCase):
    """Test suite for main.py simulation loop and rendering in headless mode."""

    def test_headless_display_init(self):
        pygame_mod, screen = init_display(headless=True)
        self.assertIsNotNone(screen)
        self.assertEqual(screen.get_size(), (1200, 800))
        pygame_mod.quit()

    def test_run_simulation_headless_step_limit(self):
        tel = run_simulation(pop_size=20, max_steps=10, steps_limit=25, headless=True)
        self.assertIn("alive_count", tel)
        self.assertIn("generation", tel)
        self.assertIn("survival_rate", tel)

    def test_run_simulation_headless_comparison_mode(self):
        tel = run_simulation(pop_size=15, max_steps=5, steps_limit=15, comparison=True, headless=True)
        self.assertIn("alive_count", tel)
        self.assertIn("collision_count", tel)


if __name__ == "__main__":
    unittest.main()
