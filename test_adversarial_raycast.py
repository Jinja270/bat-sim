"""
bat_sim/test_adversarial_raycast.py
===================================
Adversarial stress-test suite challenging ray casting algorithms in bat_sim/physics.py:
1. Ray-circle interactions: exact tangency, sub-pixel grazing, origins inside circle,
   origins on perimeter, rays pointing 180 deg away, and zero-radius limits.
2. Ray-rectangle (AABB & OBB) interactions: rotations at 0, 30, 45, 90, 135, 180, 270 deg,
   full 360-degree rotation sweep vs independent geometric oracle, origins inside OBB.
3. Arena wall boundary conditions: center-to-wall orthogonal hits, center-to-corner hits,
   corner-to-interior diagonal hits, and wall boundary behavior.
4. Parity verification: 1,000 randomized drone-obstacle states comparing scalar
   cast_rays_with_doppler vs vectorized batch_cast_rays_all (asserting max diff < 1e-9).
"""

from __future__ import annotations

import math
import os
import sys
import time
import unittest
from typing import List, Tuple

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import bat_sim.physics as p


# ==============================================================================
# INDEPENDENT GEOMETRIC ORACLES FOR VERIFICATION
# ==============================================================================

def cross2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """2D cross product scalar (v1_x * v2_y - v1_y * v2_x)."""
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def oracle_ray_segment_intersect(
    O: np.ndarray,
    D: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> float | None:
    """
    Compute intersection distance between 2D ray (O + t*D, t > 0)
    and line segment (P1 to P2) using 2D cross product.
    Returns t > 1e-6 if intersection exists, else None.
    """
    V = P2 - P1
    denom = cross2d(D, V)
    if abs(denom) < 1e-12:
        return None  # Parallel or collinear

    num_t = cross2d(P1 - O, V)
    num_s = cross2d(P1 - O, D)
    t = num_t / denom
    s = num_s / denom

    if -1e-9 <= s <= 1.0 + 1e-9 and t > 1e-6:
        return float(t)
    return None


def oracle_ray_rect_intersect(
    origin: np.ndarray | Tuple[float, float],
    direction: np.ndarray | Tuple[float, float],
    center: np.ndarray | Tuple[float, float],
    size: Tuple[float, float],
    angle: float,
    max_range: float = 300.0,
) -> float:
    """
    Independent ground-truth oracle for ray-oriented rectangle intersection.
    Explicitly tests all 4 line segments of the rotated rectangle.
    """
    O = np.asarray(origin, dtype=np.float64)
    D = np.asarray(direction, dtype=np.float64)
    D = D / np.linalg.norm(D)
    C = np.asarray(center, dtype=np.float64)
    w, h = float(size[0]), float(size[1])
    hw, hh = w / 2.0, h / 2.0

    # 4 local vertices counter-clockwise
    local_verts = [
        np.array([-hw, -hh], dtype=np.float64),
        np.array([hw, -hh], dtype=np.float64),
        np.array([hw, hh], dtype=np.float64),
        np.array([-hw, hh], dtype=np.float64),
    ]

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    world_verts = [C + R @ v for v in local_verts]

    hits: List[float] = []
    for i in range(4):
        P1 = world_verts[i]
        P2 = world_verts[(i + 1) % 4]
        t = oracle_ray_segment_intersect(O, D, P1, P2)
        if t is not None:
            hits.append(t)

    # Check if origin is inside the rectangle
    rel_o = O - C
    ox_loc = rel_o[0] * cos_a + rel_o[1] * sin_a
    oy_loc = -rel_o[0] * sin_a + rel_o[1] * cos_a
    if abs(ox_loc) <= hw + 1e-6 and abs(oy_loc) <= hh + 1e-6:
        # Inside: return smallest positive exit distance
        if hits:
            return float(min(min(hits), max_range))

    if hits:
        return float(min(min(hits), max_range))
    return float(max_range)


# ==============================================================================
# TEST SUITE 1: RAY-CIRCLE ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestRayCircleAdversarial(unittest.TestCase):
    """Stress tests challenging ray-circle intersection geometry."""

    def test_circle_exact_tangency(self):
        """Test ray exactly tangent to circle perimeter (discriminant == 0)."""
        center = (100.0, 50.0)
        radius = 50.0
        # Line y = 100 touches circle at (100, 100)
        origin = (0.0, 100.0)
        direction = (1.0, 0.0)

        dist = p.cast_ray_circle(origin, direction, center, radius, max_range=300.0)
        # Expected hit distance from (0, 100) to (100, 100) is 100.0
        self.assertAlmostEqual(dist, 100.0, places=9)

    def test_circle_subpixel_grazing(self):
        """Test sub-pixel grazing rays slightly inside vs slightly outside the tangent."""
        center = (100.0, 50.0)
        radius = 50.0
        direction = (1.0, 0.0)

        # 1e-5 px outside circle -> discriminant < 0 -> must miss (max_range)
        o_outside = (0.0, 100.0 + 1e-5)
        dist_miss = p.cast_ray_circle(o_outside, direction, center, radius, max_range=300.0)
        self.assertEqual(dist_miss, 300.0)

        # 1e-5 px inside circle -> discriminant > 0 -> must hit near 100.0
        o_inside = (0.0, 100.0 - 1e-5)
        dist_hit = p.cast_ray_circle(o_inside, direction, center, radius, max_range=300.0)
        self.assertLess(dist_hit, 300.0)
        self.assertAlmostEqual(dist_hit, 100.0, delta=0.1)

    def test_circle_origin_at_center(self):
        """Test ray starting exactly at circle center in all 360-degree directions."""
        center = (200.0, 200.0)
        radius = 42.0

        for angle in np.linspace(0, 2 * np.pi, num=36, endpoint=False):
            direction = (math.cos(angle), math.sin(angle))
            dist = p.cast_ray_circle(center, direction, center, radius, max_range=300.0)
            self.assertAlmostEqual(dist, radius, places=9)

    def test_circle_origin_inside_eccentric(self):
        """Test ray origins inside circle but offset from center."""
        center = (100.0, 100.0)
        radius = 50.0

        # Origin at (120, 100), 20 px right of center
        # Looking right (+1, 0): boundary at x = 150 -> distance = 30.0
        d_right = p.cast_ray_circle((120.0, 100.0), (1.0, 0.0), center, radius)
        self.assertAlmostEqual(d_right, 30.0, places=9)

        # Looking left (-1, 0): boundary at x = 50 -> distance = 70.0
        d_left = p.cast_ray_circle((120.0, 100.0), (-1.0, 0.0), center, radius)
        self.assertAlmostEqual(d_left, 70.0, places=9)

        # Looking up (0, +1): x = 120 -> (120-100)^2 + (y-100)^2 = 2500 -> (y-100)^2 = 2100
        # y = 100 + sqrt(2100) -> distance = sqrt(2100)
        d_up = p.cast_ray_circle((120.0, 100.0), (0.0, 1.0), center, radius)
        self.assertAlmostEqual(d_up, math.sqrt(2100.0), places=9)

    def test_circle_origin_on_perimeter(self):
        """Test ray origins exactly on the perimeter pointing inward vs outward."""
        center = (100.0, 100.0)
        radius = 50.0

        # Perimeter point at (50, 100)
        p_on_rim = (50.0, 100.0)

        # Looking inward (+1, 0): passes through center to far side (150, 100) -> dist = 100.0
        d_inward = p.cast_ray_circle(p_on_rim, (1.0, 0.0), center, radius)
        self.assertAlmostEqual(d_inward, 100.0, places=7)

        # Looking outward (-1, 0): points directly away -> misses -> max_range
        d_outward = p.cast_ray_circle(p_on_rim, (-1.0, 0.0), center, radius, max_range=300.0)
        self.assertEqual(d_outward, 300.0)

    def test_circle_ray_180_deg_away(self):
        """Test rays pointing 180 degrees away from circle from an exterior point."""
        center = (150.0, 100.0)
        radius = 30.0
        origin = (50.0, 100.0)  # Circle is directly to the right

        # Pointing left (-1, 0) -> directly away
        dist = p.cast_ray_circle(origin, (-1.0, 0.0), center, radius, max_range=300.0)
        self.assertEqual(dist, 300.0)

        # Pointing diagonally away (-1, 1)
        dist_diag = p.cast_ray_circle(origin, (-1.0, 1.0), center, radius, max_range=300.0)
        self.assertEqual(dist_diag, 300.0)

    def test_circle_zero_radius(self):
        """Test degenerate circle with zero radius."""
        center = (100.0, 100.0)
        origin = (50.0, 100.0)
        # Directly along line of center
        dist = p.cast_ray_circle(origin, (1.0, 0.0), center, radius=0.0, max_range=300.0)
        self.assertAlmostEqual(dist, 50.0, places=9)


# ==============================================================================
# TEST SUITE 2: RAY-RECTANGLE (AABB & OBB) ROTATION STRESS TESTS
# ==============================================================================

class TestRayRectRotatedAdversarial(unittest.TestCase):
    """Stress tests challenging oriented bounding box (OBB) ray intersections."""

    def setUp(self):
        self.center = (500.0, 500.0)
        self.size = (60.0, 40.0)  # w=60, h=40, hw=30, hh=20
        self.origin = (400.0, 500.0)
        self.direction = (1.0, 0.0)

    def test_rect_cardinal_and_specified_rotations(self):
        """
        Test OBB at required angles: 0, 30, 45, 90, 135, 180, 270 degrees.
        Compare against independent geometric oracle.
        """
        test_angles_deg = [0, 30, 45, 90, 135, 180, 270]
        test_angles_rad = [math.radians(deg) for deg in test_angles_deg]

        for deg, rad in zip(test_angles_deg, test_angles_rad):
            rect_obb = (self.center[0], self.center[1], self.size[0], self.size[1], rad)
            tested_dist = p.cast_ray_rect(self.origin, self.direction, rect_obb, max_range=300.0)
            oracle_dist = oracle_ray_rect_intersect(
                self.origin, self.direction, self.center, self.size, rad, max_range=300.0
            )

            self.assertAlmostEqual(
                tested_dist,
                oracle_dist,
                places=9,
                msg=f"Mismatch at angle {deg} deg: tested={tested_dist}, oracle={oracle_dist}",
            )

    def test_rect_symmetry_invariance(self):
        """Verify 180-degree rotational symmetry: 0 deg == 180 deg, and 90 deg == 270 deg."""
        r0 = p.cast_ray_rect(self.origin, self.direction, (500, 500, 60, 40, 0.0))
        r180 = p.cast_ray_rect(self.origin, self.direction, (500, 500, 60, 40, math.pi))
        self.assertAlmostEqual(r0, r180, places=12)

        r90 = p.cast_ray_rect(self.origin, self.direction, (500, 500, 60, 40, math.pi / 2.0))
        r270 = p.cast_ray_rect(self.origin, self.direction, (500, 500, 60, 40, 3 * math.pi / 2.0))
        self.assertAlmostEqual(r90, r270, places=12)

    def test_rect_full_360_degree_sweep(self):
        """Adversarial sweep across 72 discrete angles (every 5 degrees from 0 to 355)."""
        angles = np.linspace(0, 2 * math.pi, num=72, endpoint=False)
        for ang in angles:
            rect_obb = (self.center[0], self.center[1], self.size[0], self.size[1], ang)
            tested = p.cast_ray_rect(self.origin, self.direction, rect_obb, max_range=300.0)
            oracle = oracle_ray_rect_intersect(
                self.origin, self.direction, self.center, self.size, ang, max_range=300.0
            )
            self.assertAlmostEqual(
                tested, oracle, places=9,
                msg=f"Sweep mismatch at {math.degrees(ang):.1f} deg: tested={tested}, oracle={oracle}"
            )

    def test_rect_origin_inside_obb(self):
        """Test ray starting inside a 45-degree rotated OBB."""
        # Square 40x40 at 45 degrees: hw=20, hh=20
        # From center (100, 100), facing (+1, 0)
        # Expected exit distance is 20 * sqrt(2) = 28.284271247
        rect = (100.0, 100.0, 40.0, 40.0, math.pi / 4.0)
        dist = p.cast_ray_rect((100.0, 100.0), (1.0, 0.0), rect)
        self.assertAlmostEqual(dist, 20.0 * math.sqrt(2.0), places=9)

    def test_rect_zero_dimension(self):
        """Test degenerate 0-width or 0-height rectangle."""
        rect_zero_w = (500.0, 500.0, 0.0, 40.0, 0.0)
        dist = p.cast_ray_rect((400.0, 500.0), (1.0, 0.0), rect_zero_w)
        # Hits the vertical line segment x = 500 at distance 100.0
        self.assertAlmostEqual(dist, 100.0, places=9)


# ==============================================================================
# TEST SUITE 3: ARENA WALL BOUNDARY & CORNER STRESS TESTS
# ==============================================================================

class TestArenaWallAdversarial(unittest.TestCase):
    """Stress tests challenging arena wall boundary calculations."""

    def test_arena_orthogonal_hits_from_center(self):
        """Cast rays from arena center (600, 400) orthogonally in 4 cardinal directions."""
        center = (600.0, 400.0)
        # Left wall (x = 0): distance = 600.0
        self.assertAlmostEqual(p.cast_ray_arena(center, (-1.0, 0.0)), 600.0, places=9)
        # Right wall (x = 1200): distance = 600.0
        self.assertAlmostEqual(p.cast_ray_arena(center, (1.0, 0.0)), 600.0, places=9)
        # Top wall (y = 0): distance = 400.0
        self.assertAlmostEqual(p.cast_ray_arena(center, (0.0, -1.0)), 400.0, places=9)
        # Bottom wall (y = 800): distance = 400.0
        self.assertAlmostEqual(p.cast_ray_arena(center, (0.0, 1.0)), 400.0, places=9)

    def test_arena_corner_hits_from_center(self):
        """Cast rays from center directly towards all 4 arena corners."""
        center = (600.0, 400.0)
        corners = [(0.0, 0.0), (1200.0, 0.0), (1200.0, 800.0), (0.0, 800.0)]
        expected_dist = math.sqrt(600.0**2 + 400.0**2)  # ~721.110255

        for cx, cy in corners:
            dir_vec = (cx - center[0], cy - center[1])
            dist = p.cast_ray_arena(center, dir_vec, max_range=1000.0)
            self.assertAlmostEqual(dist, expected_dist, places=9)

    def test_arena_from_corners_diagonal(self):
        """Cast rays from corners pointing diagonally into the arena."""
        # Top-left corner (0, 0) looking along (1, 1) / sqrt(2)
        # Nearest wall hit is bottom wall at y = 800 -> t = 800 * sqrt(2)
        dist = p.cast_ray_arena((0.0, 0.0), (1.0, 1.0), max_range=2000.0)
        self.assertAlmostEqual(dist, 800.0 * math.sqrt(2.0), places=9)

        # Bottom-right corner (1200, 800) looking along (-1, -1) / sqrt(2)
        # Nearest wall hit is top wall at y = 0 -> t = 800 * sqrt(2)
        dist2 = p.cast_ray_arena((1200.0, 800.0), (-1.0, -1.0), max_range=2000.0)
        self.assertAlmostEqual(dist2, 800.0 * math.sqrt(2.0), places=9)

    def test_arena_boundary_discrepancy_empirically(self):
        """
        Empirically document behavior when ray origin is exactly on the arena boundary.
        - cast_ray_arena uses strict `t > 0`, discarding hits when origin is on the wall facing into it.
        - batch_cast_rays_all computes `-px / safe_dx` directly, returning 0.0.
        """
        origin_on_left_wall = np.array([0.0, 400.0])
        dir_facing_left = np.array([-1.0, 0.0])

        # Scalar call ignores wall at t=0 and returns max_range
        scalar_dist = p.cast_ray_arena(origin_on_left_wall, dir_facing_left, max_range=300.0)
        self.assertEqual(scalar_dist, 300.0)

        # Batch call includes t=0 and returns 0.0
        batch_dist, _ = p.batch_cast_rays_all(
            origin_on_left_wall[None, :],
            np.array([math.pi]),
            np.array([[0.0, 0.0]]),
            ray_offsets=[0.0],
            max_range=300.0,
        )
        self.assertEqual(batch_dist[0, 0], 0.0)


# ==============================================================================
# TEST SUITE 4: SCALAR VS BATCH PARITY TEST (1,000 RANDOM STATES)
# ==============================================================================

class TestScalarBatchParity1000States(unittest.TestCase):
    """
    Stress test verifying strict parity between scalar cast_rays_with_doppler
    and vectorized batch_cast_rays_all across 1,000 randomized states.
    Asserts max difference < 1e-9 across both distances and Doppler velocities.
    """

    def test_parity_1000_batch_vs_scalar_fixed_scene(self):
        """
        Evaluate 1,000 randomized drone positions simultaneously in a batch
        against 1,000 sequential scalar cast_rays_with_doppler calls.
        """
        np.random.seed(42)
        N_TRIALS = 1000
        MAX_RANGE = 300.0
        ARENA_SIZE = (1200.0, 800.0)

        origins = np.random.uniform([10.0, 10.0], [1190.0, 790.0], size=(N_TRIALS, 2))
        angles = np.random.uniform(-np.pi, np.pi, size=(N_TRIALS,))
        vels = np.random.uniform(-10.0, 10.0, size=(N_TRIALS, 2))

        # Obstacle environment (5 circles, 5 OBBs)
        n_circ = 5
        c_centers = np.random.uniform([100.0, 100.0], [1100.0, 700.0], size=(n_circ, 2))
        c_radii = np.random.uniform(15.0, 45.0, size=(n_circ,))
        c_vels = np.random.uniform(-3.0, 3.0, size=(n_circ, 2))

        n_rect = 5
        r_centers = np.random.uniform([100.0, 100.0], [1100.0, 700.0], size=(n_rect, 2))
        r_sizes = np.random.uniform([30.0, 30.0], [80.0, 80.0], size=(n_rect, 2))
        r_angs = np.random.uniform(0, 2 * np.pi, size=(n_rect,))
        r_vels = np.random.uniform(-3.0, 3.0, size=(n_rect, 2))

        obstacles = []
        for i in range(n_circ):
            obstacles.append({
                "type": "circle",
                "center": c_centers[i],
                "radius": c_radii[i],
                "velocity": c_vels[i],
            })
        for i in range(n_rect):
            obstacles.append({
                "type": "rect",
                "rect": (r_centers[i, 0], r_centers[i, 1], r_sizes[i, 0], r_sizes[i, 1], r_angs[i]),
                "velocity": r_vels[i],
            })

        # 1. Execute vectorized batch implementation
        t_batch_start = time.perf_counter()
        batch_dists, batch_front_vc = p.batch_cast_rays_all(
            origins,
            angles,
            vels,
            circle_centers=c_centers,
            circle_radii=c_radii,
            circle_velocities=c_vels,
            rect_centers=r_centers,
            rect_sizes=r_sizes,
            rect_angles=r_angs,
            rect_velocities=r_vels,
            arena_size=ARENA_SIZE,
            ray_offsets=p.RAY_ANGLES,
            max_range=MAX_RANGE,
        )
        batch_duration = time.perf_counter() - t_batch_start

        # 2. Execute scalar implementation iteratively
        t_scalar_start = time.perf_counter()
        scalar_dists = np.empty_like(batch_dists)
        scalar_front_vc = np.empty(N_TRIALS)

        for i in range(N_TRIALS):
            dists, closing_arr, front_vc = p.cast_rays_with_doppler(
                origins[i],
                angles[i],
                ray_offsets=p.RAY_ANGLES,
                drone_velocity=vels[i],
                obstacles=obstacles,
                arena_size=ARENA_SIZE,
                max_range=MAX_RANGE,
            )
            scalar_dists[i] = dists
            scalar_front_vc[i] = front_vc
        scalar_duration = time.perf_counter() - t_scalar_start

        # 3. Compute error norms
        dist_diffs = np.abs(batch_dists - scalar_dists)
        vc_diffs = np.abs(batch_front_vc - scalar_front_vc)

        max_dist_error = float(np.max(dist_diffs))
        max_vc_error = float(np.max(vc_diffs))

        # Log performance
        speedup = scalar_duration / max(1e-9, batch_duration)
        print(f"\n[PARITY PERFORMANCE - FIXED SCENE]")
        print(f"  Batch 1000 drones:  {batch_duration * 1000:.2f} ms")
        print(f"  Scalar 1000 drones: {scalar_duration * 1000:.2f} ms")
        print(f"  Vectorized Speedup: {speedup:.1f}x")
        print(f"  Max Distance Diff:  {max_dist_error:.3e}")
        print(f"  Max Doppler Diff:   {max_vc_error:.3e}")

        # Assert contract: difference < 1e-9
        self.assertLess(max_dist_error, 1e-9)
        self.assertLess(max_vc_error, 1e-9)

    def test_parity_1000_dynamic_random_scenes(self):
        """
        Evaluate 1,000 independent trials where each trial features a randomized
        drone state and randomized obstacle configurations (varying counts, positions,
        dimensions, rotation angles, and velocities).
        """
        np.random.seed(999)
        N_TRIALS = 1000
        max_d_diff = 0.0
        max_v_diff = 0.0

        for trial in range(N_TRIALS):
            origin = np.random.uniform([20.0, 20.0], [1180.0, 780.0], size=(2,))
            angle = float(np.random.uniform(-np.pi, np.pi))
            vel = np.random.uniform(-10.0, 10.0, size=(2,))

            nc = np.random.randint(1, 4)
            nr = np.random.randint(1, 4)
            c_centers = np.random.uniform([50.0, 50.0], [1150.0, 750.0], size=(nc, 2))
            c_radii = np.random.uniform(10.0, 40.0, size=(nc,))
            c_vels = np.random.uniform(-5.0, 5.0, size=(nc, 2))

            r_centers = np.random.uniform([50.0, 50.0], [1150.0, 750.0], size=(nr, 2))
            r_sizes = np.random.uniform([20.0, 20.0], [80.0, 80.0], size=(nr, 2))
            r_angs = np.random.uniform(-np.pi, np.pi, size=(nr,))
            r_vels = np.random.uniform(-5.0, 5.0, size=(nr, 2))

            obstacles = []
            for i in range(nc):
                obstacles.append({
                    "type": "circle",
                    "center": c_centers[i],
                    "radius": c_radii[i],
                    "velocity": c_vels[i],
                })
            for i in range(nr):
                obstacles.append({
                    "type": "rect",
                    "rect": (r_centers[i, 0], r_centers[i, 1], r_sizes[i, 0], r_sizes[i, 1], r_angs[i]),
                    "velocity": r_vels[i],
                })

            s_dists, _, s_fvc = p.cast_rays_with_doppler(origin, angle, p.RAY_ANGLES, vel, obstacles=obstacles)
            b_dists, b_fvc = p.batch_cast_rays_all(
                origin[None, :], np.array([angle]), vel[None, :],
                circle_centers=c_centers, circle_radii=c_radii, circle_velocities=c_vels,
                rect_centers=r_centers, rect_sizes=r_sizes, rect_angles=r_angs, rect_velocities=r_vels,
            )

            d_diff = float(np.max(np.abs(s_dists - b_dists[0])))
            v_diff = float(abs(s_fvc - b_fvc[0]))
            if d_diff > max_d_diff:
                max_d_diff = d_diff
            if v_diff > max_v_diff:
                max_v_diff = v_diff

        print(f"\n[PARITY PERFORMANCE - DYNAMIC SCENES]")
        print(f"  Trials:             {N_TRIALS}")
        print(f"  Max Distance Diff:  {max_d_diff:.3e}")
        print(f"  Max Doppler Diff:   {max_v_diff:.3e}")

        self.assertLess(max_d_diff, 1e-9)
        self.assertLess(max_v_diff, 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
