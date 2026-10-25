"""
bat_sim/physics.py
==================
Bio-Acoustic Physics & Sensor Engine for Autonomous Bat Echolocation Simulation.

Implements fundamental acoustic physics and sensor simulation:
- Exact Time-of-Flight (ToF) acoustic echo distance and round-trip delay.
- Two-way acoustic Doppler shift closing velocity and observed frequency.
- Dynamic pulse rate frequency mapping (terminal buzz) based on neural urgency.
- High-precision 2D ray casting against circles, rectangles (AABB & OBB), and arena walls.
- Multi-ray acoustic sensing with Doppler closing velocity detection.
- High-throughput vectorized batch ray casting for population simulation.
- 5-element sensor input normalization and denormalization supporting single-drone and batch shapes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ==============================================================================
# PHYSICAL AND SIMULATION CONSTANTS
# ==============================================================================

SPEED_OF_SOUND: float = 343.0
"""Acoustic sound propagation speed in dry air at 20°C in m/s (or units/s)."""

DEFAULT_PULSE_FREQ_MIN: float = 10.0
"""Minimum pulse emission frequency in Hz (search phase)."""
PULSE_RATE_MIN: float = DEFAULT_PULSE_FREQ_MIN

DEFAULT_PULSE_FREQ_MAX: float = 150.0
"""Maximum pulse emission frequency in Hz (terminal buzz phase)."""
PULSE_RATE_MAX: float = DEFAULT_PULSE_FREQ_MAX

DEFAULT_CARRIER_FREQ: float = 40000.0
"""Default ultrasonic pulse carrier frequency in Hz (40 kHz)."""
DEFAULT_EMIT_FREQ: float = DEFAULT_CARRIER_FREQ

DEFAULT_MAX_RANGE: float = 300.0
"""Default maximum echolocation perception range in simulation units/pixels."""

DEFAULT_MAX_SPEED: float = 100.0
"""Default maximum drone linear forward speed in simulation units/s."""

ARENA_WIDTH: float = 1200.0
"""Default width of simulation arena in pixels."""
DEFAULT_ARENA_WIDTH: float = ARENA_WIDTH

ARENA_HEIGHT: float = 800.0
"""Default height of simulation arena in pixels."""
DEFAULT_ARENA_HEIGHT: float = ARENA_HEIGHT

RAY_ANGLES: Tuple[float, float, float] = (-math.pi / 6.0, 0.0, math.pi / 6.0)
"""Acoustic ray angular offsets relative to heading: Left (-30°), Front (0°), Right (+30°)."""
RAY_OFFSETS: Tuple[float, float, float] = RAY_ANGLES
DEFAULT_RAY_OFFSETS: Tuple[float, float, float] = RAY_ANGLES

SENSOR_INPUT_DIM: int = 5
"""Number of normalized features in the neural network perception input vector."""

SENSOR_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 1.0),   # Feature 0: Left ray distance / max_range
    (0.0, 1.0),   # Feature 1: Right ray distance / max_range
    (0.0, 1.0),   # Feature 2: Front ray distance / max_range
    (0.0, 1.0),   # Feature 3: Forward drone speed / max_speed
    (-1.0, 1.0),  # Feature 4: Frontal closing velocity / max_speed
)
"""Exact lower and upper clamping bounds for each normalized sensor feature."""


# ==============================================================================
# ACOUSTIC PHYSICS & PULSE RATE FUNCTIONS
# ==============================================================================

def tof_distance(delta_t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate physical distance from round-trip Time-of-Flight (ToF) echo delay.

    Formula:
        d = (c * delta_t) / 2

    Args:
        delta_t: Round-trip echo delay in seconds (must be non-negative).

    Returns:
        Distance to reflector in meters/units.

    Raises:
        ValueError: If delta_t contains negative values.
    """
    arr = np.asarray(delta_t)
    if np.any(arr < 0.0):
        raise ValueError(f"Time of flight delta_t must be non-negative (>= 0.0), got {delta_t}")
    result = (SPEED_OF_SOUND * arr) / 2.0
    return float(result) if np.isscalar(delta_t) else result


def tof_time(distance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate round-trip Time-of-Flight (ToF) delay for a given physical distance.

    Formula:
        delta_t = (2 * d) / c

    Args:
        distance: One-way distance to reflector in meters/units (must be non-negative).

    Returns:
        Round-trip echo travel time in seconds.

    Raises:
        ValueError: If distance contains negative values.
    """
    arr = np.asarray(distance)
    if np.any(arr < 0.0):
        raise ValueError(f"Distance must be non-negative (>= 0.0), got {distance}")
    result = (2.0 * arr) / SPEED_OF_SOUND
    return float(result) if np.isscalar(distance) else result


def doppler_velocity(
    f_obs: Union[float, np.ndarray],
    f_emit: Union[float, np.ndarray] = DEFAULT_CARRIER_FREQ,
) -> Union[float, np.ndarray]:
    """
    Calculate relative closing velocity from emitted and observed frequencies using two-way Doppler shift.

    Formula:
        v_closing = c * (f_obs - f_emit) / (2 * f_emit)

    Sign Convention:
        v_closing > 0: Distance decreasing (approaching target, blue-shifted).
        v_closing < 0: Distance increasing (receding target, red-shifted).
        v_closing == 0: Relative distance stationary.

    Args:
        f_obs: Observed echo frequency in Hz (negative frequencies algebraically represent
            closing velocities v_closing < -c/2).
        f_emit: Emitted acoustic carrier frequency in Hz (must be strictly positive).

    Returns:
        Relative closing velocity in m/s (or units/s).

    Raises:
        ValueError: If f_emit <= 0.0.
    """
    arr_emit = np.asarray(f_emit)
    arr_obs = np.asarray(f_obs)

    if np.any(arr_emit <= 0.0):
        raise ValueError(f"f_emit must be strictly positive (> 0.0), got {f_emit}")

    result = SPEED_OF_SOUND * (arr_obs - arr_emit) / (2.0 * arr_emit)
    if np.isscalar(f_obs) and np.isscalar(f_emit):
        return float(result)
    return result


def doppler_observed_freq(
    v_closing: Union[float, np.ndarray],
    f_emit: Union[float, np.ndarray] = DEFAULT_CARRIER_FREQ,
) -> Union[float, np.ndarray]:
    """
    Calculate observed echo frequency under two-way Doppler shift given closing velocity.

    Formula:
        f_obs = f_emit * (1 + 2 * v_closing / c)

    Args:
        v_closing: Relative closing velocity in m/s (positive = closing, negative = opening).
        f_emit: Emitted acoustic carrier frequency in Hz (must be strictly positive).

    Returns:
        Observed echo frequency in Hz.

    Raises:
        ValueError: If f_emit <= 0.0.
    """
    arr_emit = np.asarray(f_emit)
    arr_vc = np.asarray(v_closing)

    if np.any(arr_emit <= 0.0):
        raise ValueError(f"f_emit must be strictly positive (> 0.0), got {f_emit}")

    result = arr_emit * (1.0 + (2.0 * arr_vc) / SPEED_OF_SOUND)
    if np.isscalar(v_closing) and np.isscalar(f_emit):
        return float(result)
    return result


def urgency_to_pulse_rate(
    urgency: Union[float, np.ndarray],
    clamp: bool = True,
) -> Union[float, np.ndarray]:
    """
    Map neural urgency lambda in [0.0, 1.0] to pulse emission rate in [10.0, 150.0] Hz.

    Formula:
        f_ping = 10.0 + 140.0 * lambda

    Biological context:
        lambda = 0.0 -> 10.0 Hz (Search phase: energy-efficient, long range)
        lambda = 0.5 -> 80.0 Hz (Approach phase: intermediate tracking)
        lambda = 1.0 -> 150.0 Hz (Terminal buzz phase: dense rapid sampling)

    Args:
        urgency: Dimensionless urgency factor nominally in [0.0, 1.0].
        clamp: If True, clamp urgency into [0.0, 1.0]. If False and out of bounds, raises ValueError.

    Returns:
        Pulse emission rate f_ping in Hz.

    Raises:
        ValueError: If clamp is False and urgency contains values outside [0.0, 1.0].
    """
    arr = np.asarray(urgency)
    if not clamp and (np.any(arr < 0.0) or np.any(arr > 1.0)):
        raise ValueError(f"Urgency must be in [0.0, 1.0], got {urgency}")

    delta_rate = PULSE_RATE_MAX - PULSE_RATE_MIN
    clamped_u = np.clip(arr, 0.0, 1.0) if clamp else arr
    result = PULSE_RATE_MIN + delta_rate * clamped_u
    return float(result) if np.isscalar(urgency) else result


def pulse_rate_to_urgency(
    pulse_rate: Union[float, np.ndarray],
    clamp: bool = True,
) -> Union[float, np.ndarray]:
    """
    Inverse mapping: Convert pulse emission rate in Hz back to neural urgency lambda in [0.0, 1.0].

    Formula:
        lambda = (f_ping - 10.0) / 140.0

    Args:
        pulse_rate: Pulse rate in Hz.
        clamp: If True, clamp resulting urgency to [0.0, 1.0].

    Returns:
        Urgency factor lambda in [0.0, 1.0].
    """
    delta_rate = PULSE_RATE_MAX - PULSE_RATE_MIN
    arr = np.asarray(pulse_rate)
    result = (arr - PULSE_RATE_MIN) / delta_rate
    if clamp:
        result = np.clip(result, 0.0, 1.0)
    return float(result) if np.isscalar(pulse_rate) else result


# ==============================================================================
# RAY-CASTING GEOMETRY FUNCTIONS
# ==============================================================================

def compute_ray_directions(
    heading_angle: float,
    ray_offsets: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Compute unit direction vectors for sensor rays given drone heading angle.

    Args:
        heading_angle: Drone heading orientation in radians.
        ray_offsets: Optional sequence of angular offsets relative to heading.
                     Defaults to RAY_ANGLES (-pi/6, 0.0, +pi/6).

    Returns:
        np.ndarray: Array of shape (K, 2) containing unit vectors [cos(theta), sin(theta)].
    """
    if ray_offsets is None:
        offsets = np.asarray(RAY_ANGLES, dtype=np.float64)
    else:
        offsets = np.asarray(ray_offsets, dtype=np.float64)
    angles = float(heading_angle) + offsets
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def cast_ray_circle(
    origin: Union[Sequence[float], np.ndarray],
    direction: Union[Sequence[float], np.ndarray],
    circle_center: Union[Sequence[float], np.ndarray],
    radius: float,
    max_range: float = DEFAULT_MAX_RANGE,
) -> float:
    """
    Compute nearest forward intersection distance between a 2D ray and a circular obstacle.

    Solves the quadratic equation:
        ||(origin - center) + t * direction||^2 = radius^2
        t^2 + 2b*t + c = 0
    where b = (origin - center) . direction, c = ||origin - center||^2 - radius^2.

    Args:
        origin: 2D ray origin [x, y].
        direction: 2D ray direction vector [dx, dy].
        circle_center: 2D circle center [cx, cy].
        radius: Circle obstacle radius (positive float).
        max_range: Maximum detection distance horizon.

    Returns:
        float: Nearest forward intersection distance in (0, max_range],
               or max_range if no intersection exists.
    """
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    c = np.asarray(circle_center, dtype=np.float64)
    r = float(radius)

    dir_norm = float(np.linalg.norm(d))
    if dir_norm < 1e-9:
        return float(max_range)
    d = d / dir_norm

    m = o - c
    b = float(np.dot(m, d))
    c_eq = float(np.dot(m, m) - r * r)

    disc = b * b - c_eq
    if disc < 0.0:
        return float(max_range)

    sqrt_disc = math.sqrt(disc)
    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc

    eps = 1e-6
    if t1 > eps:
        t = t1
    elif t2 > eps:
        t = t2
    else:
        return float(max_range)

    return float(min(t, max_range))


def cast_ray_rect(
    origin: Union[Sequence[float], np.ndarray],
    direction: Union[Sequence[float], np.ndarray],
    rect: Union[Sequence[float], Tuple, np.ndarray, Any],
    max_range: float = DEFAULT_MAX_RANGE,
) -> float:
    """
    Compute nearest forward intersection distance between a 2D ray and a rectangle obstacle.
    Supports both Axis-Aligned Bounding Boxes (AABB) and Oriented Bounding Boxes (OBB).

    Uses local coordinate frame transformation R(-phi) and the slab method:
        o' = R(-phi) * (o - center)
        d' = R(-phi) * d
    Distance is invariant under isometry: t_world == t_local.

    Args:
        origin: 2D ray origin point [x, y].
        direction: 2D ray direction vector [dx, dy].
        rect: Rectangle descriptor:
              - 4-tuple: (x, y, w, h) [AABB with top-left (x, y) and size (w, h)]
              - 5-tuple: (cx, cy, w, h, angle) [OBB centered at (cx, cy) rotated by angle in rad]
              - pygame.Rect-like object with center/x/y/w/h attributes.
        max_range: Maximum detection distance.

    Returns:
        float: Nearest forward intersection distance in (0, max_range],
               or max_range if no intersection.
    """
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)

    dir_norm = float(np.linalg.norm(d))
    if dir_norm < 1e-9:
        return float(max_range)
    d = d / dir_norm

    # Parse rectangle specification
    if hasattr(rect, "center") and hasattr(rect, "width") and hasattr(rect, "height"):
        cx, cy = float(rect.center[0]), float(rect.center[1])
        w, h = float(rect.width), float(rect.height)
        angle = getattr(rect, "angle", 0.0)
    elif isinstance(rect, (list, tuple, np.ndarray)):
        if len(rect) == 4:
            x, y, w, h = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
            cx = x + w / 2.0
            cy = y + h / 2.0
            angle = 0.0
        elif len(rect) == 5:
            cx, cy, w, h, angle = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]), float(rect[4])
        else:
            raise ValueError(f"Rectangle sequence must have length 4 or 5, got {len(rect)}")
    elif isinstance(rect, dict):
        if "center" in rect:
            cx, cy = float(rect["center"][0]), float(rect["center"][1])
        else:
            cx = float(rect["x"]) + float(rect["w"]) / 2.0
            cy = float(rect["y"]) + float(rect["h"]) / 2.0
        w = float(rect.get("w", rect.get("width", 0.0)))
        h = float(rect.get("h", rect.get("height", 0.0)))
        angle = float(rect.get("angle", 0.0))
    else:
        raise ValueError(f"Unsupported rectangle descriptor type: {type(rect)}")

    hw = w / 2.0
    hh = h / 2.0

    # Local frame coordinate transform
    rel_ox = o[0] - cx
    rel_oy = o[1] - cy
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    ox_loc = rel_ox * cos_a + rel_oy * sin_a
    oy_loc = -rel_ox * sin_a + rel_oy * cos_a

    dx_loc = d[0] * cos_a + d[1] * sin_a
    dy_loc = -d[0] * sin_a + d[1] * cos_a

    eps = 1e-9

    # X slab
    if abs(dx_loc) > eps:
        tx1 = (-hw - ox_loc) / dx_loc
        tx2 = (hw - ox_loc) / dx_loc
        tmin_x = min(tx1, tx2)
        tmax_x = max(tx1, tx2)
    else:
        if ox_loc < -hw or ox_loc > hw:
            return float(max_range)
        tmin_x = -float("inf")
        tmax_x = float("inf")

    # Y slab
    if abs(dy_loc) > eps:
        ty1 = (-hh - oy_loc) / dy_loc
        ty2 = (hh - oy_loc) / dy_loc
        tmin_y = min(ty1, ty2)
        tmax_y = max(ty1, ty2)
    else:
        if oy_loc < -hh or oy_loc > hh:
            return float(max_range)
        tmin_y = -float("inf")
        tmax_y = float("inf")

    t_enter = max(tmin_x, tmin_y)
    t_exit = min(tmax_x, tmax_y)

    hit_eps = 1e-6
    if t_enter <= t_exit and t_exit > hit_eps:
        t = t_enter if t_enter > hit_eps else t_exit
        return float(min(t, max_range))
    return float(max_range)


def cast_ray_arena(
    origin: Union[Sequence[float], np.ndarray],
    direction: Union[Sequence[float], np.ndarray],
    width: float = ARENA_WIDTH,
    height: float = ARENA_HEIGHT,
    max_range: Optional[float] = None,
) -> float:
    """
    Compute intersection distance between a ray and the 4 arena boundary walls [0, width] x [0, height].

    Args:
        origin: 2D ray origin point [x, y].
        direction: 2D ray direction vector [dx, dy].
        width: Arena width in pixels (default 1200.0).
        height: Arena height in pixels (default 800.0).
        max_range: Optional maximum range limit.

    Returns:
        float: Distance to nearest intersecting boundary wall (clamped to max_range if provided).
    """
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)

    dir_norm = float(np.linalg.norm(d))
    if dir_norm < 1e-9:
        return float(max_range) if max_range is not None else float("inf")
    dx = d[0] / dir_norm
    dy = d[1] / dir_norm
    ox, oy = o[0], o[1]

    eps = 1e-9
    t_hits: List[float] = []

    # Left wall (x = 0)
    if dx < -eps:
        t = -ox / dx
        if t > 0:
            t_hits.append(t)
    # Right wall (x = width)
    elif dx > eps:
        t = (width - ox) / dx
        if t > 0:
            t_hits.append(t)

    # Top wall (y = 0)
    if dy < -eps:
        t = -oy / dy
        if t > 0:
            t_hits.append(t)
    # Bottom wall (y = height)
    elif dy > eps:
        t = (height - oy) / dy
        if t > 0:
            t_hits.append(t)

    if not t_hits:
        t_min = float("inf")
    else:
        t_min = min(t_hits)

    if max_range is not None:
        return float(min(t_min, max_range))
    return float(t_min)


def _extract_obstacle_geometry(obs: Any) -> Tuple[str, Any, np.ndarray]:
    """
    Extract geometry type ('circle' or 'rect'), geometrical parameters, and velocity vector from an obstacle.

    Args:
        obs: Obstacle instance, dictionary, or tuple descriptor.

    Returns:
        Tuple:
            - geom_type: 'circle' or 'rect'
            - params: (center, radius) for circle, or rect tuple/object for rect
            - velocity: np.ndarray shape (2,) of [vx, vy]
    """
    vel = np.array([0.0, 0.0], dtype=np.float64)
    if hasattr(obs, "velocity"):
        vel = np.asarray(obs.velocity, dtype=np.float64)
    elif hasattr(obs, "vx") and hasattr(obs, "vy"):
        vel = np.array([obs.vx, obs.vy], dtype=np.float64)
    elif isinstance(obs, dict) and "velocity" in obs:
        vel = np.asarray(obs["velocity"], dtype=np.float64)

    # Circle obstacle
    if hasattr(obs, "radius") or (isinstance(obs, dict) and obs.get("type") == "circle"):
        if hasattr(obs, "center"):
            c = obs.center
        elif hasattr(obs, "pos"):
            c = obs.pos
        elif hasattr(obs, "position"):
            c = obs.position
        else:
            c = obs["center"]
        r = obs.radius if hasattr(obs, "radius") else obs["radius"]
        return "circle", (np.asarray(c, dtype=np.float64), float(r)), vel

    # Rectangle obstacle
    if hasattr(obs, "rect") or (isinstance(obs, dict) and obs.get("type") == "rect"):
        rect_desc = obs.rect if hasattr(obs, "rect") else obs["rect"]
        return "rect", rect_desc, vel

    # Tuple / list descriptor: ('circle', (cx, cy), r, (vx, vy)) or ('rect', rect_desc, (vx, vy))
    if isinstance(obs, (tuple, list)) and len(obs) >= 2:
        if obs[0] == "circle":
            c = np.asarray(obs[1], dtype=np.float64)
            r = float(obs[2])
            v = np.asarray(obs[3], dtype=np.float64) if len(obs) > 3 else vel
            return "circle", (c, r), v
        elif obs[0] == "rect":
            rect_desc = obs[1]
            v = np.asarray(obs[2], dtype=np.float64) if len(obs) > 2 else vel
            return "rect", rect_desc, v

    raise ValueError(f"Unrecognized obstacle format: {type(obs)}: {obs}")


def cast_rays_all(
    origin: Union[Sequence[float], np.ndarray],
    angle: float,
    ray_offsets: Sequence[float] = RAY_ANGLES,
    obstacles: Optional[Sequence[Any]] = None,
    arena_size: Tuple[float, float] = (ARENA_WIDTH, ARENA_HEIGHT),
    max_range: float = DEFAULT_MAX_RANGE,
    drone_velocity: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Cast multiple acoustic rays against arena walls and obstacles.

    Conforms to PROJECT.md interface contract:
    - If drone_velocity is None: returns np.ndarray of shape (K,) with detected distances.
    - If drone_velocity is provided: returns (distances, closing_velocities).

    Args:
        origin: 2D drone position [x, y].
        angle: Drone heading orientation in radians.
        ray_offsets: Angular offsets relative to heading (defaults to RAY_ANGLES).
        obstacles: Optional list of obstacle objects, dicts, or tuples.
        arena_size: (width, height) of the arena.
        max_range: Maximum detection distance.
        drone_velocity: Optional 2D drone velocity vector [vx, vy].

    Returns:
        np.ndarray (K,) distances, OR Tuple[np.ndarray (K,), np.ndarray (K,)] of (distances, closing_vels).
    """
    orig = np.asarray(origin, dtype=np.float64)
    width, height = arena_size
    v_drone = np.asarray(drone_velocity, dtype=np.float64) if drone_velocity is not None else None

    distances: List[float] = []
    closing_vels: List[float] = []

    for offset in ray_offsets:
        ray_ang = angle + offset
        dir_vec = np.array([math.cos(ray_ang), math.sin(ray_ang)], dtype=np.float64)

        # 1. Cast against arena boundary walls (wall velocity is [0, 0])
        t_wall = cast_ray_arena(orig, dir_vec, width=width, height=height, max_range=max_range)
        t_min = t_wall
        hit_vel = np.array([0.0, 0.0], dtype=np.float64)

        # 2. Cast against all obstacles
        if obstacles:
            for obs in obstacles:
                obs_type, params, obs_vel = _extract_obstacle_geometry(obs)
                if obs_type == "circle":
                    c, r = params
                    t_obs = cast_ray_circle(orig, dir_vec, c, r, max_range)
                else:
                    t_obs = cast_ray_rect(orig, dir_vec, params, max_range)

                if t_obs < t_min:
                    t_min = t_obs
                    hit_vel = obs_vel

        distances.append(t_min)

        if v_drone is not None:
            if t_min < max_range:
                v_rel = v_drone - hit_vel
                vc = float(np.dot(v_rel, dir_vec))
            else:
                vc = 0.0
            closing_vels.append(vc)

    dist_arr = np.array(distances, dtype=np.float64)
    if v_drone is not None:
        return dist_arr, np.array(closing_vels, dtype=np.float64)
    return dist_arr


def cast_rays_with_doppler(
    origin: Union[Sequence[float], np.ndarray],
    angle: float,
    ray_offsets: Sequence[float],
    drone_velocity: Union[Sequence[float], np.ndarray],
    obstacles: Optional[Sequence[Any]] = None,
    arena_size: Tuple[float, float] = (ARENA_WIDTH, ARENA_HEIGHT),
    max_range: float = DEFAULT_MAX_RANGE,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Cast acoustic rays and compute Doppler closing velocity for all rays and frontal beam.

    Args:
        origin: 2D drone position [x, y].
        angle: Drone heading orientation in radians.
        ray_offsets: Angular offsets relative to heading.
        drone_velocity: 2D drone velocity vector [vx, vy].
        obstacles: Optional list of obstacles.
        arena_size: (width, height) of the arena.
        max_range: Maximum detection distance.

    Returns:
        Tuple:
            - distances: np.ndarray shape (K,) with hit distances.
            - closing_velocities: np.ndarray shape (K,) with closing velocities.
            - front_closing_velocity: float, closing velocity along the front ray.
    """
    dist_arr, closing_arr = cast_rays_all(
        origin=origin,
        angle=angle,
        ray_offsets=ray_offsets,
        obstacles=obstacles,
        arena_size=arena_size,
        max_range=max_range,
        drone_velocity=drone_velocity,
    )

    # Front ray is index 1 for standard 3-ray config [-pi/6, 0.0, pi/6]
    front_idx = 1 if len(ray_offsets) == 3 and abs(ray_offsets[1]) < 1e-4 else int(np.argmin(np.abs(ray_offsets)))
    front_closing = float(closing_arr[front_idx])

    return dist_arr, closing_arr, front_closing


def batch_cast_rays_all(
    origins: np.ndarray,
    angles: np.ndarray,
    velocities: np.ndarray,
    circle_centers: Optional[np.ndarray] = None,
    circle_radii: Optional[np.ndarray] = None,
    circle_velocities: Optional[np.ndarray] = None,
    rect_centers: Optional[np.ndarray] = None,
    rect_sizes: Optional[np.ndarray] = None,
    rect_angles: Optional[np.ndarray] = None,
    rect_velocities: Optional[np.ndarray] = None,
    arena_size: Tuple[float, float] = (ARENA_WIDTH, ARENA_HEIGHT),
    ray_offsets: Optional[Sequence[float]] = None,
    max_range: float = DEFAULT_MAX_RANGE,
    obstacles: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch acoustic ray-casting for N drones simultaneously.

    Args:
        origins: (N, 2) drone positions.
        angles: (N,) drone heading angles in radians.
        velocities: (N, 2) drone velocity vectors.
        circle_centers: Optional (Mc, 2) circle obstacle centers.
        circle_radii: Optional (Mc,) circle radii.
        circle_velocities: Optional (Mc, 2) circle velocities.
        rect_centers: Optional (Mr, 2) rectangle centers.
        rect_sizes: Optional (Mr, 2) rectangle [width, height].
        rect_angles: Optional (Mr,) rectangle rotation angles.
        rect_velocities: Optional (Mr, 2) rectangle velocities.
        arena_size: (width, height) of the arena.
        ray_offsets: (K,) ray angular offsets (defaults to RAY_ANGLES).
        max_range: Maximum detection distance.
        obstacles: Optional list of obstacles to unpack automatically if arrays are not passed.

    Returns:
        Tuple:
            - distances: (N, K) detected distances for each ray of each drone.
            - front_closing_vel: (N,) closing velocities along the front ray for each drone.
    """
    origins_arr = np.asarray(origins, dtype=np.float64)
    angles_arr = np.asarray(angles, dtype=np.float64)
    vels_arr = np.asarray(velocities, dtype=np.float64)

    N = origins_arr.shape[0]
    if ray_offsets is None:
        offsets = np.asarray(RAY_ANGLES, dtype=np.float64)
    else:
        offsets = np.asarray(ray_offsets, dtype=np.float64)
    K = len(offsets)
    width, height = arena_size

    # Ray direction unit vectors: shape (N, K, 2)
    ray_angles = angles_arr[:, None] + offsets[None, :]
    dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)

    # 1. Arena Walls: shape (N, K)
    px = origins_arr[:, 0:1]
    py = origins_arr[:, 1:2]
    dx = dirs[:, :, 0]
    dy = dirs[:, :, 1]

    safe_dx = np.where(np.abs(dx) > 1e-6, dx, 1.0)
    safe_dy = np.where(np.abs(dy) > 1e-6, dy, 1.0)
    t_left = np.where(dx < -1e-6, -px / safe_dx, np.inf)
    t_right = np.where(dx > 1e-6, (width - px) / safe_dx, np.inf)
    t_top = np.where(dy < -1e-6, -py / safe_dy, np.inf)
    t_bottom = np.where(dy > 1e-6, (height - py) / safe_dy, np.inf)

    t_min = np.minimum(np.minimum(t_left, t_right), np.minimum(t_top, t_bottom))
    t_min = np.minimum(t_min, max_range)

    # Hit velocity: shape (N, K, 2)
    hit_vel = np.zeros((N, K, 2), dtype=np.float64)

    # Unpack obstacles list if provided and explicit arrays are omitted
    if obstacles is not None and circle_centers is None and rect_centers is None:
        c_centers_list, c_radii_list, c_vels_list = [], [], []
        r_centers_list, r_sizes_list, r_angles_list, r_vels_list = [], [], [], []
        for obs in obstacles:
            obs_type, params, ovel = _extract_obstacle_geometry(obs)
            if obs_type == "circle":
                c, r = params
                c_centers_list.append(c)
                c_radii_list.append(r)
                c_vels_list.append(ovel)
            else:
                if len(params) == 4:
                    x, y, w, h = params
                    r_centers_list.append([x + w / 2.0, y + h / 2.0])
                    r_sizes_list.append([w, h])
                    r_angles_list.append(0.0)
                else:
                    cx, cy, w, h, a = params
                    r_centers_list.append([cx, cy])
                    r_sizes_list.append([w, h])
                    r_angles_list.append(a)
                r_vels_list.append(ovel)

        if c_centers_list:
            circle_centers = np.array(c_centers_list, dtype=np.float64)
            circle_radii = np.array(c_radii_list, dtype=np.float64)
            circle_velocities = np.array(c_vels_list, dtype=np.float64)
        if r_centers_list:
            rect_centers = np.array(r_centers_list, dtype=np.float64)
            rect_sizes = np.array(r_sizes_list, dtype=np.float64)
            rect_angles = np.array(r_angles_list, dtype=np.float64)
            rect_velocities = np.array(r_vels_list, dtype=np.float64)

    # 2. Circle Obstacles
    if circle_centers is not None and len(circle_centers) > 0:
        c_centers = np.asarray(circle_centers, dtype=np.float64)
        c_radii = np.asarray(circle_radii, dtype=np.float64)
        c_vels = np.asarray(circle_velocities, dtype=np.float64) if circle_velocities is not None else np.zeros((len(c_centers), 2))

        # m = origin - center: shape (N, Mc, 2)
        m = origins_arr[:, None, :] - c_centers[None, :, :]
        # b = dot(m, dirs): shape (N, K, Mc)
        b = np.einsum("ikd,ijd->ikj", dirs, m)
        # c_eq = ||m||^2 - R^2: shape (N, Mc)
        c_eq = np.sum(m**2, axis=-1) - (c_radii**2)[None, :]
        disc = b**2 - c_eq[:, None, :]

        sqrt_disc = np.sqrt(np.maximum(0.0, disc))
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        valid1 = (disc >= 0) & (t1 > 1e-6)
        valid2 = (disc >= 0) & (~valid1) & (t2 > 1e-6)
        t_circ = np.where(valid1, t1, np.where(valid2, t2, np.inf))

        min_circ_idx = np.argmin(t_circ, axis=-1)
        min_circ_dist = np.take_along_axis(t_circ, min_circ_idx[:, :, None], axis=-1).squeeze(-1)

        circle_hit_mask = min_circ_dist < t_min
        t_min = np.where(circle_hit_mask, min_circ_dist, t_min)
        chosen_c_vel = c_vels[min_circ_idx]
        hit_vel = np.where(circle_hit_mask[:, :, None], chosen_c_vel, hit_vel)

    # 3. Rectangle Obstacles
    if rect_centers is not None and len(rect_centers) > 0:
        r_centers = np.asarray(rect_centers, dtype=np.float64)
        r_sizes = np.asarray(rect_sizes, dtype=np.float64)
        r_angs = np.asarray(rect_angles, dtype=np.float64) if rect_angles is not None else np.zeros(len(r_centers))
        r_vels = np.asarray(rect_velocities, dtype=np.float64) if rect_velocities is not None else np.zeros((len(r_centers), 2))

        hw = r_sizes[:, 0] / 2.0
        hh = r_sizes[:, 1] / 2.0

        rel_o = origins_arr[:, None, :] - r_centers[None, :, :]
        cos_a = np.cos(r_angs)
        sin_a = np.sin(r_angs)

        ox_loc = rel_o[:, :, 0] * cos_a[None, :] + rel_o[:, :, 1] * sin_a[None, :]
        oy_loc = -rel_o[:, :, 0] * sin_a[None, :] + rel_o[:, :, 1] * cos_a[None, :]

        dx_loc = dx[:, :, None] * cos_a[None, None, :] + dy[:, :, None] * sin_a[None, None, :]
        dy_loc = -dx[:, :, None] * sin_a[None, None, :] + dy[:, :, None] * cos_a[None, None, :]

        eps = 1e-9
        nonzero_dx = np.abs(dx_loc) > eps
        safe_dx = np.where(nonzero_dx, dx_loc, 1.0)
        tx1 = (-hw[None, None, :] - ox_loc[:, None, :]) / safe_dx
        tx2 = (hw[None, None, :] - ox_loc[:, None, :]) / safe_dx
        tmin_x_calc = np.minimum(tx1, tx2)
        tmax_x_calc = np.maximum(tx1, tx2)
        inside_x = (ox_loc[:, None, :] >= -hw[None, None, :]) & (ox_loc[:, None, :] <= hw[None, None, :])
        tmin_x = np.where(nonzero_dx, tmin_x_calc, np.where(inside_x, -np.inf, np.inf))
        tmax_x = np.where(nonzero_dx, tmax_x_calc, np.where(inside_x, np.inf, -np.inf))

        nonzero_dy = np.abs(dy_loc) > eps
        safe_dy = np.where(nonzero_dy, dy_loc, 1.0)
        ty1 = (-hh[None, None, :] - oy_loc[:, None, :]) / safe_dy
        ty2 = (hh[None, None, :] - oy_loc[:, None, :]) / safe_dy
        tmin_y_calc = np.minimum(ty1, ty2)
        tmax_y_calc = np.maximum(ty1, ty2)
        inside_y = (oy_loc[:, None, :] >= -hh[None, None, :]) & (oy_loc[:, None, :] <= hh[None, None, :])
        tmin_y = np.where(nonzero_dy, tmin_y_calc, np.where(inside_y, -np.inf, np.inf))
        tmax_y = np.where(nonzero_dy, tmax_y_calc, np.where(inside_y, np.inf, -np.inf))

        t_enter = np.maximum(tmin_x, tmin_y)
        t_exit = np.minimum(tmax_x, tmax_y)

        hit_mask = (t_enter <= t_exit) & (t_exit > 1e-6)
        t_rect = np.where(hit_mask, np.where(t_enter > 1e-6, t_enter, t_exit), np.inf)

        min_rect_idx = np.argmin(t_rect, axis=-1)
        min_rect_dist = np.take_along_axis(t_rect, min_rect_idx[:, :, None], axis=-1).squeeze(-1)

        rect_hit_mask = min_rect_dist < t_min
        t_min = np.where(rect_hit_mask, min_rect_dist, t_min)
        chosen_r_vel = r_vels[min_rect_idx]
        hit_vel = np.where(rect_hit_mask[:, :, None], chosen_r_vel, hit_vel)

    # Clamping
    distances = np.minimum(t_min, max_range)

    # Doppler closing velocity along rays
    v_rel = vels_arr[:, None, :] - hit_vel
    v_closing = np.sum(v_rel * dirs, axis=-1)
    v_closing = np.where(distances < max_range - 1e-4, v_closing, 0.0)

    # Front ray closing velocity (index 1 for standard 3-ray config)
    front_idx = 1 if K == 3 and abs(offsets[1]) < 1e-4 else int(np.argmin(np.abs(offsets)))
    front_closing = v_closing[:, front_idx]

    return distances, front_closing


# ==============================================================================
# SENSOR NORMALIZATION FUNCTIONS
# ==============================================================================

def normalize_sensor_readings(
    left_dist: Union[float, np.ndarray],
    right_dist: Optional[Union[float, np.ndarray]] = None,
    front_dist: Optional[Union[float, np.ndarray]] = None,
    speed: Optional[Union[float, np.ndarray]] = None,
    closing_vel: Optional[Union[float, np.ndarray]] = None,
    max_range: float = DEFAULT_MAX_RANGE,
    max_speed: float = DEFAULT_MAX_SPEED,
) -> np.ndarray:
    """
    Normalize raw sensor readings into a standardized 5-element float64 vector or (N, 5) batch matrix.

    Features and Bounds:
        [0]: left_dist / max_range     in [0.0, 1.0]
        [1]: right_dist / max_range    in [0.0, 1.0]
        [2]: front_dist / max_range    in [0.0, 1.0]
        [3]: speed / max_speed         in [0.0, 1.0]
        [4]: closing_vel / max_speed   in [-1.0, 1.0]

    Can be invoked in two ways:
    1. By passing 5 features individually:
       `normalize_sensor_readings(left_d, right_d, front_d, speed, closing_v, max_range, max_speed)`
    2. By passing an already-assembled (5,) or (N, 5) array as the first argument:
       `normalize_sensor_readings(raw_array, max_range=..., max_speed=...)`

    Args:
        left_dist: Left acoustic sensor distance, or assembled array.
        right_dist: Right acoustic sensor distance.
        front_dist: Front acoustic sensor distance.
        speed: Current forward drone speed.
        closing_vel: Frontal closing velocity.
        max_range: Normalization distance upper bound (must be > 0.0).
        max_speed: Normalization speed upper bound (must be > 0.0).

    Returns:
        np.ndarray: 1D array of shape (5,) or 2D array of shape (N, 5) with dtype np.float64.

    Raises:
        ValueError: If max_range <= 0, max_speed <= 0, or missing arguments.
    """
    if max_range <= 0.0:
        raise ValueError(f"max_range must be positive (> 0.0), got {max_range}")
    if max_speed <= 0.0:
        raise ValueError(f"max_speed must be positive (> 0.0), got {max_speed}")

    # Check for pre-assembled array passed as single argument
    if right_dist is None and isinstance(left_dist, np.ndarray):
        return normalize_sensor_array(left_dist, max_range=max_range, max_speed=max_speed)

    if right_dist is None or front_dist is None or speed is None or closing_vel is None:
        raise ValueError("All 5 sensor reading arguments must be provided when not passing a pre-assembled array")

    is_scalar = all(
        np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
        for x in (left_dist, right_dist, front_dist, speed, closing_vel)
    )

    if is_scalar:
        nl = float(np.clip(float(left_dist) / max_range, 0.0, 1.0))
        nr = float(np.clip(float(right_dist) / max_range, 0.0, 1.0))
        nf = float(np.clip(float(front_dist) / max_range, 0.0, 1.0))
        ns = float(np.clip(float(speed) / max_speed, 0.0, 1.0))
        nc = float(np.clip(float(closing_vel) / max_speed, -1.0, 1.0))
        return np.array([nl, nr, nf, ns, nc], dtype=np.float64)

    # Vectorized batch processing
    arrs = [np.asarray(x, dtype=np.float64) for x in (left_dist, right_dist, front_dist, speed, closing_vel)]
    b_l, b_r, b_f, b_s, b_c = np.broadcast_arrays(*arrs)

    nl = np.clip(b_l / max_range, 0.0, 1.0)
    nr = np.clip(b_r / max_range, 0.0, 1.0)
    nf = np.clip(b_f / max_range, 0.0, 1.0)
    ns = np.clip(b_s / max_speed, 0.0, 1.0)
    nc = np.clip(b_c / max_speed, -1.0, 1.0)

    return np.stack([nl, nr, nf, ns, nc], axis=-1)


def normalize_sensor_array(
    raw_inputs: np.ndarray,
    max_range: float = DEFAULT_MAX_RANGE,
    max_speed: float = DEFAULT_MAX_SPEED,
) -> np.ndarray:
    """
    Normalize an already-assembled 1D array of shape (5,) or 2D batch array of shape (N, 5).

    Args:
        raw_inputs: Array of shape (5,) or (N, 5).
        max_range: Normalization distance upper bound.
        max_speed: Normalization speed upper bound.

    Returns:
        np.ndarray: Normalized array of identical shape (5,) or (N, 5), dtype np.float64.

    Raises:
        ValueError: If input shape does not match (5,) or (N, 5).
    """
    if max_range <= 0.0:
        raise ValueError(f"max_range must be positive (> 0.0), got {max_range}")
    if max_speed <= 0.0:
        raise ValueError(f"max_speed must be positive (> 0.0), got {max_speed}")

    arr = np.asarray(raw_inputs, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != SENSOR_INPUT_DIM:
            raise ValueError(f"Expected 1D array of shape ({SENSOR_INPUT_DIM},), got {arr.shape}")
        norm = np.empty(SENSOR_INPUT_DIM, dtype=np.float64)
        norm[0:3] = np.clip(arr[0:3] / max_range, 0.0, 1.0)
        norm[3] = np.clip(arr[3] / max_speed, 0.0, 1.0)
        norm[4] = np.clip(arr[4] / max_speed, -1.0, 1.0)
        return norm
    elif arr.ndim == 2:
        if arr.shape[1] != SENSOR_INPUT_DIM:
            raise ValueError(f"Expected 2D array of shape (N, {SENSOR_INPUT_DIM}), got {arr.shape}")
        norm = np.empty_like(arr, dtype=np.float64)
        norm[:, 0:3] = np.clip(arr[:, 0:3] / max_range, 0.0, 1.0)
        norm[:, 3] = np.clip(arr[:, 3] / max_speed, 0.0, 1.0)
        norm[:, 4] = np.clip(arr[:, 4] / max_speed, -1.0, 1.0)
        return norm
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array of shape {arr.shape}")


def denormalize_sensor_readings(
    normalized: np.ndarray,
    max_range: float = DEFAULT_MAX_RANGE,
    max_speed: float = DEFAULT_MAX_SPEED,
) -> np.ndarray:
    """
    Inverse denormalization converting normalized features back to physical units.

    Args:
        normalized: Normalized array of shape (5,) or (N, 5).
        max_range: Distance scaling factor.
        max_speed: Speed scaling factor.

    Returns:
        np.ndarray: Denormalized array in physical units.

    Raises:
        ValueError: If input shape does not match (5,) or (N, 5).
    """
    arr = np.asarray(normalized, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != SENSOR_INPUT_DIM:
            raise ValueError(f"Expected 1D array of shape ({SENSOR_INPUT_DIM},), got {arr.shape}")
        denorm = np.empty(SENSOR_INPUT_DIM, dtype=np.float64)
        denorm[0:3] = arr[0:3] * max_range
        denorm[3] = arr[3] * max_speed
        denorm[4] = arr[4] * max_speed
        return denorm
    elif arr.ndim == 2:
        if arr.shape[1] != SENSOR_INPUT_DIM:
            raise ValueError(f"Expected 2D array of shape (N, {SENSOR_INPUT_DIM}), got {arr.shape}")
        denorm = np.empty_like(arr, dtype=np.float64)
        denorm[:, 0:3] = arr[:, 0:3] * max_range
        denorm[:, 3] = arr[:, 3] * max_speed
        denorm[:, 4] = arr[:, 4] * max_speed
        return denorm
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array of shape {arr.shape}")
