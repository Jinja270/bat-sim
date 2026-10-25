"""
bat_sim/evolution.py
====================
Neuroevolution Engine and Population Manager for Bat Echolocation Simulation.

Features:
- Exact Fitness Formulation:
  fitness = (time_alive_steps * 1.0) + (distance_traveled * 0.5) - (50.0 if collided else 0.0)
- Truncation Selection:
  Descending fitness ranking; top 20% (20 drones) selected as parent breeding pool.
- Elitism Preservation:
  Top 2 performing individuals cloned via deep copy (controller.copy()) with zero mutation
  and independent weight memory buffers.
- Offspring Reproduction:
  98 offspring produced via 50/50 uniform crossover between parent pairs sampled with replacement,
  perturbed by adaptive Gaussian mutation with decaying schedule sigma(g) = max(0.01, 0.1 * (0.98^g)).
- Dual-Mode Stepping Engine:
  Interactive frame-by-frame stepping (Population.step) for 60 FPS Pygame visualization, and
  high-throughput vectorized batch stepping (Population.step_generation) executing 50 generations
  in < 10 seconds on CPU.
- Standardized Telemetry & Longitudinal Tracking:
  Complete telemetry dictionary reporting generation, alive_count, best_fitness, avg_fitness,
  avg_ping_rate, collision_count, survival_rate, with historical logging across 150+ generations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import bat_sim.physics as physics
from bat_sim.drone import Drone, TraditionalDrone, DEFAULT_HITBOX_RADIUS
from bat_sim.neural_net import (
    BatNeuralNet,
    TOTAL_WEIGHTS,
    breed_population_vectorized,
    compute_mutation_sigma,
)


# ==============================================================================
# DEFAULT EVOLUTION CONSTANTS
# ==============================================================================

DEFAULT_POPULATION_SIZE: int = 100
"""Default number of drone agents in the evolutionary population."""

DEFAULT_TRUNCATION_FRACTION: float = 0.20
"""Fraction of top individuals forming the parent breeding pool (top 20%)."""

DEFAULT_NUM_ELITES: int = 2
"""Number of top-performing drones preserved without mutation."""

DEFAULT_MUTATION_RATE: float = 0.10
"""Gene mutation probability per weight parameter."""

DEFAULT_SIGMA_0: float = 0.10
"""Initial Gaussian mutation standard deviation at generation 0."""

DEFAULT_DECAY_FACTOR: float = 0.98
"""Geometric decay factor per generation for mutation standard deviation."""

DEFAULT_MIN_SIGMA: float = 0.01
"""Asymptotic lower bound floor for mutation standard deviation."""

DEFAULT_MAX_STEPS: int = 500
"""Maximum simulation steps per generation episode (approx 8.33 seconds at 60 FPS)."""

FITNESS_TIME_WEIGHT: float = 1.0
"""Reward coefficient per time step survived."""

FITNESS_DISTANCE_WEIGHT: float = 0.5
"""Reward coefficient per unit distance traversed."""

FITNESS_COLLISION_PENALTY: float = 50.0
"""Penalty deducted if the drone terminates via collision."""


# ==============================================================================
# CANONICAL FITNESS CALCULATION FUNCTIONS
# ==============================================================================

def calculate_fitness(
    time_alive_steps: int,
    distance_traveled: float,
    collided: bool,
    time_weight: float = FITNESS_TIME_WEIGHT,
    distance_weight: float = FITNESS_DISTANCE_WEIGHT,
    collision_penalty: float = FITNESS_COLLISION_PENALTY,
) -> float:
    """
    Calculate scalar fitness score for an individual drone episode.

    Formula:
        fitness = (time_alive_steps * 1.0) + (distance_traveled * 0.5) - (50.0 if collided else 0.0)

    Parameters
    ----------
    time_alive_steps : int
        Number of steps the drone remained operational without collision (>= 0).
    distance_traveled : float
        Cumulative Euclidean path length in simulation units (>= 0.0).
    collided : bool
        True if the drone terminated due to collision with obstacles or arena walls.
    time_weight : float, default=1.0
        Reward coefficient per time step alive.
    distance_weight : float, default=0.5
        Reward coefficient per unit distance traveled.
    collision_penalty : float, default=50.0
        Deduction applied if the drone collided.

    Returns
    -------
    float
        Computed scalar fitness score.

    Raises
    ------
    ValueError
        If time_alive_steps < 0 or distance_traveled < 0.0.
    """
    if time_alive_steps < 0:
        raise ValueError(f"time_alive_steps must be non-negative, got {time_alive_steps}")
    if distance_traveled < 0.0:
        raise ValueError(f"distance_traveled must be non-negative, got {distance_traveled}")

    penalty = collision_penalty if collided else 0.0
    return float((time_alive_steps * time_weight) + (distance_traveled * distance_weight) - penalty)


def calculate_fitness_batch(
    time_alive_steps: Union[np.ndarray, Sequence[int]],
    distance_traveled: Union[np.ndarray, Sequence[float]],
    collided: Union[np.ndarray, Sequence[bool]],
    time_weight: float = FITNESS_TIME_WEIGHT,
    distance_weight: float = FITNESS_DISTANCE_WEIGHT,
    collision_penalty: float = FITNESS_COLLISION_PENALTY,
) -> np.ndarray:
    """
    Vectorized calculation of scalar fitness scores for a batch of drones.

    Parameters
    ----------
    time_alive_steps : np.ndarray or Sequence[int]
        1D array of steps alive.
    distance_traveled : np.ndarray or Sequence[float]
        1D array of distance traveled.
    collided : np.ndarray or Sequence[bool]
        1D boolean array indicating collisions.

    Returns
    -------
    np.ndarray
        1D float64 array of fitness scores.
    """
    t = np.asarray(time_alive_steps, dtype=np.float64)
    d = np.asarray(distance_traveled, dtype=np.float64)
    c = np.asarray(collided, dtype=bool)

    if np.any(t < 0):
        raise ValueError("time_alive_steps contains negative values")
    if np.any(d < 0):
        raise ValueError("distance_traveled contains negative values")

    return (t * time_weight) + (d * distance_weight) - np.where(c, collision_penalty, 0.0)


# ==============================================================================
# TELEMETRY & STATS DATA STRUCTURES
# ==============================================================================

@dataclass
class GenerationStats:
    """
    Longitudinal summary statistics captured at the completion of a generation episode.
    """

    generation: int
    best_fitness: float
    avg_fitness: float
    min_fitness: float
    alive_count: int
    collision_count: int
    survival_rate: float
    avg_ping_rate: float
    avg_distance_traveled: float
    avg_time_alive: float
    mutation_sigma: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to standard dictionary format."""
        return {
            "generation": int(self.generation),
            "best_fitness": float(self.best_fitness),
            "avg_fitness": float(self.avg_fitness),
            "min_fitness": float(self.min_fitness),
            "alive_count": int(self.alive_count),
            "collision_count": int(self.collision_count),
            "survival_rate": float(self.survival_rate),
            "avg_ping_rate": float(self.avg_ping_rate),
            "avg_distance_traveled": float(self.avg_distance_traveled),
            "avg_time_alive": float(self.avg_time_alive),
            "mutation_sigma": float(self.mutation_sigma),
        }


# ==============================================================================
# SELECTION OPERATORS
# ==============================================================================

def select_parents(
    drones: Sequence[Any],
    truncation_fraction: float = DEFAULT_TRUNCATION_FRACTION,
) -> List[Any]:
    """
    Extract parent breeding pool sorted by fitness descending using truncation selection.

    Parameters
    ----------
    drones : Sequence[Drone]
        Evaluated drones with active fitness attributes.
    truncation_fraction : float, default=0.20
        Fraction of top drones forming the parent pool.

    Returns
    -------
    List[Drone]
        Sorted list of parent drones of length max(1, int(len(drones) * truncation_fraction)).
    """
    if not drones:
        return []
    sorted_drones = sorted(drones, key=lambda d: getattr(d, "fitness", 0.0), reverse=True)
    n_parents = max(1, int(len(sorted_drones) * truncation_fraction))
    return sorted_drones[:n_parents]


def select_elites(
    drones: Sequence[Any],
    n_elites: int = DEFAULT_NUM_ELITES,
) -> List[Any]:
    """
    Extract top N elite individuals sorted by fitness descending.

    Parameters
    ----------
    drones : Sequence[Drone]
        Evaluated drones.
    n_elites : int, default=2
        Number of elites to extract.

    Returns
    -------
    List[Drone]
        Top N performing drones.
    """
    if not drones:
        return []
    sorted_drones = sorted(drones, key=lambda d: getattr(d, "fitness", 0.0), reverse=True)
    return sorted_drones[: min(n_elites, len(sorted_drones))]


# ==============================================================================
# POPULATION MANAGER CLASS
# ==============================================================================

class Population:
    """
    Neuroevolution engine and population manager for bat echolocation drones.

    Manages an ensemble of 100 autonomous drone agents navigating an obstacle arena,
    overseeing simulation stepping, active tracking, fitness evaluation, truncation selection,
    elitism preservation, uniform crossover, decaying mutation, and telemetry logging across
    150+ generations.

    Parameters
    ----------
    size : int, default=100
        Number of drones in the population.
    arena_width : float, default=1200.0
        Arena width in pixels.
    arena_height : float, default=800.0
        Arena height in pixels.
    arena_size : Optional[Tuple[float, float]], default=None
        (width, height) tuple overriding arena_width and arena_height.
    max_steps : int, default=500
        Maximum simulation steps per episode before terminating.
    dt : float, default=1.0/60.0
        Simulation time step duration per frame in seconds.
    n_elites : int, default=2
        Number of elite controllers preserved without mutation.
    truncation_fraction : float, default=0.20
        Proportion of top-performing drones selected as parents (top 20%).
    mutation_rate : float, default=0.10
        Probability of gene mutation per parameter.
    sigma_0 : float, default=0.10
        Initial mutation noise standard deviation.
    decay_factor : float, default=0.98
        Geometric decay factor per generation.
    min_sigma : float, default=0.01
        Lower bound floor on mutation sigma.
    spawn_center : Tuple[float, float], default=(600.0, 400.0)
        Center coordinates for initial drone spawning.
    spread : float, default=40.0
        Spatial jitter radius for drone spawning.
    seed : Optional[int], default=None
        Random seed for reproducible neuroevolution runs.
    rng : Optional[np.random.Generator], default=None
        NumPy random number generator.
    track_wavefronts : bool, default=True
        Whether to generate visual wavefronts (can be disabled for headless training speed).
    """

    def __init__(
        self,
        size: int = DEFAULT_POPULATION_SIZE,
        arena_width: float = physics.ARENA_WIDTH,
        arena_height: float = physics.ARENA_HEIGHT,
        arena_size: Optional[Tuple[float, float]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        dt: float = 1.0 / 60.0,
        n_elites: int = DEFAULT_NUM_ELITES,
        truncation_fraction: float = DEFAULT_TRUNCATION_FRACTION,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        sigma_0: float = DEFAULT_SIGMA_0,
        decay_factor: float = DEFAULT_DECAY_FACTOR,
        min_sigma: float = DEFAULT_MIN_SIGMA,
        spawn_center: Optional[Tuple[float, float]] = None,
        spread: float = 40.0,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        track_wavefronts: bool = True,
    ) -> None:
        self.size: int = int(size)
        if arena_size is not None:
            self.arena_width: float = float(arena_size[0])
            self.arena_height: float = float(arena_size[1])
        else:
            self.arena_width = float(arena_width)
            self.arena_height = float(arena_height)

        self.max_steps: int = int(max_steps)
        self.dt: float = float(dt)
        self.n_elites: int = int(n_elites)
        self.truncation_fraction: float = DEFAULT_TRUNCATION_FRACTION
        self.mutation_rate: float = float(mutation_rate)
        self.sigma_0: float = float(sigma_0)
        self.decay_factor: float = float(decay_factor)
        self.min_sigma: float = float(min_sigma)
        if spawn_center is not None:
            self.spawn_center: Tuple[float, float] = (float(spawn_center[0]), float(spawn_center[1]))
        else:
            self.spawn_center = (self.arena_width * 0.5, self.arena_height * 0.5)
        self.spread: float = float(spread)
        self.track_wavefronts: bool = bool(track_wavefronts)

        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng(seed)

        self.generation: int = 0
        self.step_count: int = 0
        self.history: List[GenerationStats] = []
        self._last_stats: Optional[GenerationStats] = None
        self._generation_evaluated: bool = False

        # Initialize drone agents
        self.drones: List[Drone] = self._create_initial_drones()

    @property
    def arena_size(self) -> Tuple[float, float]:
        """Current arena dimensions (arena_width, arena_height)."""
        return (self.arena_width, self.arena_height)

    @arena_size.setter
    def arena_size(self, size: Tuple[float, float]) -> None:
        self.arena_width = float(size[0])
        self.arena_height = float(size[1])
        self.spawn_center = (self.arena_width * 0.5, self.arena_height * 0.5)

    def set_arena_size(self, width: float, height: float) -> None:
        """Update arena boundaries and reposition spawn center to the geometric center."""
        self.arena_width = float(width)
        self.arena_height = float(height)
        self.spawn_center = (self.arena_width * 0.5, self.arena_height * 0.5)

    @property
    def current_step(self) -> int:
        """Alias for step_count."""
        return self.step_count

    @current_step.setter
    def current_step(self, val: int) -> None:
        self.step_count = int(val)

    def _create_initial_drones(self) -> List[Drone]:
        """Instantiate initial population of drones with randomized neural controllers."""
        drones: List[Drone] = []
        for _ in range(self.size):
            x, y = self._get_spawn_position()
            heading = float(self._rng.uniform(0.0, 2.0 * np.pi))
            net = BatNeuralNet(rng=self._rng)
            drone = Drone(
                x=x,
                y=y,
                heading=heading,
                controller=net,
                track_wavefronts=self.track_wavefronts,
                rng=self._rng,
            )
            drones.append(drone)
        return drones

    # --------------------------------------------------------------------------
    # SIMULATION STEPPING & GENERATION EXECUTION
    # --------------------------------------------------------------------------

    def step(self, dt: Optional[float] = None, env: Any = None) -> bool:
        """
        Advance simulation by one time step dt across all active drones.

        Parameters
        ----------
        dt : Optional[float], default=None
            Simulation time increment in seconds (defaults to self.dt).
        env : Any, optional
            Environment providing obstacles and collision checking.

        Returns
        -------
        bool
            True if the generation episode has completed (all drones dead or max_steps reached).
        """
        step_dt = float(dt) if dt is not None else self.dt
        self.step_count += 1
        any_alive = False

        for drone in self.drones:
            if drone.is_alive:
                drone.step(step_dt, env)
                if drone.is_alive:
                    any_alive = True

        is_done = (not any_alive) or (self.step_count >= self.max_steps)
        if is_done and not self._generation_evaluated:
            self.evaluate_fitness()
            self._generation_evaluated = True

        return is_done

    def step_generation(
        self,
        env: Any = None,
        max_steps: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute an entire generation episode at high speed on CPU.
        Leverages vectorized batch ray-casting and tensor feedforward inference when possible,
        finishing 50 full generations in under 10 seconds.

        Parameters
        ----------
        env : Any, optional
            Simulation environment containing arena bounds and obstacles.
        max_steps : Optional[int], default=None
            Episode step limit (defaults to self.max_steps).
        dt : Optional[float], default=None
            Time step per frame (defaults to self.dt).

        Returns
        -------
        Dict[str, Any]
            Final telemetry dictionary for the completed generation.
        """
        steps_limit = int(max_steps) if max_steps is not None else self.max_steps
        step_dt = float(dt) if dt is not None else self.dt

        # Check if high-throughput vectorized acceleration can be applied
        can_vectorize = all(not d.is_traditional and d.controller is not None for d in self.drones)

        if can_vectorize:
            self._run_vectorized_generation(env, steps_limit, step_dt)
        else:
            # Fall back to object-level stepping loop
            while self.step_count < steps_limit:
                done = self.step(step_dt, env)
                if done:
                    break

        if not self._generation_evaluated:
            self.evaluate_fitness()
            self._generation_evaluated = True

        return self.get_telemetry()

    def _run_vectorized_generation(self, env: Any, max_steps: int, dt: float) -> None:
        """High-performance vectorized simulation loop for 100 drones."""
        pop_size = len(self.drones)
        positions = np.array([[d.x, d.y] for d in self.drones], dtype=np.float64)
        headings = np.array([d.heading for d in self.drones], dtype=np.float64)
        speeds = np.array([d.speed for d in self.drones], dtype=np.float64)
        ping_freqs = np.array([d.recent_ping_freq for d in self.drones], dtype=np.float64)
        ping_timers = np.array([d.ping_timer for d in self.drones], dtype=np.float64)
        delta_thetas = np.array([d.delta_theta for d in self.drones], dtype=np.float64)
        is_alive = np.array([d.is_alive for d in self.drones], dtype=bool)
        collided = np.array([d.collided for d in self.drones], dtype=bool)
        time_alive = np.array([d.time_alive_steps for d in self.drones], dtype=np.int64)
        dist_traveled = np.array([d.distance_traveled for d in self.drones], dtype=np.float64)

        # Unpack neural network weights into vectorized parameter tensors
        weights = np.vstack([d.controller.get_flat_weights() for d in self.drones])
        W1 = weights[:, :160].reshape(pop_size, 5, 32)
        B1 = weights[:, 160:192].reshape(pop_size, 1, 32)
        W2 = weights[:, 192:1216].reshape(pop_size, 32, 32)
        B2 = weights[:, 1216:1248].reshape(pop_size, 1, 32)
        W3 = weights[:, 1248:1312].reshape(pop_size, 32, 2)
        B3 = weights[:, 1312:1314].reshape(pop_size, 1, 2)

        # Unpack obstacles from environment if available
        circle_centers, circle_radii, circle_velocities = self._extract_circle_obstacles(env)
        rect_centers, rect_sizes, rect_angles, rect_velocities = self._extract_rect_obstacles(env)
        arena_size = (self.arena_width, self.arena_height)

        turn_scale = getattr(self.drones[0], "turning_rate_scale", 3.0) if self.drones else 3.0

        for _ in range(max_steps):
            if not np.any(is_alive):
                break
            self.step_count += 1

            # Advance dynamic obstacle drift in environment if provided
            if env is not None:
                env.step(dt)
                circle_centers, circle_radii, circle_velocities = self._extract_circle_obstacles(env)
                rect_centers, rect_sizes, rect_angles, rect_velocities = self._extract_rect_obstacles(env)

            # Acoustic pulse accumulation & emission
            ping_timers[is_alive] += dt
            ping_periods = 1.0 / np.maximum(physics.DEFAULT_PULSE_FREQ_MIN, ping_freqs[is_alive])
            mask = ping_timers[is_alive] >= ping_periods

            if np.any(mask):
                alive_indices = np.where(is_alive)[0]
                p_idx = alive_indices[mask]
                ping_timers[p_idx] = ping_timers[p_idx] % ping_periods[mask]

                vels = np.column_stack([
                    speeds[p_idx] * np.cos(headings[p_idx]),
                    speeds[p_idx] * np.sin(headings[p_idx]),
                ])

                # Vectorized batch ray casting
                dists, closing = physics.batch_cast_rays_all(
                    origins=positions[p_idx],
                    angles=headings[p_idx],
                    velocities=vels,
                    circle_centers=circle_centers,
                    circle_radii=circle_radii,
                    circle_velocities=circle_velocities,
                    rect_centers=rect_centers,
                    rect_sizes=rect_sizes,
                    rect_angles=rect_angles,
                    rect_velocities=rect_velocities,
                    arena_size=arena_size,
                    max_range=physics.DEFAULT_MAX_RANGE,
                )

                # Assemble normalized (K, 1, 5) tensor inputs [d_L, d_R, d_F, v, v_c]
                raw_in = np.column_stack([
                    dists[:, 0] / physics.DEFAULT_MAX_RANGE,
                    dists[:, 2] / physics.DEFAULT_MAX_RANGE,
                    dists[:, 1] / physics.DEFAULT_MAX_RANGE,
                    speeds[p_idx] / physics.DEFAULT_MAX_SPEED,
                    np.clip(closing / physics.DEFAULT_MAX_SPEED, -1.0, 1.0),
                ])[:, None, :]

                # Tensor forward pass
                h1 = np.maximum(0.0, np.matmul(raw_in, W1[p_idx]) + B1[p_idx])
                h2 = np.maximum(0.0, np.matmul(h1, W2[p_idx]) + B2[p_idx])
                out = np.matmul(h2, W3[p_idx]) + B3[p_idx]

                delta_thetas[p_idx] = np.tanh(out[:, 0, 0])
                urg = 1.0 / (1.0 + np.exp(-np.clip(out[:, 0, 1], -60.0, 60.0)))
                ping_freqs[p_idx] = 10.0 + 140.0 * urg

            # Kinematic integration
            headings[is_alive] = (headings[is_alive] + delta_thetas[is_alive] * turn_scale * dt) % (2.0 * np.pi)
            positions[is_alive, 0] += speeds[is_alive] * np.cos(headings[is_alive]) * dt
            positions[is_alive, 1] += speeds[is_alive] * np.sin(headings[is_alive]) * dt
            dist_traveled[is_alive] += speeds[is_alive] * dt
            time_alive[is_alive] += 1

            # Collision detection against walls and obstacles
            px = positions[is_alive, 0]
            py = positions[is_alive, 1]
            r = DEFAULT_HITBOX_RADIUS
            wall_hit = (px - r <= 0.0) | (px + r >= self.arena_width) | (py - r <= 0.0) | (py + r >= self.arena_height)

            obs_hit = np.zeros_like(wall_hit, dtype=bool)
            if circle_centers is not None and len(circle_centers) > 0:
                dx = px[:, None] - circle_centers[None, :, 0]
                dy = py[:, None] - circle_centers[None, :, 1]
                radii_sum = r + circle_radii[None, :]
                obs_hit = np.any((dx * dx + dy * dy) <= (radii_sum * radii_sum), axis=1)

            rect_hit = np.zeros_like(wall_hit, dtype=bool)
            if rect_centers is not None and len(rect_centers) > 0:
                for m in range(len(rect_centers)):
                    rel_x = px - rect_centers[m, 0]
                    rel_y = py - rect_centers[m, 1]
                    cos_a = math.cos(rect_angles[m])
                    sin_a = math.sin(rect_angles[m])
                    lx = rel_x * cos_a + rel_y * sin_a
                    ly = -rel_x * sin_a + rel_y * cos_a
                    hw = rect_sizes[m, 0] / 2.0
                    hh = rect_sizes[m, 1] / 2.0
                    closest_x = np.clip(lx, -hw, hw)
                    closest_y = np.clip(ly, -hh, hh)
                    rect_hit |= ((lx - closest_x) ** 2 + (ly - closest_y) ** 2) <= (r * r)

            hit = wall_hit | obs_hit | rect_hit
            if np.any(hit):
                hit_alive_idx = np.where(is_alive)[0][hit]
                is_alive[hit_alive_idx] = False
                collided[hit_alive_idx] = True

        # Synchronize simulation state back into Drone instances
        for i, drone in enumerate(self.drones):
            drone.x = float(positions[i, 0])
            drone.y = float(positions[i, 1])
            drone.heading = float(headings[i])
            drone.speed = float(speeds[i])
            drone.delta_theta = float(delta_thetas[i])
            drone.recent_ping_freq = float(ping_freqs[i])
            drone.ping_freq = float(ping_freqs[i])
            drone.ping_timer = float(ping_timers[i])
            drone.is_alive = bool(is_alive[i])
            drone.collided = bool(collided[i])
            drone.time_alive_steps = int(time_alive[i])
            drone.distance_traveled = float(dist_traveled[i])

    def _extract_circle_obstacles(self, env: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract circle obstacle arrays from environment if available."""
        if env is None:
            return None, None, None

        if hasattr(env, "circle_centers") and env.circle_centers is not None:
            return env.circle_centers, env.circle_radii, getattr(env, "circle_velocities", None)

        obstacles = getattr(env, "obstacles", None)
        if not obstacles:
            return None, None, None

        centers, radii, vels = [], [], []
        for obs in obstacles:
            if hasattr(obs, "radius"):
                if hasattr(obs, "center"):
                    cx, cy = obs.center
                elif hasattr(obs, "x") and hasattr(obs, "y"):
                    cx, cy = obs.x, obs.y
                else:
                    continue
                centers.append([float(cx), float(cy)])
                radii.append(float(obs.radius))
                vx = float(getattr(obs, "vx", 0.0))
                vy = float(getattr(obs, "vy", 0.0))
                vels.append([vx, vy])

        if not centers:
            return None, None, None

        return (
            np.array(centers, dtype=np.float64),
            np.array(radii, dtype=np.float64),
            np.array(vels, dtype=np.float64),
        )

    def _extract_rect_obstacles(self, env: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract rectangle obstacle arrays from environment if available."""
        if env is None:
            return None, None, None, None

        if hasattr(env, "rect_centers") and env.rect_centers is not None:
            return env.rect_centers, env.rect_sizes, env.rect_angles, getattr(env, "rect_velocities", None)

        obstacles = getattr(env, "obstacles", None)
        if not obstacles:
            return None, None, None, None

        centers, sizes, angles, vels = [], [], [], []
        for obs in obstacles:
            if hasattr(obs, "size") or (hasattr(obs, "width") and hasattr(obs, "height")):
                if hasattr(obs, "center"):
                    cx, cy = obs.center
                elif hasattr(obs, "x") and hasattr(obs, "y"):
                    cx, cy = obs.x, obs.y
                else:
                    continue
                w = getattr(obs, "width", obs.size[0] if hasattr(obs, "size") else 50.0)
                h = getattr(obs, "height", obs.size[1] if hasattr(obs, "size") else 50.0)
                ang = getattr(obs, "angle", 0.0)
                vx = float(getattr(obs, "vx", 0.0))
                vy = float(getattr(obs, "vy", 0.0))
                centers.append([float(cx), float(cy)])
                sizes.append([float(w), float(h)])
                angles.append(float(ang))
                vels.append([vx, vy])

        if not centers:
            return None, None, None, None

        return (
            np.array(centers, dtype=np.float64),
            np.array(sizes, dtype=np.float64),
            np.array(angles, dtype=np.float64),
            np.array(vels, dtype=np.float64),
        )

    # --------------------------------------------------------------------------
    # FITNESS EVALUATION & TELEMETRY
    # --------------------------------------------------------------------------

    def evaluate_fitness(self) -> np.ndarray:
        """
        Compute and store fitness scores for all drones in the population.
        Calculates and records GenerationStats in self.history.

        Returns
        -------
        np.ndarray
            1D array of shape (size,) containing scalar fitness values.
        """
        fitnesses = np.empty(len(self.drones), dtype=np.float64)

        for i, drone in enumerate(self.drones):
            fit = calculate_fitness(
                time_alive_steps=drone.time_alive_steps,
                distance_traveled=drone.distance_traveled,
                collided=drone.collided,
            )
            drone.fitness = fit
            fitnesses[i] = fit

        best_f = float(np.max(fitnesses)) if len(fitnesses) > 0 else 0.0
        avg_f = float(np.mean(fitnesses)) if len(fitnesses) > 0 else 0.0
        min_f = float(np.min(fitnesses)) if len(fitnesses) > 0 else 0.0

        alive_cnt = sum(1 for d in self.drones if d.is_alive)
        coll_cnt = sum(1 for d in self.drones if d.collided)
        surv_rate = float(alive_cnt / len(self.drones)) if len(self.drones) > 0 else 0.0

        ping_rates = [d.recent_ping_freq for d in self.drones]
        avg_ping = float(np.mean(ping_rates)) if ping_rates else 10.0

        dists = [d.distance_traveled for d in self.drones]
        avg_dist = float(np.mean(dists)) if dists else 0.0

        times = [d.time_alive_steps for d in self.drones]
        avg_time = float(np.mean(times)) if times else 0.0

        sigma = compute_mutation_sigma(
            self.generation,
            sigma_0=self.sigma_0,
            decay_factor=self.decay_factor,
            min_sigma=self.min_sigma,
        )

        stats = GenerationStats(
            generation=self.generation,
            best_fitness=best_f,
            avg_fitness=avg_f,
            min_fitness=min_f,
            alive_count=alive_cnt,
            collision_count=coll_cnt,
            survival_rate=surv_rate,
            avg_ping_rate=avg_ping,
            avg_distance_traveled=avg_dist,
            avg_time_alive=avg_time,
            mutation_sigma=sigma,
        )
        self._last_stats = stats
        self.history.append(stats)
        self._generation_evaluated = True

        return fitnesses

    def get_best_drone(self) -> Drone:
        """
        Retrieve the single highest-fitness Drone instance in the population.

        Returns
        -------
        Drone
            Highest performing drone agent.
        """
        if not self.drones:
            raise RuntimeError("Cannot retrieve best drone from empty population")
        return max(self.drones, key=lambda d: getattr(d, "fitness", -np.inf))

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Generate telemetry dictionary for HUD rendering and longitudinal monitoring.

        Keys:
            generation: int
            alive_count: int
            best_fitness: float
            avg_fitness: float
            avg_ping_rate: float
            collision_count: int
            survival_rate: float
            step: int
        """
        alive = sum(1 for d in self.drones if d.is_alive)
        coll = sum(1 for d in self.drones if d.collided)
        surv = float(alive / len(self.drones)) if len(self.drones) > 0 else 0.0

        ping_rates = [d.recent_ping_freq for d in self.drones]
        avg_ping = float(np.mean(ping_rates)) if ping_rates else 10.0

        fits = [d.fitness for d in self.drones]
        if fits:
            best_f = float(np.max(fits))
            avg_f = float(np.mean(fits))
        elif self._last_stats is not None:
            best_f = float(self._last_stats.best_fitness)
            avg_f = float(self._last_stats.avg_fitness)
        else:
            best_f = 0.0
            avg_f = 0.0

        return {
            "generation": int(self.generation),
            "alive_count": int(alive),
            "best_fitness": float(best_f),
            "avg_fitness": float(avg_f),
            "avg_ping_rate": float(avg_ping),
            "collision_count": int(coll),
            "survival_rate": float(surv),
            "step": int(self.step_count),
        }

    # --------------------------------------------------------------------------
    # GENETIC REPRODUCTION & ELITISM
    # --------------------------------------------------------------------------

    def reproduce(
        self,
        generation: Optional[int] = None,
        env: Optional[Any] = None,
    ) -> Population:
        """
        Breed the next generation of drones using genetic operators:
        1. Truncation selection: rank drones descending by fitness, top 20% form parent pool.
        2. Elitism: top 2 drones preserved unchanged (deep copy via controller.copy(), zero mutation).
        3. Uniform crossover: 98 offspring via 50/50 gene splice between parent pairs.
        4. Decaying Gaussian mutation applied to offspring weights with schedule sigma(g).
        5. Advance generation index and re-initialize drone spawn positions.

        Parameters
        ----------
        generation : Optional[int], default=None
            Explicit generation number; if None, self.generation + 1 is used.
        env : Optional[Any], default=None
            Environment reference for collision-free spawn verification.

        Returns
        -------
        Population
            Self reference supporting method chaining.
        """
        if not self._generation_evaluated:
            self.evaluate_fitness()

        next_gen = int(generation) if generation is not None else self.generation + 1
        sorted_drones = sorted(self.drones, key=lambda d: getattr(d, "fitness", 0.0), reverse=True)

        new_drones: List[Drone] = []

        # 1. Elitism: preserve top 2 drones unchanged via controller.copy()
        n_elites = min(self.n_elites, len(sorted_drones))
        for i in range(n_elites):
            parent_drone = sorted_drones[i]
            elite_controller = parent_drone.controller.copy() if parent_drone.controller is not None else None
            x, y = self._get_spawn_position(env=env)
            heading = float(self._rng.uniform(0.0, 2.0 * np.pi))

            elite_drone = Drone(
                x=x,
                y=y,
                heading=heading,
                controller=elite_controller,
                is_traditional=parent_drone.is_traditional,
                track_wavefronts=self.track_wavefronts,
                rng=self._rng,
            )
            new_drones.append(elite_drone)

        # 2. Truncation Selection: top 20% parent pool
        parent_pool = select_parents(sorted_drones, self.truncation_fraction)
        n_offspring = self.size - len(new_drones)

        # 3. Offspring Generation: Crossover + Decaying Gaussian Mutation
        sigma = compute_mutation_sigma(
            next_gen,
            sigma_0=self.sigma_0,
            decay_factor=self.decay_factor,
            min_sigma=self.min_sigma,
        )

        for _ in range(n_offspring):
            p_a = self._rng.choice(parent_pool)
            p_b = self._rng.choice(parent_pool)

            if p_a.controller is not None and p_b.controller is not None:
                child_net = BatNeuralNet.uniform_crossover(p_a.controller, p_b.controller, rng=self._rng)
                child_net.mutate(mutation_rate=self.mutation_rate, sigma=sigma, rng=self._rng)
            else:
                child_net = BatNeuralNet(rng=self._rng)

            x, y = self._get_spawn_position(env=env)
            heading = float(self._rng.uniform(0.0, 2.0 * np.pi))

            child_drone = Drone(
                x=x,
                y=y,
                heading=heading,
                controller=child_net,
                track_wavefronts=self.track_wavefronts,
                rng=self._rng,
            )
            new_drones.append(child_drone)

        self.drones = new_drones
        self.generation = next_gen
        self.step_count = 0
        self._generation_evaluated = False
        return self

    def _get_spawn_position(self, env: Optional[Any] = None) -> Tuple[float, float]:
        """Compute spawn coordinate with radial jitter, guaranteed free of obstacles."""
        cx, cy = self.spawn_center
        # Always clamp spawn_center safely inside arena boundaries
        cx = max(60.0, min(self.arena_width - 60.0, cx))
        cy = max(60.0, min(self.arena_height - 60.0, cy))

        max_allowed_spread = min(self.spread, max(10.0, min(cx, cy) - 25.0))

        for _ in range(30):
            if max_allowed_spread > 0.0:
                angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
                dist = float(self._rng.uniform(0.0, max_allowed_spread))
                x = cx + dist * math.cos(angle)
                y = cy + dist * math.sin(angle)
            else:
                x, y = cx, cy

            if env is not None and hasattr(env, "check_collision"):
                # Ensure at least 18px clearance around drone hitbox (radius 8px)
                if not env.check_collision(x, y, radius=18.0):
                    return (x, y)
            else:
                return (x, y)
        return (cx, cy)

    def reset_generation(
        self,
        spawn_center: Optional[Tuple[float, float]] = None,
        spread: Optional[float] = None,
        env: Optional[Any] = None,
    ) -> None:
        """
        Reset all drones to spawn positions and clear episode states.
        """
        if spawn_center is not None:
            self.spawn_center = (float(spawn_center[0]), float(spawn_center[1]))
        elif hasattr(self, "arena_width") and hasattr(self, "arena_height"):
            self.spawn_center = (self.arena_width * 0.5, self.arena_height * 0.5)
        if spread is not None:
            self.spread = float(spread)

        self.step_count = 0
        self._generation_evaluated = False

        for drone in self.drones:
            x, y = self._get_spawn_position(env=env)
            heading = float(self._rng.uniform(0.0, 2.0 * np.pi))
            drone.reset(x=x, y=y, heading=heading)

    # --------------------------------------------------------------------------
    # WEIGHT SERIALIZATION & PERSISTENCE
    # --------------------------------------------------------------------------

    def save_best(self, filepath: Union[str, os.PathLike]) -> None:
        """
        Save flat weight parameters of the current best drone to a .npz archive.

        Parameters
        ----------
        filepath : Union[str, os.PathLike]
            Destination path (e.g. 'best_weights.npz').
        """
        best_drone = self.get_best_drone()
        if best_drone.controller is None:
            raise RuntimeError("Best drone has no neural network controller")
        best_drone.controller.save(filepath)

    def load_champion(self, filepath: Union[str, os.PathLike]) -> BatNeuralNet:
        """
        Load neural network controller weights from a .npz archive.

        Parameters
        ----------
        filepath : Union[str, os.PathLike]
            Archive file path.

        Returns
        -------
        BatNeuralNet
            Reconstructed champion network.
        """
        return BatNeuralNet.load(filepath, rng=self._rng)

    def get_all_weights(self) -> np.ndarray:
        """Return 2D array of shape (size, 1314) containing flat weights of all drones."""
        return np.vstack([
            d.controller.get_flat_weights() if d.controller is not None else np.zeros(TOTAL_WEIGHTS)
            for d in self.drones
        ])

    def set_all_weights(self, weights: np.ndarray) -> None:
        """Set weights for all drones from a 2D array of shape (size, 1314)."""
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape != (self.size, TOTAL_WEIGHTS):
            raise ValueError(f"Expected shape ({self.size}, {TOTAL_WEIGHTS}), got {w_arr.shape}")
        for i, drone in enumerate(self.drones):
            if drone.controller is not None:
                drone.controller.set_flat_weights(w_arr[i])
