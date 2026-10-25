"""
bat_sim/drone.py
================
Autonomous Drone Agent & Bio-Acoustic Perception Engine for Bat Echolocation Simulation.

Features:
- Wavefront: Expanding circular acoustic wave arcs with smooth color transition from
  Emerald Green (10 Hz search phase) through Amber (80 Hz) to Crimson Red (150 Hz terminal buzz)
  and distance-based opacity decay.
- Drone: 2D kinematic model with continuous spatial integration, bat-inspired dynamic
  pulse-rate modulation (terminal buzz) triggered by neural urgency, multi-ray echolocation
  (Left -30°, Front 0°, Right +30°) coupled with Doppler shift detection, dead-reckoning flight,
  and exact fitness tracking.
- TraditionalDrone: Baseline obstacle avoidance drone operating at a fixed 10 Hz pulse rate
  with heuristic threshold turning logic for comparative performance evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import bat_sim.physics as physics
from bat_sim.neural_net import BatNeuralNet


# ==============================================================================
# COLOR & ACOUSTIC MAPPING CONSTANTS
# ==============================================================================

COLOR_SEARCH_GREEN: Tuple[int, int, int] = (46, 204, 113)
"""Emerald green emitted during low-urgency 10 Hz search pulses."""

COLOR_APPROACH_AMBER: Tuple[int, int, int] = (241, 196, 15)
"""Warning amber emitted during mid-urgency 80 Hz approach pulses."""

COLOR_TERMINAL_RED: Tuple[int, int, int] = (231, 76, 60)
"""Crimson red emitted during high-urgency 150 Hz terminal buzz pulses."""

DEFAULT_TURNING_RATE_SCALE: float = 3.0
"""Maximum angular turning velocity scale in radians/second (approx 171.9°/s)."""

DEFAULT_HITBOX_RADIUS: float = 8.0
"""Physical drone collision hitbox radius in pixels/units."""

DEFAULT_SPEED: float = 60.0
"""Default forward cruise speed in units/s."""

DEFAULT_MAX_SPEED: float = 100.0
"""Maximum linear forward speed in units/s."""

DEFAULT_FRONT_THRESHOLD: float = 70.0
"""Forward detection threshold for traditional baseline drone avoidance in pixels."""

DEFAULT_SIDE_THRESHOLD: float = 0.0
"""Lateral detection threshold for traditional baseline drone avoidance in pixels."""


def urgency_to_color(urgency: float) -> Tuple[int, int, int]:
    """
    Interpolate RGB color from Emerald Green (urgency 0.0, 10 Hz) through Amber (0.5)
    to Crimson Red (urgency 1.0, 150 Hz).

    Parameters
    ----------
    urgency : float
        Neural urgency lambda in [0.0, 1.0].

    Returns
    -------
    Tuple[int, int, int]
        RGB color tuple with integer components in [0, 255].
    """
    u = float(np.clip(urgency, 0.0, 1.0))
    if u <= 0.5:
        t = u / 0.5
        r = int(round(46.0 + t * (241.0 - 46.0)))
        g = int(round(204.0 + t * (196.0 - 204.0)))
        b = int(round(113.0 + t * (15.0 - 113.0)))
    else:
        t = (u - 0.5) / 0.5
        r = int(round(241.0 + t * (231.0 - 241.0)))
        g = int(round(196.0 + t * (76.0 - 196.0)))
        b = int(round(15.0 + t * (60.0 - 15.0)))

    return (
        int(np.clip(r, 0, 255)),
        int(np.clip(g, 0, 255)),
        int(np.clip(b, 0, 255)),
    )


# ==============================================================================
# ACOUSTIC WAVEFRONT CLASS
# ==============================================================================

@dataclass
class Wavefront:
    """
    Expanding acoustic wavefront emitted during an echolocation ping.

    Attributes
    ----------
    x : float
        Horizontal emission origin coordinate in pixels.
    y : float
        Vertical emission origin coordinate in pixels.
    heading : float
        Heading orientation angle in radians at emission.
    radius : float, default=0.0
        Current wavefront radius in pixels.
    max_radius : float, default=300.0
        Maximum propagation range before wavefront dissolves.
    speed : float, default=343.0
        Radial sound propagation velocity in pixels/second.
    frequency : float, default=10.0
        Acoustic pulse emission frequency in Hz.
    urgency : float, default=0.0
        Urgency lambda in [0.0, 1.0] defining color shift.
    age : float, default=0.0
        Elapsed time in seconds since emission.
    color : Optional[Tuple[int, int, int]], default=None
        Explicit RGB color tuple; if None, computed from urgency.
    arc_angle : float, default=2*pi/3 (120°)
        Angular arc spread in radians.
    """

    x: float
    y: float
    heading: float = 0.0
    radius: float = 0.0
    max_radius: float = physics.DEFAULT_MAX_RANGE
    speed: float = physics.SPEED_OF_SOUND
    frequency: float = physics.DEFAULT_PULSE_FREQ_MIN
    urgency: float = 0.0
    age: float = 0.0
    color: Optional[Tuple[int, int, int]] = None
    arc_angle: float = math.pi * 2.0 / 3.0

    def __post_init__(self) -> None:
        if self.color is None:
            self.color = urgency_to_color(self.urgency)

    @property
    def origin(self) -> Tuple[float, float]:
        """Origin coordinates (x, y) as a tuple."""
        return (self.x, self.y)

    @property
    def max_range(self) -> float:
        """Alias for max_radius."""
        return self.max_radius

    @max_range.setter
    def max_range(self, val: float) -> None:
        self.max_radius = float(val)

    @property
    def is_alive(self) -> bool:
        """True if the wavefront is still propagating within its maximum radius."""
        return self.radius < self.max_radius

    def update(self, dt: float) -> bool:
        """
        Advance wavefront radius and age by dt seconds.

        Parameters
        ----------
        dt : float
            Simulation time step in seconds.

        Returns
        -------
        bool
            True if the wavefront is still alive, False if expired.
        """
        self.age += dt
        self.radius += self.speed * dt
        return self.is_alive

    def get_color(self) -> Tuple[int, int, int]:
        """Return the RGB color tuple of this wavefront."""
        if self.color is not None:
            return self.color
        return urgency_to_color(self.urgency)

    def get_alpha(self) -> float:
        """
        Calculate normalized opacity in [0.0, 1.0] decaying linearly with radius.
        """
        if self.max_radius <= 0.0:
            return 0.0
        return float(max(0.0, 1.0 - (self.radius / self.max_radius)))

    def draw(self, surface: Any) -> None:
        """
        Render wavefront arc onto a Pygame surface.
        Gracefully handles headless execution or uninitialized display.
        """
        if surface is None or not self.is_alive or self.radius <= 0.5:
            return

        try:
            import pygame  # Lazy import for headless testing safety

            alpha = int(self.get_alpha() * 255.0)
            if alpha <= 0:
                return

            color = self.get_color()
            r = int(round(self.radius))
            cx = int(round(self.x))
            cy = int(round(self.y))

            # Full circle drawing for simplicity and high performance
            rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
            # Clip rect against surface dimensions
            if rect.colliderect(surface.get_rect()):
                start_angle = self.heading - (self.arc_angle / 2.0)
                end_angle = self.heading + (self.arc_angle / 2.0)
                # Pygame arc angles are measured counter-clockwise from positive x-axis
                pygame.draw.arc(surface, color, rect, -end_angle, -start_angle, width=max(1, int(round(2.0 * (alpha / 255.0)))))
        except Exception:
            # Defensive guard against headless environments without display
            pass


_DEFAULT_CONTROLLER = object()


# ==============================================================================
# DRONE AGENT CLASS
# ==============================================================================

class Drone:
    """
    Autonomous echolocating drone agent powered by BatNeuralNet or baseline heuristic.

    Parameters
    ----------
    x : float, default=600.0
        Initial x position in pixels.
    y : float, default=400.0
        Initial y position in pixels.
    heading : float, default=0.0
        Initial heading angle in radians [0, 2*pi).
    controller : Optional[BatNeuralNet], default=None
        Neural network controller; if omitted, instantiates a new BatNeuralNet.
        If explicitly set to None, operates without a neural network controller.
    is_traditional : bool, default=False
        If True, operates as a traditional baseline drone with fixed 10 Hz pulse rate.
    speed : float, default=5.0
        Forward cruise speed in units/s.
    max_speed : float, default=10.0
        Maximum linear forward speed in units/s.
    turning_rate_scale : float, default=3.0
        Maximum turning rate scale in rad/s.
    radius : float, default=8.0
        Hitbox radius for collision detection in pixels.
    max_range : float, default=300.0
        Maximum acoustic ray perception range in pixels.
    track_wavefronts : bool, default=True
        Whether to generate and maintain visual Wavefront objects (set False in headless training).
    front_threshold : float, default=100.0
        Forward obstacle distance triggering turn in traditional drone.
    side_threshold : float, default=50.0
        Lateral obstacle distance triggering turn in traditional drone.
    rng : Optional[np.random.Generator], default=None
        NumPy random number generator for controller initialization.
    """

    def __init__(
        self,
        x: float = 600.0,
        y: float = 400.0,
        heading: float = 0.0,
        controller: Any = _DEFAULT_CONTROLLER,
        is_traditional: bool = False,
        speed: float = DEFAULT_SPEED,
        max_speed: float = DEFAULT_MAX_SPEED,
        turning_rate_scale: float = DEFAULT_TURNING_RATE_SCALE,
        radius: float = DEFAULT_HITBOX_RADIUS,
        max_range: float = physics.DEFAULT_MAX_RANGE,
        track_wavefronts: bool = True,
        front_threshold: float = DEFAULT_FRONT_THRESHOLD,
        side_threshold: float = DEFAULT_SIDE_THRESHOLD,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.x: float = float(x)
        self.y: float = float(y)
        self.heading: float = float(heading) % (2.0 * math.pi)
        self.speed: float = float(np.clip(speed, 0.0, max_speed))
        self.max_speed: float = float(max_speed)
        self.turning_rate_scale: float = float(turning_rate_scale)
        self.radius: float = float(radius)
        self.max_range: float = float(max_range)
        self.track_wavefronts: bool = bool(track_wavefronts)
        self.front_threshold: float = float(front_threshold)
        self.side_threshold: float = float(side_threshold)
        self.is_traditional: bool = bool(is_traditional)

        # Neural network controller
        if controller is _DEFAULT_CONTROLLER:
            self.controller: Optional[BatNeuralNet] = BatNeuralNet(rng=rng) if not self.is_traditional else None
        else:
            self.controller = controller

        # Kinematic and behavioral control variables
        self.delta_theta: float = 0.0
        self.urgency: float = 0.0
        self.ping_freq: float = physics.DEFAULT_PULSE_FREQ_MIN
        self.recent_ping_freq: float = physics.DEFAULT_PULSE_FREQ_MIN
        self.ping_timer: float = 0.0

        # Life cycle and performance tracking
        self.is_alive: bool = True
        self.collided: bool = False
        self.time_alive_steps: int = 0
        self.distance_traveled: float = 0.0
        self.total_pings: int = 0
        self.accum_ping_freq: float = 0.0
        self._fitness: Optional[float] = None

        # Visual acoustic wavefronts
        self.wavefronts: List[Wavefront] = []

    # --------------------------------------------------------------------------
    # PROPERTIES & ALIASES
    # --------------------------------------------------------------------------

    @property
    def has_collided(self) -> bool:
        """Alias for collided."""
        return self.collided

    @has_collided.setter
    def has_collided(self, val: bool) -> None:
        self.collided = bool(val)

    @property
    def velocity(self) -> Tuple[float, float]:
        """2D velocity vector (vx, vy)."""
        return (self.speed * math.cos(self.heading), self.speed * math.sin(self.heading))

    @property
    def fitness(self) -> float:
        """
        Scalar fitness score strictly adhering to the project specification:
        fitness = (time_alive_steps * 1.0) + (distance_traveled * 0.5) - (50.0 if collided else 0.0)
        """
        if self._fitness is not None:
            return self._fitness
        penalty = 50.0 if self.collided else 0.0
        return float((self.time_alive_steps * 1.0) + (self.distance_traveled * 0.5) - penalty)

    @fitness.setter
    def fitness(self, val: float) -> None:
        self._fitness = float(val)

    # --------------------------------------------------------------------------
    # SIMULATION STEPPING & KINEMATICS
    # --------------------------------------------------------------------------

    def step(self, dt: float, env: Any = None) -> None:
        """
        Advance drone simulation state by dt seconds.

        Steps:
        1. If dead, return immediately (quiescence).
        2. Accumulate ping_timer.
        3. When ping_timer >= 1 / f_ping, emit ping and update steering and pulse rate.
        4. Integrate heading: theta += delta_theta * turning_rate_scale * dt.
        5. Integrate position: x += vx * dt, y += vy * dt.
        6. Accumulate distance_traveled and time_alive_steps.
        7. Check collision with arena boundaries and environment obstacles.
           If collision occurs, mark is_alive=False and collided=True.
        8. Advance active acoustic wavefronts.

        Parameters
        ----------
        dt : float
            Simulation time increment in seconds.
        env : Any, optional
            Environment providing obstacle geometry and collision verification.
        """
        if not self.is_alive:
            return

        # 1. Acoustic Pulse Modulation & Ping Emission
        self.ping_timer += dt
        period = 1.0 / max(physics.DEFAULT_PULSE_FREQ_MIN, self.ping_freq)
        if self.ping_timer >= period:
            self.emit_ping(env)
            self.ping_timer = self.ping_timer % period

        # 2. Kinematic Integration (Dead Reckoning between pings)
        self.heading = (self.heading + self.delta_theta * self.turning_rate_scale * dt) % (2.0 * math.pi)
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)

        self.x += vx * dt
        self.y += vy * dt
        self.distance_traveled += self.speed * dt
        self.time_alive_steps += 1

        # 3. Collision Verification
        self._check_collisions(env)

        # 4. Wavefront Propagation
        if self.track_wavefronts and self.wavefronts:
            for wf in self.wavefronts:
                wf.update(dt)
            self.wavefronts = [wf for wf in self.wavefronts if wf.is_alive]

    def _check_collisions(self, env: Any = None) -> None:
        """Check collision against arena boundary walls and obstacles."""
        arena_w = getattr(env, "arena_width", physics.ARENA_WIDTH) if env is not None else physics.ARENA_WIDTH
        arena_h = getattr(env, "arena_height", physics.ARENA_HEIGHT) if env is not None else physics.ARENA_HEIGHT

        # Arena boundary wall check
        if (
            self.x - self.radius <= 0.0
            or self.x + self.radius >= arena_w
            or self.y - self.radius <= 0.0
            or self.y + self.radius >= arena_h
        ):
            self.is_alive = False
            self.collided = True
            return

        # Environment obstacle check
        if env is not None:
            if hasattr(env, "check_collision"):
                try:
                    hit = env.check_collision(self.x, self.y, self.radius)
                except TypeError:
                    hit = env.check_collision(self)
                if hit:
                    self.is_alive = False
                    self.collided = True
                    return
            elif hasattr(env, "obstacles"):
                # Duck-typed obstacle list
                obstacles = getattr(env, "obstacles", None)
                if obstacles:
                    for obs in obstacles:
                        # Circle obstacle check
                        if hasattr(obs, "radius") and hasattr(obs, "center"):
                            cx, cy = obs.center
                            r = obs.radius
                            if (self.x - cx) ** 2 + (self.y - cy) ** 2 <= (self.radius + r) ** 2:
                                self.is_alive = False
                                self.collided = True
                                return
                        elif hasattr(obs, "radius") and hasattr(obs, "x") and hasattr(obs, "y"):
                            if (self.x - obs.x) ** 2 + (self.y - obs.y) ** 2 <= (self.radius + obs.radius) ** 2:
                                self.is_alive = False
                                self.collided = True
                                return

    # --------------------------------------------------------------------------
    # ACOUSTIC SENSING & NEURAL CONTROL
    # --------------------------------------------------------------------------

    def emit_ping(self, env: Any = None) -> None:
        """
        Emit bio-acoustic echolocation pulse:
        1. Cast 3 acoustic rays: Left (-30°), Front (0°), Right (+30°).
        2. Compute Doppler closing velocities.
        3. Format and normalize 5-element sensor vector: [d_L, d_R, d_F, v, v_c].
        4. Query controller (neural network or baseline heuristic) to update steering Δθ and urgency λ.
        5. Modulate emission frequency f_ping = 10 + 140*λ.
        6. Queue expanding visual acoustic wavefront.
        """
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)

        obstacles = getattr(env, "obstacles", None) if env is not None else None
        arena_w = getattr(env, "arena_width", physics.ARENA_WIDTH) if env is not None else physics.ARENA_WIDTH
        arena_h = getattr(env, "arena_height", physics.ARENA_HEIGHT) if env is not None else physics.ARENA_HEIGHT

        # Multi-ray acoustic casting
        dist_arr, closing_arr, front_closing = physics.cast_rays_with_doppler(
            origin=(self.x, self.y),
            angle=self.heading,
            ray_offsets=physics.RAY_ANGLES,
            drone_velocity=(vx, vy),
            obstacles=obstacles,
            arena_size=(arena_w, arena_h),
            max_range=self.max_range,
        )

        # Feature mapping: Left is dist_arr[0], Front is dist_arr[1], Right is dist_arr[2]
        d_L = float(dist_arr[0])
        d_F = float(dist_arr[1])
        d_R = float(dist_arr[2])

        if not self.is_traditional and self.controller is not None:
            # 5-element normalized input vector
            u = physics.normalize_sensor_readings(
                left_dist=d_L,
                right_dist=d_R,
                front_dist=d_F,
                speed=self.speed,
                closing_vel=front_closing,
                max_range=self.max_range,
                max_speed=self.max_speed,
            )

            # Neural network inference
            out = self.controller.forward(u)
            self.delta_theta = float(np.clip(out[0], -1.0, 1.0))
            self.urgency = float(np.clip(out[1], 0.0, 1.0))
            self.ping_freq = physics.DEFAULT_PULSE_FREQ_MIN + (
                physics.DEFAULT_PULSE_FREQ_MAX - physics.DEFAULT_PULSE_FREQ_MIN
            ) * self.urgency
            self.recent_ping_freq = self.ping_freq

            # Bio-acoustic Corner Wedge Breakaway Reflex:
            # In orthogonal 90-degree arena corners, symmetric ray hits (d_L == d_R) with front
            # open (d_F > d_L, d_R) deceive the neural network into interpreting the acute wedge
            # as an open corridor gap, commanding near-zero steering straight into the apex.
            # When approaching within 120px of any corner along the diagonal, emergency
            # terminal buzz (150 Hz) and hard banking turn toward the arena center are executed.
            #
            # Extended to also cover straight wall proximity: when a drone is within 50px of
            # any arena boundary and heading toward it, the same emergency manoeuvre fires.
            hx = math.cos(self.heading)
            hy = math.sin(self.heading)
            need_reflex = False

            # 1. Wall proximity check (within 50px and heading toward the wall)
            wall_margin = 50.0
            if (self.x < wall_margin and hx < -0.25) or \
               (self.x > arena_w - wall_margin and hx > 0.25) or \
               (self.y < wall_margin and hy < -0.25) or \
               (self.y > arena_h - wall_margin and hy > 0.25):
                need_reflex = True

            # 2. Corner proximity check (120px diagonal distance, heading toward apex)
            if not need_reflex:
                corners = [(0.0, 0.0), (arena_w, 0.0), (0.0, arena_h), (arena_w, arena_h)]
                for cx, cy in corners:
                    dx = cx - self.x
                    dy = cy - self.y
                    dist = math.hypot(dx, dy)
                    if dist < 120.0:
                        dot = (hx * dx + hy * dy) / max(1e-3, dist)
                        if dot > 0.15:
                            need_reflex = True
                            break

            if need_reflex:
                to_cx = arena_w * 0.5 - self.x
                to_cy = arena_h * 0.5 - self.y
                cross = hx * to_cy - hy * to_cx
                self.delta_theta = 1.0 if cross >= 0 else -1.0
                self.urgency = 1.0
                self.ping_freq = physics.DEFAULT_PULSE_FREQ_MAX
                self.recent_ping_freq = self.ping_freq
        elif self.is_traditional:
            # Traditional Baseline Heuristic: fixed 10 Hz pulse rate, threshold-based steering
            self.urgency = 0.0
            self.ping_freq = physics.DEFAULT_PULSE_FREQ_MIN
            self.recent_ping_freq = physics.DEFAULT_PULSE_FREQ_MIN

            if d_F < self.front_threshold:
                # Steer away from the closer side obstacle
                if d_L < d_R:
                    self.delta_theta = 1.0  # Steer right
                elif d_R < d_L:
                    self.delta_theta = -1.0  # Steer left
                else:
                    self.delta_theta = 1.0  # Default right turn
            elif d_L < self.side_threshold:
                self.delta_theta = 0.5  # Moderate right turn
            elif d_R < self.side_threshold:
                self.delta_theta = -0.5  # Moderate left turn
            else:
                self.delta_theta = 0.0  # Straight ahead

        # Wavefront queueing
        if self.track_wavefronts:
            wf = Wavefront(
                x=self.x,
                y=self.y,
                heading=self.heading,
                radius=0.0,
                max_radius=self.max_range,
                speed=physics.SPEED_OF_SOUND,
                frequency=self.ping_freq,
                urgency=self.urgency,
            )
            self.wavefronts.append(wf)

        self.total_pings += 1
        self.accum_ping_freq += self.ping_freq

    # --------------------------------------------------------------------------
    # LIFECYCLE, RESET & CLONING
    # --------------------------------------------------------------------------

    def reset(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        heading: Optional[float] = None,
    ) -> None:
        """
        Reset drone state for a new generation or episode while preserving controller weights.
        """
        if x is not None:
            self.x = float(x)
        if y is not None:
            self.y = float(y)
        if heading is not None:
            self.heading = float(heading) % (2.0 * math.pi)

        self.is_alive = True
        self.collided = False
        self.time_alive_steps = 0
        self.distance_traveled = 0.0
        self.ping_timer = 0.0
        self.delta_theta = 0.0
        self.urgency = 0.0
        self.ping_freq = physics.DEFAULT_PULSE_FREQ_MIN
        self.recent_ping_freq = physics.DEFAULT_PULSE_FREQ_MIN
        self.total_pings = 0
        self.accum_ping_freq = 0.0
        self._fitness = None
        self.wavefronts.clear()

    def copy(self) -> Drone:
        """
        Create an independent deep copy of this drone for elitism preservation.
        Clones controller weights with zero memory sharing.
        """
        controller_copy = self.controller.copy() if self.controller is not None else None
        new_drone = Drone(
            x=self.x,
            y=self.y,
            heading=self.heading,
            controller=controller_copy,
            is_traditional=self.is_traditional,
            speed=self.speed,
            max_speed=self.max_speed,
            turning_rate_scale=self.turning_rate_scale,
            radius=self.radius,
            max_range=self.max_range,
            track_wavefronts=self.track_wavefronts,
            front_threshold=self.front_threshold,
            side_threshold=self.side_threshold,
        )
        return new_drone

    def draw(self, surface: Any, draw_wavefronts: bool = False) -> None:
        """
        Render drone body, heading direction vector, and optional acoustic wavefronts.
        Handles headless or uninitialized Pygame surfaces gracefully.
        
        Parameters
        ----------
        surface : Any
            Target Pygame surface.
        draw_wavefronts : bool, default=False
            If True, draws expanding circular acoustic wavefront arcs.
            Defaults to False to prevent visual clutter and maintain high FPS.
        """
        if surface is None:
            return

        try:
            import pygame

            # Draw wavefronts first (behind drone) only if explicitly enabled
            if draw_wavefronts and self.track_wavefronts:
                for wf in self.wavefronts:
                    wf.draw(surface)

            # Draw drone body
            cx = int(round(self.x))
            cy = int(round(self.y))
            r = int(round(self.radius))

            if not self.is_alive:
                # Dead / collided drone: dark gray
                pygame.draw.circle(surface, (100, 100, 100), (cx, cy), r)
                return

            # Active drone color based on urgency
            body_color = urgency_to_color(self.urgency) if not self.is_traditional else (52, 152, 219)  # Blue for traditional
            pygame.draw.circle(surface, body_color, (cx, cy), r)
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), r, width=1)

            # Heading indicator needle
            hx = int(round(self.x + (self.radius + 6.0) * math.cos(self.heading)))
            hy = int(round(self.y + (self.radius + 6.0) * math.sin(self.heading)))
            pygame.draw.line(surface, (255, 255, 255), (cx, cy), (hx, hy), width=2)
        except Exception:
            pass


# ==============================================================================
# TRADITIONAL BASELINE DRONE CLASS
# ==============================================================================

class TraditionalDrone(Drone):
    """
    Traditional baseline drone navigating with fixed 10 Hz pulse rate and heuristic turning.
    Used for comparison mode ('C' key) against evolved bio-inspired drones.
    """

    def __init__(
        self,
        x: float = 600.0,
        y: float = 400.0,
        heading: float = 0.0,
        speed: float = DEFAULT_SPEED,
        max_speed: float = DEFAULT_MAX_SPEED,
        turning_rate_scale: float = DEFAULT_TURNING_RATE_SCALE,
        radius: float = DEFAULT_HITBOX_RADIUS,
        max_range: float = physics.DEFAULT_MAX_RANGE,
        track_wavefronts: bool = True,
        front_threshold: float = DEFAULT_FRONT_THRESHOLD,
        side_threshold: float = DEFAULT_SIDE_THRESHOLD,
    ) -> None:
        super().__init__(
            x=x,
            y=y,
            heading=heading,
            controller=None,
            is_traditional=True,
            speed=speed,
            max_speed=max_speed,
            turning_rate_scale=turning_rate_scale,
            radius=radius,
            max_range=max_range,
            track_wavefronts=track_wavefronts,
            front_threshold=front_threshold,
            side_threshold=side_threshold,
        )
