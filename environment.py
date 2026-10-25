"""
bat_sim/environment.py
======================
Obstacle Arena & Interactive Simulation World for Bat Echolocation Simulation.

Features:
- CircularObstacle: Circular obstacle with radius, position, drift velocity, point containment,
  boundary bouncing, and Pygame rendering.
- RectangularObstacle: Axis-aligned and oriented rectangular obstacle with width, height, angle,
  drift velocity, point containment, boundary bouncing, and Pygame polygon rendering.
- Environment: 1200x800 arena world managing static and drifting obstacles, real-time mouse
  drag-and-drop interaction, click-to-spawn helpers, exact circle-circle and circle-OBB collision
  detection, and optimized cached array properties for vectorized ray casting.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import bat_sim.physics as physics


# ==============================================================================
# DEFAULT ARENA CONSTANTS
# ==============================================================================

DEFAULT_ARENA_WIDTH: float = 1200.0
DEFAULT_ARENA_HEIGHT: float = 800.0

DEFAULT_OBSTACLE_COLOR_CIRCLE: Tuple[int, int, int] = (65, 80, 95)
DEFAULT_OBSTACLE_COLOR_RECT: Tuple[int, int, int] = (55, 70, 85)
DRAGGING_BORDER_COLOR: Tuple[int, int, int] = (0, 220, 255)
NORMAL_BORDER_COLOR: Tuple[int, int, int] = (140, 155, 175)


# ==============================================================================
# CIRCULAR OBSTACLE CLASS
# ==============================================================================

class CircularObstacle:
    """
    Circular obstacle in the 2D arena with position, radius, and drift kinematics.

    Parameters
    ----------
    x : float
        Horizontal center coordinate in pixels.
    y : float
        Vertical center coordinate in pixels.
    radius : float, default=30.0
        Obstacle radius in pixels.
    vx : float, default=0.0
        Drift velocity along x-axis in pixels/second.
    vy : float, default=0.0
        Drift velocity along y-axis in pixels/second.
    color : Optional[Tuple[int, int, int]], default=None
        RGB fill color tuple; defaults to slate gray.
    """

    def __init__(
        self,
        x: float,
        y: float,
        radius: float = 30.0,
        vx: float = 0.0,
        vy: float = 0.0,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.x: float = float(x)
        self.y: float = float(y)
        self.radius: float = float(radius)
        self.vx: float = float(vx)
        self.vy: float = float(vy)
        self.color: Tuple[int, int, int] = color if color is not None else DEFAULT_OBSTACLE_COLOR_CIRCLE
        self.is_dragging: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinates (x, y) compatible with physics.py and evolution.py."""
        return (self.x, self.y)

    @center.setter
    def center(self, val: Sequence[float]) -> None:
        self.x = float(val[0])
        self.y = float(val[1])

    @property
    def pos(self) -> Tuple[float, float]:
        """Alias for center."""
        return (self.x, self.y)

    @pos.setter
    def pos(self, val: Sequence[float]) -> None:
        self.center = val

    @property
    def position(self) -> Tuple[float, float]:
        """Alias for center."""
        return (self.x, self.y)

    @position.setter
    def position(self, val: Sequence[float]) -> None:
        self.center = val

    @property
    def velocity(self) -> Tuple[float, float]:
        """Velocity vector (vx, vy)."""
        return (self.vx, self.vy)

    @velocity.setter
    def velocity(self, val: Sequence[float]) -> None:
        self.vx = float(val[0])
        self.vy = float(val[1])

    def contains_point(self, px: float, py: float) -> bool:
        """
        Hit-test point (px, py) for mouse click and drag selection.
        """
        return bool((float(px) - self.x) ** 2 + (float(py) - self.y) ** 2 <= self.radius ** 2)

    def update(
        self,
        dt: float,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, DEFAULT_ARENA_WIDTH, DEFAULT_ARENA_HEIGHT),
    ) -> None:
        """
        Update position under drift velocity and reflect upon reaching arena bounds.

        Parameters
        ----------
        dt : float
            Simulation time increment in seconds.
        bounds : Tuple[float, float, float, float]
            Arena bounding box (min_x, min_y, max_x, max_y).
        """
        if self.is_dragging:
            return

        self.x += self.vx * dt
        self.y += self.vy * dt

        min_x, min_y, max_x, max_y = bounds

        # Horizontal boundary bounce
        if self.x - self.radius < min_x:
            self.x = min_x + self.radius
            self.vx = -self.vx
        elif self.x + self.radius > max_x:
            self.x = max_x - self.radius
            self.vx = -self.vx

        # Vertical boundary bounce
        if self.y - self.radius < min_y:
            self.y = min_y + self.radius
            self.vy = -self.vy
        elif self.y + self.radius > max_y:
            self.y = max_y - self.radius
            self.vy = -self.vy

    def step(
        self,
        dt: float,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, DEFAULT_ARENA_WIDTH, DEFAULT_ARENA_HEIGHT),
    ) -> None:
        """Alias for update."""
        self.update(dt, bounds)

    def draw(self, surface: Any) -> None:
        """
        Render circular obstacle onto a Pygame surface.
        """
        if surface is None:
            return
        try:
            import pygame

            cx = int(round(self.x))
            cy = int(round(self.y))
            r = int(round(self.radius))

            pygame.draw.circle(surface, self.color, (cx, cy), r)
            border_col = DRAGGING_BORDER_COLOR if self.is_dragging else NORMAL_BORDER_COLOR
            pygame.draw.circle(surface, border_col, (cx, cy), r, width=2)

            # Draw drift indicator arrow if moving
            if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
                spd = math.hypot(self.vx, self.vy)
                if spd > 1e-3:
                    ax = cx + int(round((self.vx / spd) * max(2.0, r - 4)))
                    ay = cy + int(round((self.vy / spd) * max(2.0, r - 4)))
                    pygame.draw.line(surface, border_col, (cx, cy), (ax, ay), 2)
        except Exception:
            pass


# ==============================================================================
# RECTANGULAR OBSTACLE CLASS
# ==============================================================================

class RectangularObstacle:
    """
    Rectangular obstacle (AABB or OBB) in the 2D arena with position, dimensions,
    orientation angle, and drift kinematics.

    Parameters
    ----------
    x : float
        Horizontal center coordinate in pixels.
    y : float
        Vertical center coordinate in pixels.
    width : float, default=60.0
        Rectangle width in pixels.
    height : float, default=40.0
        Rectangle height in pixels.
    angle : float, default=0.0
        Orientation rotation angle in radians.
    vx : float, default=0.0
        Drift velocity along x-axis in pixels/second.
    vy : float, default=0.0
        Drift velocity along y-axis in pixels/second.
    color : Optional[Tuple[int, int, int]], default=None
        RGB fill color tuple.
    """

    def __init__(
        self,
        x: float,
        y: float,
        width: float = 60.0,
        height: float = 40.0,
        angle: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.x: float = float(x)
        self.y: float = float(y)
        self.width: float = float(width)
        self.height: float = float(height)
        self.angle: float = float(angle)
        self.vx: float = float(vx)
        self.vy: float = float(vy)
        self.color: Tuple[int, int, int] = color if color is not None else DEFAULT_OBSTACLE_COLOR_RECT
        self.is_dragging: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinates (x, y) compatible with physics.py and evolution.py."""
        return (self.x, self.y)

    @center.setter
    def center(self, val: Sequence[float]) -> None:
        self.x = float(val[0])
        self.y = float(val[1])

    @property
    def pos(self) -> Tuple[float, float]:
        """Alias for center."""
        return (self.x, self.y)

    @pos.setter
    def pos(self, val: Sequence[float]) -> None:
        self.center = val

    @property
    def size(self) -> Tuple[float, float]:
        """Dimensions tuple (width, height)."""
        return (self.width, self.height)

    @size.setter
    def size(self, val: Sequence[float]) -> None:
        self.width = float(val[0])
        self.height = float(val[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Velocity vector (vx, vy)."""
        return (self.vx, self.vy)

    @velocity.setter
    def velocity(self, val: Sequence[float]) -> None:
        self.vx = float(val[0])
        self.vy = float(val[1])

    @property
    def rect(self) -> Tuple[float, float, float, float, float]:
        """5-tuple (cx, cy, w, h, angle) expected by physics.cast_ray_rect."""
        return (self.x, self.y, self.width, self.height, self.angle)

    def contains_point(self, px: float, py: float) -> bool:
        """
        Hit-test point (px, py) for mouse click and drag selection in local coordinates.
        """
        rel_x = float(px) - self.x
        rel_y = float(py) - self.y
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        lx = rel_x * cos_a + rel_y * sin_a
        ly = -rel_x * sin_a + rel_y * cos_a
        return bool(abs(lx) <= self.width / 2.0 and abs(ly) <= self.height / 2.0)

    def get_corners(self) -> List[Tuple[float, float]]:
        """
        Compute the 4 world-coordinate vertices of the rotated rectangle.
        """
        hw = self.width / 2.0
        hh = self.height / 2.0
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        local_corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        return [(self.x + lx * cos_a - ly * sin_a, self.y + lx * sin_a + ly * cos_a) for lx, ly in local_corners]

    def update(
        self,
        dt: float,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, DEFAULT_ARENA_WIDTH, DEFAULT_ARENA_HEIGHT),
    ) -> None:
        """
        Update position under drift velocity and reflect upon reaching arena bounds.
        """
        if self.is_dragging:
            return

        self.x += self.vx * dt
        self.y += self.vy * dt

        min_x, min_y, max_x, max_y = bounds

        # Project oriented half-extents onto world coordinate axes
        hw_proj = 0.5 * (abs(self.width * math.cos(self.angle)) + abs(self.height * math.sin(self.angle)))
        hh_proj = 0.5 * (abs(self.width * math.sin(self.angle)) + abs(self.height * math.cos(self.angle)))

        # Horizontal boundary bounce
        if self.x - hw_proj < min_x:
            self.x = min_x + hw_proj
            self.vx = -self.vx
        elif self.x + hw_proj > max_x:
            self.x = max_x - hw_proj
            self.vx = -self.vx

        # Vertical boundary bounce
        if self.y - hh_proj < min_y:
            self.y = min_y + hh_proj
            self.vy = -self.vy
        elif self.y + hh_proj > max_y:
            self.y = max_y - hh_proj
            self.vy = -self.vy

    def step(
        self,
        dt: float,
        bounds: Tuple[float, float, float, float] = (0.0, 0.0, DEFAULT_ARENA_WIDTH, DEFAULT_ARENA_HEIGHT),
    ) -> None:
        """Alias for update."""
        self.update(dt, bounds)

    def draw(self, surface: Any) -> None:
        """
        Render oriented rectangle polygon onto a Pygame surface.
        """
        if surface is None:
            return
        try:
            import pygame

            corners = self.get_corners()
            pygame.draw.polygon(surface, self.color, corners)
            border_col = DRAGGING_BORDER_COLOR if self.is_dragging else NORMAL_BORDER_COLOR
            pygame.draw.polygon(surface, border_col, corners, width=2)

            # Draw drift indicator arrow if moving
            if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
                spd = math.hypot(self.vx, self.vy)
                if spd > 1e-3:
                    cx = int(round(self.x))
                    cy = int(round(self.y))
                    reach = min(self.width, self.height) * 0.4
                    ax = cx + int(round((self.vx / spd) * max(4.0, reach)))
                    ay = cy + int(round((self.vy / spd) * max(4.0, reach)))
                    pygame.draw.line(surface, border_col, (cx, cy), (ax, ay), 2)
        except Exception:
            pass


# ==============================================================================
# ENVIRONMENT CLASS
# ==============================================================================

class Environment:
    """
    Simulation world and obstacle arena (1200x800) managing obstacles,
    collision checking, drift updates, and mouse click/drag interactions.

    Parameters
    ----------
    arena_width : float, default=1200.0
        Width of the simulation arena in pixels.
    arena_height : float, default=800.0
        Height of the simulation arena in pixels.
    obstacles : Optional[Sequence[Any]], default=None
        Initial list of obstacles; if None, initializes with default balanced obstacle field.
    """

    def __init__(
        self,
        arena_width: float = DEFAULT_ARENA_WIDTH,
        arena_height: float = DEFAULT_ARENA_HEIGHT,
        obstacles: Optional[Sequence[Any]] = None,
    ) -> None:
        self.arena_width: float = float(arena_width)
        self.arena_height: float = float(arena_height)
        self.arena_size: Tuple[float, float] = (self.arena_width, self.arena_height)

        self.obstacles: List[Any] = []
        if obstacles is not None:
            self.obstacles.extend(obstacles)
        else:
            self.reset_default_obstacles()

        # Mouse interaction tracking
        self.dragged_obstacle: Optional[Any] = None
        self.drag_offset_x: float = 0.0
        self.drag_offset_y: float = 0.0

    # --------------------------------------------------------------------------
    # OBSTACLE MANAGEMENT & SETUP
    # --------------------------------------------------------------------------

    def set_arena_size(self, width: float, height: float) -> None:
        """Update arena boundaries and redistribute default obstacles proportionally."""
        self.arena_width = float(width)
        self.arena_height = float(height)
        self.arena_size = (self.arena_width, self.arena_height)
        self.reset_default_obstacles()

    def add_obstacle(self, obstacle: Any) -> None:
        """Add an obstacle to the arena."""
        self.obstacles.append(obstacle)

    def remove_obstacle(self, obstacle: Any) -> bool:
        """Remove an obstacle from the arena. Returns True if removed."""
        if obstacle in self.obstacles:
            self.obstacles.remove(obstacle)
            if self.dragged_obstacle is obstacle:
                self.stop_drag()
            return True
        return False

    def clear_obstacles(self) -> None:
        """Clear all obstacles from the arena."""
        self.obstacles.clear()
        self.dragged_obstacle = None

    def reset_default_obstacles(self) -> None:
        """
        Spawn a balanced, challenging default obstacle field leaving the drone spawn
        center (arena center) completely clear of obstacles with a >= 140px safe harbor.

        All obstacle dimensions scale proportionally to arena size so that corridors
        remain navigable by a champion model trained at the reference 1200x800 arena.
        """
        self.obstacles.clear()
        self.dragged_obstacle = None

        aw = self.arena_width
        ah = self.arena_height
        cx, cy = aw * 0.5, ah * 0.5

        # Proportional scale factor relative to the 1200x800 training arena.
        # Obstacles shrink linearly with the smaller arena dimension so that
        # the ratio of obstacle-radius to corridor-width stays constant.
        scale = min(1.0, min(aw, ah) / 800.0)

        # Static Circular Obstacles (quadrant pylons far from center)
        pylon_r = max(12.0, 30.0 * scale)
        self.obstacles.append(CircularObstacle(aw * 0.22, ah * 0.22, radius=pylon_r))
        self.obstacles.append(CircularObstacle(aw * 0.78, ah * 0.22, radius=pylon_r))
        self.obstacles.append(CircularObstacle(aw * 0.22, ah * 0.78, radius=pylon_r))
        self.obstacles.append(CircularObstacle(aw * 0.78, ah * 0.78, radius=pylon_r))

        # Static Rectangular Obstacles (corridor baffles near top/bottom walls)
        baffle_w = min(150.0 * scale, aw * 0.20)
        baffle_h = max(12.0, 28.0 * scale)
        self.obstacles.append(RectangularObstacle(cx, ah * 0.12, width=baffle_w, height=baffle_h, angle=0.0))
        self.obstacles.append(RectangularObstacle(cx, ah * 0.88, width=baffle_w, height=baffle_h, angle=0.0))

        # Dynamic Drifting Circular Obstacles (flanking corridors far left and right)
        drift_r = max(10.0, 26.0 * scale)
        drift_speed = 25.0 * scale
        self.obstacles.append(CircularObstacle(aw * 0.12, cy, radius=drift_r, vx=0.0, vy=drift_speed))
        self.obstacles.append(CircularObstacle(aw * 0.88, cy, radius=drift_r, vx=0.0, vy=-drift_speed))

        # Dynamic Drifting Rectangular Obstacle in upper-left quadrant (never in center)
        drift_rw = max(20.0, 45.0 * scale)
        drift_rh = max(14.0, 30.0 * scale)
        self.obstacles.append(RectangularObstacle(
            aw * (350.0 / 1200.0), ah * (230.0 / 800.0),
            width=drift_rw, height=drift_rh, angle=0.25,
            vx=12.0 * scale, vy=18.0 * scale,
        ))

    def reset(self) -> None:
        """Reset environment to default layout."""
        self.reset_default_obstacles()

    # --------------------------------------------------------------------------
    # MOUSE INTERACTION HELPERS
    # --------------------------------------------------------------------------

    def get_obstacle_at(self, x: float, y: float) -> Optional[Any]:
        """
        Hit-test all obstacles at (x, y), checking topmost (most recently added) first.
        """
        px = float(x)
        py = float(y)
        for obs in reversed(self.obstacles):
            if hasattr(obs, "contains_point") and obs.contains_point(px, py):
                return obs
        return None

    def start_drag(self, x: float, y: float) -> Optional[Any]:
        """
        Initiate dragging for the obstacle located under cursor (x, y).
        """
        px = float(x)
        py = float(y)
        obs = self.get_obstacle_at(px, py)
        if obs is not None:
            self.dragged_obstacle = obs
            obs.is_dragging = True
            self.drag_offset_x = px - obs.x
            self.drag_offset_y = py - obs.y
            return obs
        return None

    def drag_to(self, x: float, y: float) -> None:
        """
        Update position of currently dragged obstacle, clamping within arena bounds.
        """
        if self.dragged_obstacle is None:
            return

        obs = self.dragged_obstacle
        target_x = float(x) - self.drag_offset_x
        target_y = float(y) - self.drag_offset_y

        if hasattr(obs, "radius"):
            # Circular obstacle clamping
            r = float(obs.radius)
            obs.x = max(r, min(self.arena_width - r, target_x))
            obs.y = max(r, min(self.arena_height - r, target_y))
        elif hasattr(obs, "width") and hasattr(obs, "height"):
            # Rectangular obstacle clamping
            angle = float(getattr(obs, "angle", 0.0))
            hw = 0.5 * (abs(float(obs.width) * math.cos(angle)) + abs(float(obs.height) * math.sin(angle)))
            hh = 0.5 * (abs(float(obs.width) * math.sin(angle)) + abs(float(obs.height) * math.cos(angle)))
            obs.x = max(hw, min(self.arena_width - hw, target_x))
            obs.y = max(hh, min(self.arena_height - hh, target_y))
        else:
            obs.x = max(0.0, min(self.arena_width, target_x))
            obs.y = max(0.0, min(self.arena_height, target_y))

    def stop_drag(self) -> None:
        """Release currently dragged obstacle."""
        if self.dragged_obstacle is not None:
            self.dragged_obstacle.is_dragging = False
            self.dragged_obstacle = None

    def spawn_circle(
        self,
        x: float,
        y: float,
        radius: float = 30.0,
        vx: float = 0.0,
        vy: float = 0.0,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> CircularObstacle:
        """Spawn and return a new circular obstacle at (x, y)."""
        r = float(radius)
        clamped_x = max(r, min(self.arena_width - r, float(x)))
        clamped_y = max(r, min(self.arena_height - r, float(y)))
        obs = CircularObstacle(x=clamped_x, y=clamped_y, radius=r, vx=vx, vy=vy, color=color)
        self.add_obstacle(obs)
        return obs

    def spawn_rect(
        self,
        x: float,
        y: float,
        width: float = 60.0,
        height: float = 40.0,
        angle: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> RectangularObstacle:
        """Spawn and return a new rectangular obstacle at (x, y)."""
        w, h = float(width), float(height)
        hw = 0.5 * (abs(w * math.cos(angle)) + abs(h * math.sin(angle)))
        hh = 0.5 * (abs(w * math.sin(angle)) + abs(h * math.cos(angle)))
        clamped_x = max(hw, min(self.arena_width - hw, float(x)))
        clamped_y = max(hh, min(self.arena_height - hh, float(y)))
        obs = RectangularObstacle(x=clamped_x, y=clamped_y, width=w, height=h, angle=angle, vx=vx, vy=vy, color=color)
        self.add_obstacle(obs)
        return obs

    # --------------------------------------------------------------------------
    # SIMULATION STEPPING & COLLISION DETECTION
    # --------------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance obstacle drift kinematics and handle boundary bounces."""
        bounds = (0.0, 0.0, self.arena_width, self.arena_height)
        for obs in self.obstacles:
            if hasattr(obs, "update"):
                obs.update(dt, bounds)
            elif hasattr(obs, "step"):
                obs.step(dt, bounds)

    def update(self, dt: float) -> None:
        """Alias for step(dt)."""
        self.step(dt)

    def is_out_of_bounds(self, x: float, y: float, radius: float = 8.0) -> bool:
        """True if drone hitbox intersects arena boundary walls."""
        px = float(x)
        py = float(y)
        pr = float(radius)
        return bool(
            px - pr <= 0.0
            or px + pr >= self.arena_width
            or py - pr <= 0.0
            or py + pr >= self.arena_height
        )

    def check_collision(
        self,
        x_or_drone: Union[float, Any],
        y: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> bool:
        """
        Exact collision detection for drones against arena walls and all obstacles in the arena.
        Supports both env.check_collision(x, y, radius) and env.check_collision(drone).
        """
        if hasattr(x_or_drone, "x") and hasattr(x_or_drone, "y"):
            px = float(x_or_drone.x)
            py = float(x_or_drone.y)
            pr = float(getattr(x_or_drone, "radius", 8.0))
        else:
            px = float(x_or_drone)
            py = float(y) if y is not None else 0.0
            pr = float(radius) if radius is not None else 8.0

        # Boundary wall check
        if self.is_out_of_bounds(px, py, pr):
            return True

        for obs in self.obstacles:
            # 1. Circular obstacle collision check
            if hasattr(obs, "radius"):
                if hasattr(obs, "center"):
                    cx, cy = obs.center
                else:
                    cx, cy = getattr(obs, "x", 0.0), getattr(obs, "y", 0.0)
                cr = float(obs.radius)
                if (px - float(cx)) ** 2 + (py - float(cy)) ** 2 <= (pr + cr) ** 2:
                    return True

            # 2. Rectangular obstacle collision check (AABB and OBB via local frame)
            elif hasattr(obs, "width") and hasattr(obs, "height"):
                if hasattr(obs, "center"):
                    cx, cy = obs.center
                else:
                    cx, cy = getattr(obs, "x", 0.0), getattr(obs, "y", 0.0)
                w = float(obs.width)
                h = float(obs.height)
                angle = float(getattr(obs, "angle", 0.0))

                rel_x = px - float(cx)
                rel_y = py - float(cy)
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                lx = rel_x * cos_a + rel_y * sin_a
                ly = -rel_x * sin_a + rel_y * cos_a

                hw = w / 2.0
                hh = h / 2.0

                closest_x = max(-hw, min(hw, lx))
                closest_y = max(-hh, min(hh, ly))

                if (lx - closest_x) ** 2 + (ly - closest_y) ** 2 <= pr ** 2:
                    return True

        return False

    # --------------------------------------------------------------------------
    # FORMATTED PROPERTIES FOR VECTORIZED RAY-CASTING & EVOLUTION
    # --------------------------------------------------------------------------

    @property
    def circle_centers(self) -> Optional[np.ndarray]:
        """Array of shape (Mc, 2) of circular obstacle centers for batch_cast_rays_all."""
        c = [obs.center for obs in self.obstacles if hasattr(obs, "radius")]
        return np.array(c, dtype=np.float64) if c else None

    @property
    def circle_radii(self) -> Optional[np.ndarray]:
        """Array of shape (Mc,) of circular obstacle radii."""
        r = [obs.radius for obs in self.obstacles if hasattr(obs, "radius")]
        return np.array(r, dtype=np.float64) if r else None

    @property
    def circle_velocities(self) -> Optional[np.ndarray]:
        """Array of shape (Mc, 2) of circular obstacle velocities."""
        v = [obs.velocity for obs in self.obstacles if hasattr(obs, "radius")]
        return np.array(v, dtype=np.float64) if v else None

    @property
    def rect_centers(self) -> Optional[np.ndarray]:
        """Array of shape (Mr, 2) of rectangular obstacle centers."""
        c = [obs.center for obs in self.obstacles if hasattr(obs, "width") and hasattr(obs, "height")]
        return np.array(c, dtype=np.float64) if c else None

    @property
    def rect_sizes(self) -> Optional[np.ndarray]:
        """Array of shape (Mr, 2) of rectangular obstacle sizes [width, height]."""
        s = [obs.size for obs in self.obstacles if hasattr(obs, "width") and hasattr(obs, "height")]
        return np.array(s, dtype=np.float64) if s else None

    @property
    def rect_angles(self) -> Optional[np.ndarray]:
        """Array of shape (Mr,) of rectangular obstacle rotation angles."""
        a = [getattr(obs, "angle", 0.0) for obs in self.obstacles if hasattr(obs, "width") and hasattr(obs, "height")]
        return np.array(a, dtype=np.float64) if a else None

    @property
    def rect_velocities(self) -> Optional[np.ndarray]:
        """Array of shape (Mr, 2) of rectangular obstacle velocities."""
        v = [obs.velocity for obs in self.obstacles if hasattr(obs, "width") and hasattr(obs, "height")]
        return np.array(v, dtype=np.float64) if v else None

    # --------------------------------------------------------------------------
    # RENDERING
    # --------------------------------------------------------------------------

    def draw(self, surface: Any) -> None:
        """Render all obstacles onto Pygame surface."""
        if surface is None:
            return
        for obs in self.obstacles:
            if hasattr(obs, "draw"):
                obs.draw(surface)
