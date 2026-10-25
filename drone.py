from __future__ import annotations
from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import bat_sim.physics as physics
from bat_sim.neural_net import BatNeuralNet
COLOR_SEARCH_GREEN: Tuple[int, int, int] = (46, 204, 113)
COLOR_APPROACH_AMBER: Tuple[int, int, int] = (241, 196, 15)
COLOR_TERMINAL_RED: Tuple[int, int, int] = (231, 76, 60)
DEFAULT_TURNING_RATE_SCALE: float = 3.0
DEFAULT_HITBOX_RADIUS: float = 8.0
DEFAULT_SPEED: float = 60.0
DEFAULT_MAX_SPEED: float = 100.0
DEFAULT_FRONT_THRESHOLD: float = 70.0
DEFAULT_SIDE_THRESHOLD: float = 0.0

def urgency_to_color(urgency: float) -> Tuple[int, int, int]:
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
    return (int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255)))

@dataclass
class Wavefront:
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
        return (self.x, self.y)

    @property
    def max_range(self) -> float:
        return self.max_radius

    @max_range.setter
    def max_range(self, val: float) -> None:
        self.max_radius = float(val)

    @property
    def is_alive(self) -> bool:
        return self.radius < self.max_radius

    def update(self, dt: float) -> bool:
        self.age += dt
        self.radius += self.speed * dt
        return self.is_alive

    def get_color(self) -> Tuple[int, int, int]:
        if self.color is not None:
            return self.color
        return urgency_to_color(self.urgency)

    def get_alpha(self) -> float:
        if self.max_radius <= 0.0:
            return 0.0
        return float(max(0.0, 1.0 - self.radius / self.max_radius))

    def draw(self, surface: Any) -> None:
        if surface is None or not self.is_alive or self.radius <= 0.5:
            return
        try:
            import pygame
            alpha = int(self.get_alpha() * 255.0)
            if alpha <= 0:
                return
            color = self.get_color()
            r = int(round(self.radius))
            cx = int(round(self.x))
            cy = int(round(self.y))
            rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
            if rect.colliderect(surface.get_rect()):
                start_angle = self.heading - self.arc_angle / 2.0
                end_angle = self.heading + self.arc_angle / 2.0
                pygame.draw.arc(surface, color, rect, -end_angle, -start_angle, width=max(1, int(round(2.0 * (alpha / 255.0)))))
        except Exception:
            pass
_DEFAULT_CONTROLLER = object()

class Drone:

    def __init__(self, x: float=600.0, y: float=400.0, heading: float=0.0, controller: Any=_DEFAULT_CONTROLLER, is_traditional: bool=False, speed: float=DEFAULT_SPEED, max_speed: float=DEFAULT_MAX_SPEED, turning_rate_scale: float=DEFAULT_TURNING_RATE_SCALE, radius: float=DEFAULT_HITBOX_RADIUS, max_range: float=physics.DEFAULT_MAX_RANGE, track_wavefronts: bool=True, front_threshold: float=DEFAULT_FRONT_THRESHOLD, side_threshold: float=DEFAULT_SIDE_THRESHOLD, rng: Optional[np.random.Generator]=None) -> None:
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
        if controller is _DEFAULT_CONTROLLER:
            self.controller: Optional[BatNeuralNet] = BatNeuralNet(rng=rng) if not self.is_traditional else None
        else:
            self.controller = controller
        self.delta_theta: float = 0.0
        self.urgency: float = 0.0
        self.ping_freq: float = physics.DEFAULT_PULSE_FREQ_MIN
        self.recent_ping_freq: float = physics.DEFAULT_PULSE_FREQ_MIN
        self.ping_timer: float = 0.0
        self.is_alive: bool = True
        self.collided: bool = False
        self.time_alive_steps: int = 0
        self.distance_traveled: float = 0.0
        self.total_pings: int = 0
        self.accum_ping_freq: float = 0.0
        self._fitness: Optional[float] = None
        self.wavefronts: List[Wavefront] = []

    @property
    def has_collided(self) -> bool:
        return self.collided

    @has_collided.setter
    def has_collided(self, val: bool) -> None:
        self.collided = bool(val)

    @property
    def velocity(self) -> Tuple[float, float]:
        return (self.speed * math.cos(self.heading), self.speed * math.sin(self.heading))

    @property
    def fitness(self) -> float:
        if self._fitness is not None:
            return self._fitness
        penalty = 50.0 if self.collided else 0.0
        return float(self.time_alive_steps * 1.0 + self.distance_traveled * 0.5 - penalty)

    @fitness.setter
    def fitness(self, val: float) -> None:
        self._fitness = float(val)

    def step(self, dt: float, env: Any=None) -> None:
        if not self.is_alive:
            return
        self.ping_timer += dt
        period = 1.0 / max(physics.DEFAULT_PULSE_FREQ_MIN, self.ping_freq)
        if self.ping_timer >= period:
            self.emit_ping(env)
            self.ping_timer = self.ping_timer % period
        self.heading = (self.heading + self.delta_theta * self.turning_rate_scale * dt) % (2.0 * math.pi)
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.x += vx * dt
        self.y += vy * dt
        self.distance_traveled += self.speed * dt
        self.time_alive_steps += 1
        self._check_collisions(env)
        if self.track_wavefronts and self.wavefronts:
            for wf in self.wavefronts:
                wf.update(dt)
            self.wavefronts = [wf for wf in self.wavefronts if wf.is_alive]

    def _check_collisions(self, env: Any=None) -> None:
        arena_w = getattr(env, 'arena_width', physics.ARENA_WIDTH) if env is not None else physics.ARENA_WIDTH
        arena_h = getattr(env, 'arena_height', physics.ARENA_HEIGHT) if env is not None else physics.ARENA_HEIGHT
        if self.x - self.radius <= 0.0 or self.x + self.radius >= arena_w or self.y - self.radius <= 0.0 or (self.y + self.radius >= arena_h):
            self.is_alive = False
            self.collided = True
            return
        if env is not None:
            if hasattr(env, 'check_collision'):
                try:
                    hit = env.check_collision(self.x, self.y, self.radius)
                except TypeError:
                    hit = env.check_collision(self)
                if hit:
                    self.is_alive = False
                    self.collided = True
                    return
            elif hasattr(env, 'obstacles'):
                obstacles = getattr(env, 'obstacles', None)
                if obstacles:
                    for obs in obstacles:
                        if hasattr(obs, 'radius') and hasattr(obs, 'center'):
                            cx, cy = obs.center
                            r = obs.radius
                            if (self.x - cx) ** 2 + (self.y - cy) ** 2 <= (self.radius + r) ** 2:
                                self.is_alive = False
                                self.collided = True
                                return
                        elif hasattr(obs, 'radius') and hasattr(obs, 'x') and hasattr(obs, 'y'):
                            if (self.x - obs.x) ** 2 + (self.y - obs.y) ** 2 <= (self.radius + obs.radius) ** 2:
                                self.is_alive = False
                                self.collided = True
                                return

    def emit_ping(self, env: Any=None) -> None:
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        obstacles = getattr(env, 'obstacles', None) if env is not None else None
        arena_w = getattr(env, 'arena_width', physics.ARENA_WIDTH) if env is not None else physics.ARENA_WIDTH
        arena_h = getattr(env, 'arena_height', physics.ARENA_HEIGHT) if env is not None else physics.ARENA_HEIGHT
        dist_arr, closing_arr, front_closing = physics.cast_rays_with_doppler(origin=(self.x, self.y), angle=self.heading, ray_offsets=physics.RAY_ANGLES, drone_velocity=(vx, vy), obstacles=obstacles, arena_size=(arena_w, arena_h), max_range=self.max_range)
        d_L = float(dist_arr[0])
        d_F = float(dist_arr[1])
        d_R = float(dist_arr[2])
        if not self.is_traditional and self.controller is not None:
            u = physics.normalize_sensor_readings(left_dist=d_L, right_dist=d_R, front_dist=d_F, speed=self.speed, closing_vel=front_closing, max_range=self.max_range, max_speed=self.max_speed)
            out = self.controller.forward(u)
            self.delta_theta = float(np.clip(out[0], -1.0, 1.0))
            self.urgency = float(np.clip(out[1], 0.0, 1.0))
            self.ping_freq = physics.DEFAULT_PULSE_FREQ_MIN + (physics.DEFAULT_PULSE_FREQ_MAX - physics.DEFAULT_PULSE_FREQ_MIN) * self.urgency
            self.recent_ping_freq = self.ping_freq
            hx = math.cos(self.heading)
            hy = math.sin(self.heading)
            need_reflex = False
            wall_margin = 50.0
            if self.x < wall_margin and hx < -0.25 or (self.x > arena_w - wall_margin and hx > 0.25) or (self.y < wall_margin and hy < -0.25) or (self.y > arena_h - wall_margin and hy > 0.25):
                need_reflex = True
            if not need_reflex:
                corners = [(0.0, 0.0), (arena_w, 0.0), (0.0, arena_h), (arena_w, arena_h)]
                for cx, cy in corners:
                    dx = cx - self.x
                    dy = cy - self.y
                    dist = math.hypot(dx, dy)
                    if dist < 120.0:
                        dot = (hx * dx + hy * dy) / max(0.001, dist)
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
            self.urgency = 0.0
            self.ping_freq = physics.DEFAULT_PULSE_FREQ_MIN
            self.recent_ping_freq = physics.DEFAULT_PULSE_FREQ_MIN
            if d_F < self.front_threshold:
                if d_L < d_R:
                    self.delta_theta = 1.0
                elif d_R < d_L:
                    self.delta_theta = -1.0
                else:
                    self.delta_theta = 1.0
            elif d_L < self.side_threshold:
                self.delta_theta = 0.5
            elif d_R < self.side_threshold:
                self.delta_theta = -0.5
            else:
                self.delta_theta = 0.0
        if self.track_wavefronts:
            wf = Wavefront(x=self.x, y=self.y, heading=self.heading, radius=0.0, max_radius=self.max_range, speed=physics.SPEED_OF_SOUND, frequency=self.ping_freq, urgency=self.urgency)
            self.wavefronts.append(wf)
        self.total_pings += 1
        self.accum_ping_freq += self.ping_freq

    def reset(self, x: Optional[float]=None, y: Optional[float]=None, heading: Optional[float]=None) -> None:
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
        controller_copy = self.controller.copy() if self.controller is not None else None
        new_drone = Drone(x=self.x, y=self.y, heading=self.heading, controller=controller_copy, is_traditional=self.is_traditional, speed=self.speed, max_speed=self.max_speed, turning_rate_scale=self.turning_rate_scale, radius=self.radius, max_range=self.max_range, track_wavefronts=self.track_wavefronts, front_threshold=self.front_threshold, side_threshold=self.side_threshold)
        return new_drone

    def draw(self, surface: Any, draw_wavefronts: bool=False) -> None:
        if surface is None:
            return
        try:
            import pygame
            if draw_wavefronts and self.track_wavefronts:
                for wf in self.wavefronts:
                    wf.draw(surface)
            cx = int(round(self.x))
            cy = int(round(self.y))
            r = int(round(self.radius))
            if not self.is_alive:
                pygame.draw.circle(surface, (100, 100, 100), (cx, cy), r)
                return
            body_color = urgency_to_color(self.urgency) if not self.is_traditional else (52, 152, 219)
            pygame.draw.circle(surface, body_color, (cx, cy), r)
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), r, width=1)
            hx = int(round(self.x + (self.radius + 6.0) * math.cos(self.heading)))
            hy = int(round(self.y + (self.radius + 6.0) * math.sin(self.heading)))
            pygame.draw.line(surface, (255, 255, 255), (cx, cy), (hx, hy), width=2)
        except Exception:
            pass

class TraditionalDrone(Drone):

    def __init__(self, x: float=600.0, y: float=400.0, heading: float=0.0, speed: float=DEFAULT_SPEED, max_speed: float=DEFAULT_MAX_SPEED, turning_rate_scale: float=DEFAULT_TURNING_RATE_SCALE, radius: float=DEFAULT_HITBOX_RADIUS, max_range: float=physics.DEFAULT_MAX_RANGE, track_wavefronts: bool=True, front_threshold: float=DEFAULT_FRONT_THRESHOLD, side_threshold: float=DEFAULT_SIDE_THRESHOLD) -> None:
        super().__init__(x=x, y=y, heading=heading, controller=None, is_traditional=True, speed=speed, max_speed=max_speed, turning_rate_scale=turning_rate_scale, radius=radius, max_range=max_range, track_wavefronts=track_wavefronts, front_threshold=front_threshold, side_threshold=side_threshold)
