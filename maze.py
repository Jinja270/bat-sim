from __future__ import annotations
import math
import os
import sys
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pygame

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bat_sim.neural_net import BatNeuralNet, TOTAL_WEIGHTS, compute_mutation_sigma
from bat_sim.neural_ipc import ActivationWriter
import bat_sim.physics as physics

COLOR_FLOOR: Tuple[int, int, int] = (12, 16, 24)
COLOR_GRID_LINE: Tuple[int, int, int] = (20, 26, 38)
COLOR_WALL_TOP: Tuple[int, int, int] = (252, 254, 255)
COLOR_WALL_BORDER: Tuple[int, int, int] = (210, 222, 238)
COLOR_WALL_SIDE_S: Tuple[int, int, int] = (155, 170, 192)
COLOR_WALL_SIDE_E: Tuple[int, int, int] = (180, 195, 215)
COLOR_WALL_SHADOW: Tuple[int, int, int] = (5, 8, 14)

COLOR_SWARM_GREEN: Tuple[int, int, int] = (42, 225, 110)
COLOR_SWARM_CORE: Tuple[int, int, int] = (140, 255, 180)
COLOR_SWARM_DEAD: Tuple[int, int, int] = (60, 75, 70)
COLOR_RAY_GREEN: Tuple[int, int, int] = (0, 240, 160)
COLOR_RAY_HIT: Tuple[int, int, int] = (220, 255, 50)

COLOR_START_ZONE: Tuple[int, int, int] = (35, 160, 85)
COLOR_GOAL_ZONE: Tuple[int, int, int] = (0, 210, 255)
COLOR_GOAL_GOLD: Tuple[int, int, int] = (255, 200, 50)


class Maze:
    def __init__(self, cols: int = 35, rows: int = 22, cell_size: float = 40.0, seed: Optional[int] = None) -> None:
        self.cols = int(cols)
        self.rows = int(rows)
        self.cell_size = float(cell_size)
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size
        self.grid = np.ones((self.rows, self.cols), dtype=np.uint8)
        self.dist_field = np.full((self.rows, self.cols), -1.0, dtype=np.float64)
        self.dist_start = np.full((self.rows, self.cols), -1.0, dtype=np.float64)
        self.start_to_goal_dist = 1.0
        self.max_corridor_depth = 1.0
        self.start_pos = (1.5 * self.cell_size, 1.5 * self.cell_size)
        self.goal_pos = ((self.cols - 2.5) * self.cell_size, (self.rows - 2.5) * self.cell_size)
        self.goal_radius = self.cell_size * 1.5
        self.rng = random.Random(seed)
        self.generate()
        self._cached_wall_surface: Optional[pygame.Surface] = None

    def generate(self) -> None:
        self.grid.fill(1)
        grid_w = (self.cols - 1) // 2
        grid_h = (self.rows - 1) // 2
        visited = np.zeros((grid_h, grid_w), dtype=bool)

        def carve(cx: int, cy: int) -> None:
            visited[cy, cx] = True
            self.grid[cy * 2 + 1, cx * 2 + 1] = 0
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.rng.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h and not visited[ny, nx]:
                    self.grid[cy * 2 + 1 + dy, cx * 2 + 1 + dx] = 0
                    carve(nx, ny)

        carve(0, 0)

        for y in range(1, self.rows - 1):
            for x in range(1, self.cols - 1):
                if self.grid[y, x] == 0:
                    wall_count = sum(self.grid[y + dy, x + dx] for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)])
                    if wall_count == 3 and self.rng.random() < 0.35:
                        walls = [
                            (x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                            if self.grid[y + dy, x + dx] == 1 and 1 <= x + dx < self.cols - 1 and 1 <= y + dy < self.rows - 1
                        ]
                        if walls:
                            wx, wy = self.rng.choice(walls)
                            self.grid[wy, wx] = 0

        self.grid[1:3, 1:3] = 0
        self.grid[-3:-1, -3:-1] = 0
        self.grid[1, 3] = 0
        self.grid[1, 4] = 0
        self.grid[3, 1] = 0
        self.grid[4, 1] = 0

        self.start_pos = (1.5 * self.cell_size, 1.5 * self.cell_size)
        self.goal_pos = ((self.cols - 2.5) * self.cell_size, (self.rows - 2.5) * self.cell_size)

        self._compute_distance_field()
        self._cached_wall_surface = None

    def _compute_distance_field(self) -> None:
        self.dist_field.fill(-1.0)
        gx = int(self.goal_pos[0] / self.cell_size)
        gy = int(self.goal_pos[1] / self.cell_size)
        gx = max(0, min(self.cols - 1, gx))
        gy = max(0, min(self.rows - 1, gy))

        q = deque([(gx, gy, 0.0)])
        self.dist_field[gy, gx] = 0.0

        while q:
            x, y, d = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.grid[ny, nx] == 0 and self.dist_field[ny, nx] < 0.0:
                        self.dist_field[ny, nx] = d + 1.0
                        q.append((nx, ny, d + 1.0))

        self.dist_start.fill(-1.0)
        sx = int(self.start_pos[0] / self.cell_size)
        sy = int(self.start_pos[1] / self.cell_size)
        qs = deque([(sx, sy, 0.0)])
        self.dist_start[sy, sx] = 0.0

        while qs:
            x, y, d = qs.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.grid[ny, nx] == 0 and self.dist_start[ny, nx] < 0.0:
                        self.dist_start[ny, nx] = d + 1.0
                        qs.append((nx, ny, d + 1.0))

        self.start_to_goal_dist = max(1.0, float(self.dist_field[sy, sx]))
        self.max_corridor_depth = max(1.0, float(np.max(self.dist_start)))

    def get_progress(self, x: float, y: float) -> float:
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        if 0 <= gx < self.cols and 0 <= gy < self.rows:
            dg = self.dist_field[gy, gx]
            ds = self.dist_start[gy, gx]
            p_goal = (self.start_to_goal_dist - dg) / self.start_to_goal_dist if dg >= 0.0 else 0.0
            p_exp = (ds / self.max_corridor_depth) * 0.85 if ds >= 0.0 else 0.0
            return max(0.0, min(1.0, max(p_goal, p_exp)))
        return 0.0

    def check_collision(self, x: float, y: float, radius: float = 6.0) -> bool:
        if x - radius <= 0.0 or x + radius >= self.width or y - radius <= 0.0 or y + radius >= self.height:
            return True

        min_gx = max(0, int((x - radius) / self.cell_size))
        max_gx = min(self.cols - 1, int((x + radius) / self.cell_size))
        min_gy = max(0, int((y - radius) / self.cell_size))
        max_gy = min(self.rows - 1, int((y + radius) / self.cell_size))

        r2 = radius * radius
        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                if self.grid[gy, gx] == 1:
                    bx1 = gx * self.cell_size
                    by1 = gy * self.cell_size
                    bx2 = bx1 + self.cell_size
                    by2 = by1 + self.cell_size
                    near_x = max(bx1, min(x, bx2))
                    near_y = max(by1, min(y, by2))
                    dx = x - near_x
                    dy = y - near_y
                    if (dx * dx + dy * dy) < r2:
                        return True
        return False

    def raycast(self, ox: float, oy: float, angle: float, max_range: float = 240.0) -> float:
        dx = math.cos(angle)
        dy = math.sin(angle)
        cs = self.cell_size

        gx = int(ox / cs)
        gy = int(oy / cs)

        step_x = 1 if dx > 0.0 else -1
        step_y = 1 if dy > 0.0 else -1

        t_delta_x = abs(cs / dx) if abs(dx) > 1e-7 else 1e9
        t_delta_y = abs(cs / dy) if abs(dy) > 1e-7 else 1e9

        if dx > 0.0:
            t_max_x = ((gx + 1) * cs - ox) / dx
        else:
            t_max_x = (ox - gx * cs) / -dx if abs(dx) > 1e-7 else 1e9

        if dy > 0.0:
            t_max_y = ((gy + 1) * cs - oy) / dy
        else:
            t_max_y = (oy - gy * cs) / -dy if abs(dy) > 1e-7 else 1e9

        dist = 0.0
        while dist < max_range:
            if 0 <= gx < self.cols and 0 <= gy < self.rows:
                if self.grid[gy, gx] == 1:
                    return min(dist, max_range)
            else:
                return min(dist, max_range)

            if t_max_x < t_max_y:
                dist = t_max_x
                t_max_x += t_delta_x
                gx += step_x
            else:
                dist = t_max_y
                t_max_y += t_delta_y
                gy += step_y

        return max_range

    def _render_wall_cache(self) -> pygame.Surface:
        surf = pygame.Surface((int(self.width), int(self.height)), pygame.SRCALPHA)
        cs = self.cell_size
        ext_h = 7.0
        shadow_off = 5.0

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    x = c * cs
                    y = r * cs
                    pygame.draw.rect(surf, (COLOR_WALL_SHADOW[0], COLOR_WALL_SHADOW[1], COLOR_WALL_SHADOW[2], 120),
                                     (x + shadow_off, y + shadow_off, cs, cs))

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    x = c * cs
                    y = r * cs

                    if r + 1 < self.rows and self.grid[r + 1, c] == 0:
                        pygame.draw.rect(surf, COLOR_WALL_SIDE_S, (x, y + cs, cs, ext_h))

                    if c + 1 < self.cols and self.grid[r, c + 1] == 0:
                        pygame.draw.rect(surf, COLOR_WALL_SIDE_E, (x + cs, y, ext_h, cs))

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    x = c * cs
                    y = r * cs
                    pygame.draw.rect(surf, COLOR_WALL_TOP, (x, y, cs, cs))
                    pygame.draw.rect(surf, COLOR_WALL_BORDER, (x, y, cs, cs), 1)

        return surf

    def draw(self, surface: pygame.Surface) -> None:
        if self._cached_wall_surface is None or self._cached_wall_surface.get_size() != (int(self.width), int(self.height)):
            self._cached_wall_surface = self._render_wall_cache()
        surface.blit(self._cached_wall_surface, (0, 0))

        cs = self.cell_size

        sx, sy = int(self.start_pos[0]), int(self.start_pos[1])
        pygame.draw.circle(surface, (COLOR_START_ZONE[0], COLOR_START_ZONE[1], COLOR_START_ZONE[2], 60), (sx, sy), int(cs * 1.2), 2)
        pygame.draw.circle(surface, COLOR_START_ZONE, (sx, sy), 5)

        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        pulse = (math.sin(time.time() * 6.0) + 1.0) * 0.5
        g_rad = int(cs * (1.1 + 0.3 * pulse))
        pygame.draw.circle(surface, (COLOR_GOAL_ZONE[0], COLOR_GOAL_ZONE[1], COLOR_GOAL_ZONE[2], 80), (gx, gy), g_rad, 2)
        pygame.draw.circle(surface, COLOR_GOAL_GOLD, (gx, gy), int(cs * 0.5), 2)
        pygame.draw.circle(surface, (255, 255, 255), (gx, gy), 4)


class MazeDrone:
    def __init__(self, x: float, y: float, heading: float, controller: Optional[BatNeuralNet] = None, speed: float = 85.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.speed = float(speed)
        self.controller = controller if controller is not None else BatNeuralNet()
        self.radius = 6.0
        self.max_range = 240.0
        self.turning_rate_scale = 5.5

        self.is_alive = True
        self.collided = False
        self.reached_goal = False
        self.time_alive_steps = 0
        self.distance_traveled = 0.0
        self.max_progress = 0.0
        self.fitness = 0.0

        self.stagnant_steps = 0
        self.last_progress_check = 0.0

        self.ping_freq = 10.0
        self.recent_ping_freq = 10.0
        self.ping_timer = 0.0
        self.urgency = 0.0
        self.delta_theta = 0.0

        self.last_dists = [self.max_range] * 7
        self.wavefronts: List[float] = []

    def reset(self, x: float, y: float, heading: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.is_alive = True
        self.collided = False
        self.reached_goal = False
        self.time_alive_steps = 0
        self.distance_traveled = 0.0
        self.max_progress = 0.0
        self.fitness = 0.0
        self.stagnant_steps = 0
        self.last_progress_check = 0.0
        self.ping_freq = 10.0
        self.recent_ping_freq = 10.0
        self.ping_timer = 0.0
        self.urgency = 0.0
        self.delta_theta = 0.0
        self.last_dists = [self.max_range] * 7
        self.wavefronts.clear()
        if hasattr(self.controller, 'reset_state'):
            self.controller.reset_state()

    def step(self, dt: float, maze: Maze) -> None:
        if not self.is_alive:
            return

        self.time_alive_steps += 1

        self.ping_timer += dt
        ping_period = 1.0 / max(10.0, self.recent_ping_freq)

        if self.ping_timer >= ping_period:
            self.ping_timer = 0.0
            self.wavefronts.append(0.0)

            ray_angles = (-math.pi * 0.50, -math.pi * 0.33, -math.pi * 0.17, 0.0, math.pi * 0.17, math.pi * 0.33, math.pi * 0.50)
            dists = [maze.raycast(self.x, self.y, self.heading + off, self.max_range) for off in ray_angles]
            self.last_dists = dists

            inv_max = 1.0 / self.max_range
            r_norm = [d * inv_max for d in dists]

            dx = maze.goal_pos[0] - self.x
            dy = maze.goal_pos[1] - self.y
            target_ang = math.atan2(dy, dx)
            bearing = (target_ang - self.heading + math.pi) % (2.0 * math.pi) - math.pi
            norm_bearing = bearing / math.pi
            norm_v = self.speed / 85.0

            raw_in = np.array(r_norm + [norm_bearing, norm_v], dtype=np.float64)
            out = self.controller.forward(raw_in)

            self.delta_theta = float(out[0])
            self.urgency = float(out[1])
            throttle = float(out[2]) if len(out) > 2 else 0.5
            self.speed = 40.0 + 45.0 * throttle
            self.recent_ping_freq = 10.0 + 140.0 * self.urgency

        self.heading = (self.heading + self.delta_theta * self.turning_rate_scale * dt) % (2.0 * math.pi)
        step_dist = self.speed * dt
        self.x += step_dist * math.cos(self.heading)
        self.y += step_dist * math.sin(self.heading)
        self.distance_traveled += step_dist

        prog = maze.get_progress(self.x, self.y)
        if prog > self.max_progress:
            self.max_progress = prog

        if self.max_progress > self.last_progress_check + 0.005:
            self.last_progress_check = self.max_progress
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
            if self.stagnant_steps > 180:
                self.is_alive = False
                self.collided = True
                return

        gdx = self.x - maze.goal_pos[0]
        gdy = self.y - maze.goal_pos[1]
        if (gdx * gdx + gdy * gdy) <= (maze.goal_radius * maze.goal_radius):
            self.reached_goal = True
            self.is_alive = False
            return

        if maze.check_collision(self.x, self.y, self.radius):
            self.is_alive = False
            self.collided = True

        new_waves = []
        for w in self.wavefronts:
            nw = w + 343.0 * dt
            if nw < self.max_range:
                new_waves.append(nw)
        self.wavefronts = new_waves

    def draw(self, surface: pygame.Surface, draw_waves: bool = False, draw_rays: bool = False, is_champ: bool = False) -> None:
        ix, iy = int(round(self.x)), int(round(self.y))

        if draw_waves and self.is_alive:
            for w in self.wavefronts:
                alpha = max(0, int(130 * (1.0 - w / self.max_range)))
                wave_surf = pygame.Surface((int(w * 2 + 4), int(w * 2 + 4)), pygame.SRCALPHA)
                pygame.draw.circle(wave_surf, (COLOR_SWARM_GREEN[0], COLOR_SWARM_GREEN[1], COLOR_SWARM_GREEN[2], alpha),
                                   (int(w + 2), int(w + 2)), int(w), 1)
                surface.blit(wave_surf, (ix - int(w + 2), iy - int(w + 2)))

        if draw_rays and self.is_alive:
            ray_offsets = (-math.pi * 0.50, -math.pi * 0.33, -math.pi * 0.17, 0.0, math.pi * 0.17, math.pi * 0.33, math.pi * 0.50)
            for d, off in zip(self.last_dists, ray_offsets):
                ang = self.heading + off
                ex = self.x + d * math.cos(ang)
                ey = self.y + d * math.sin(ang)
                r_col = COLOR_RAY_GREEN if d >= self.max_range * 0.4 else (255, 160, 40)
                width = 2 if is_champ else 1
                pygame.draw.line(surface, r_col, (ix, iy), (int(ex), int(ey)), width)
                if d < self.max_range:
                    pygame.draw.circle(surface, COLOR_RAY_HIT, (int(ex), int(ey)), 2)

        if self.is_alive:
            r = int(self.radius)
            if is_champ:
                pygame.draw.circle(surface, (255, 255, 255), (ix, iy), r + 4, 1)

            pygame.draw.circle(surface, COLOR_SWARM_GREEN, (ix, iy), r)
            pygame.draw.circle(surface, COLOR_SWARM_CORE, (ix, iy), max(2, r - 3))

            nx = ix + int(round((r + 4) * math.cos(self.heading)))
            ny = iy + int(round((r + 4) * math.sin(self.heading)))
            pygame.draw.line(surface, (255, 255, 255), (ix, iy), (nx, ny), 2)
        else:
            if self.collided:
                pygame.draw.circle(surface, COLOR_SWARM_DEAD, (ix, iy), int(self.radius - 2))


class MazePopulation:
    def __init__(self, size: int = 150, max_steps: int = 2500, seed: Optional[int] = None) -> None:
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.generation = 0
        self.stagnant_count = 0
        self.all_time_best_progress = 0.0
        self.all_time_best_controller: Optional[BatNeuralNet] = None
        self.rng = np.random.default_rng(seed)
        self.drones: List[MazeDrone] = []
        self._init_drones()

    def _init_drones(self) -> None:
        self.drones.clear()
        for _ in range(self.size):
            net = BatNeuralNet(rng=self.rng)
            flat = net.get_flat_weights()
            flat += self.rng.normal(0.0, 0.05, size=flat.shape)
            net.set_flat_weights(flat)
            d = MazeDrone(x=0.0, y=0.0, heading=0.0, controller=net)
            self.drones.append(d)
        self._sync_vectors()

    def _sync_vectors(self) -> None:
        self.W1 = np.zeros((self.size, 9, 32), dtype=np.float64)
        self.W_rec = np.zeros((self.size, 32, 32), dtype=np.float64)
        self.B1 = np.zeros((self.size, 1, 32), dtype=np.float64)
        self.W2 = np.zeros((self.size, 32, 32), dtype=np.float64)
        self.B2 = np.zeros((self.size, 1, 32), dtype=np.float64)
        self.W3 = np.zeros((self.size, 32, 3), dtype=np.float64)
        self.B3 = np.zeros((self.size, 1, 3), dtype=np.float64)
        self.W_skip = np.zeros((self.size, 9, 3), dtype=np.float64)
        self.H_states = np.zeros((self.size, 1, 32), dtype=np.float64)
        self.raw_inputs = np.zeros((self.size, 1, 9), dtype=np.float64)
        for i, d in enumerate(self.drones):
            c = d.controller
            self.W1[i] = c.w1
            self.W_rec[i] = c.w_rec
            self.B1[i, 0] = c.b1
            self.W2[i] = c.w2
            self.B2[i, 0] = c.b2
            self.W3[i] = c.w3
            self.B3[i, 0] = c.b3
            self.W_skip[i] = c.w_skip
            self.H_states[i, 0] = c.h_state

    def reset_at_maze(self, maze: Maze) -> None:
        self.step_count = 0
        self.H_states.fill(0.0)
        sx, sy = maze.start_pos
        for drone in self.drones:
            px = sx + float(self.rng.uniform(-6.0, 6.0))
            py = sy + float(self.rng.uniform(-6.0, 6.0))
            head = float(self.rng.uniform(-0.15, 0.15))
            drone.reset(px, py, head)

    def step(self, dt: float, maze: Maze) -> bool:
        self.step_count += 1
        alive_indices = [i for i, d in enumerate(self.drones) if d.is_alive]
        if not alive_indices:
            return True

        ping_indices = []
        inv_max_range = 1.0 / 240.0
        gx, gy = maze.goal_pos
        RAY_OFFSETS = (-math.pi * 0.50, -math.pi * 0.33, -math.pi * 0.17, 0.0, math.pi * 0.17, math.pi * 0.33, math.pi * 0.50)

        for i in alive_indices:
            d = self.drones[i]
            d.time_alive_steps += 1
            d.ping_timer += dt
            period = 1.0 / max(10.0, d.recent_ping_freq)
            if d.ping_timer >= period:
                d.ping_timer = 0.0
                ping_indices.append(i)
                h = d.heading
                r0 = maze.raycast(d.x, d.y, h + RAY_OFFSETS[0], 240.0)
                r1 = maze.raycast(d.x, d.y, h + RAY_OFFSETS[1], 240.0)
                r2 = maze.raycast(d.x, d.y, h + RAY_OFFSETS[2], 240.0)
                r3 = maze.raycast(d.x, d.y, h, 240.0)
                r4 = maze.raycast(d.x, d.y, h + RAY_OFFSETS[4], 240.0)
                r5 = maze.raycast(d.x, d.y, h + RAY_OFFSETS[5], 240.0)
                r6 = maze.raycast(d.x, d.y, h + RAY_OFFSETS[6], 240.0)
                d.last_dists = [r0, r1, r2, r3, r4, r5, r6]

                dx = gx - d.x
                dy = gy - d.y
                target_ang = math.atan2(dy, dx)
                bearing = (target_ang - h + math.pi) % (2.0 * math.pi) - math.pi

                self.raw_inputs[i, 0, 0] = r0 * inv_max_range
                self.raw_inputs[i, 0, 1] = r1 * inv_max_range
                self.raw_inputs[i, 0, 2] = r2 * inv_max_range
                self.raw_inputs[i, 0, 3] = r3 * inv_max_range
                self.raw_inputs[i, 0, 4] = r4 * inv_max_range
                self.raw_inputs[i, 0, 5] = r5 * inv_max_range
                self.raw_inputs[i, 0, 6] = r6 * inv_max_range
                self.raw_inputs[i, 0, 7] = bearing / math.pi
                self.raw_inputs[i, 0, 8] = d.speed / 85.0

        if ping_indices:
            p_idx = np.array(ping_indices, dtype=np.int32)
            X = self.raw_inputs[p_idx]
            H_prev = self.H_states[p_idx]

            pre_h1 = np.matmul(X, self.W1[p_idx]) + np.matmul(H_prev, self.W_rec[p_idx]) + self.B1[p_idx]
            h1 = np.where(pre_h1 > 0.0, pre_h1, 0.05 * pre_h1)
            self.H_states[p_idx] = h1

            pre_h2 = np.matmul(h1, self.W2[p_idx]) + self.B2[p_idx]
            h2 = np.where(pre_h2 > 0.0, pre_h2, 0.05 * pre_h2)

            out = np.matmul(h2, self.W3[p_idx]) + self.B3[p_idx] + np.matmul(X, self.W_skip[p_idx])

            dthetas = np.tanh(out[:, 0, 0])
            urgs = 1.0 / (1.0 + np.exp(-np.clip(out[:, 0, 1], -60.0, 60.0)))
            throttles = 1.0 / (1.0 + np.exp(-np.clip(out[:, 0, 2], -60.0, 60.0)))

            for k, i in enumerate(ping_indices):
                d = self.drones[i]
                d.delta_theta = float(dthetas[k])
                d.urgency = float(urgs[k])
                d.recent_ping_freq = 10.0 + 140.0 * d.urgency
                d.speed = 40.0 + 45.0 * float(throttles[k])
                d.controller.h_state = h1[k, 0].copy()
                d.controller.last_inputs = X[k, 0].copy()
                d.controller.last_h1 = h1[k, 0].copy()
                d.controller.last_h2 = h2[k, 0].copy()
                d.controller.last_outputs = np.array([dthetas[k], urgs[k], throttles[k]], dtype=np.float64)

        g_rad2 = maze.goal_radius * maze.goal_radius
        for i in alive_indices:
            d = self.drones[i]
            d.heading = (d.heading + d.delta_theta * d.turning_rate_scale * dt) % (2.0 * math.pi)
            step_dist = d.speed * dt
            d.x += step_dist * math.cos(d.heading)
            d.y += step_dist * math.sin(d.heading)
            d.distance_traveled += step_dist

            prog = maze.get_progress(d.x, d.y)
            if prog > d.max_progress:
                d.max_progress = prog

            if d.max_progress > d.last_progress_check + 0.005:
                d.last_progress_check = d.max_progress
                d.stagnant_steps = 0
            else:
                d.stagnant_steps += 1
                if d.stagnant_steps > 180:
                    d.is_alive = False
                    d.collided = True
                    continue

            gdx = d.x - gx
            gdy = d.y - gy
            if (gdx * gdx + gdy * gdy) <= g_rad2:
                d.reached_goal = True
                d.is_alive = False
                continue

            if maze.check_collision(d.x, d.y, d.radius):
                d.is_alive = False
                d.collided = True

        return not any(self.drones[i].is_alive for i in alive_indices)

    def evaluate_fitness(self) -> None:
        for d in self.drones:
            prog_score = (d.max_progress ** 1.3) * 4000.0
            time_score = min(d.time_alive_steps * 0.08, 40.0)
            goal_bonus = 4000.0 if d.reached_goal else 0.0
            coll_penalty = 30.0 if (d.collided and not d.reached_goal) else 0.0
            d.fitness = float(prog_score + time_score + goal_bonus - coll_penalty)

    def reproduce(self, maze: Maze) -> None:
        self.evaluate_fitness()
        sorted_d = sorted(self.drones, key=lambda d: d.fitness, reverse=True)
        pool_size = max(4, int(self.size * 0.20))
        parents = sorted_d[:pool_size]

        best_drone = sorted_d[0]
        if best_drone.max_progress > self.all_time_best_progress:
            self.all_time_best_progress = best_drone.max_progress
            self.all_time_best_controller = best_drone.controller.copy()
            self.stagnant_count = 0
        else:
            self.stagnant_count += 1

        next_gen = self.generation + 1
        if self.stagnant_count >= 5:
            sigma = 0.08
            mutation_rate = 0.12
        else:
            sigma = 0.04
            mutation_rate = 0.07

        new_drones: List[MazeDrone] = []
        if self.all_time_best_controller is not None:
            new_drones.append(MazeDrone(x=0.0, y=0.0, heading=0.0, controller=self.all_time_best_controller.copy()))
        new_drones.append(MazeDrone(x=0.0, y=0.0, heading=0.0, controller=sorted_d[0].controller.copy()))

        champ_controller = self.all_time_best_controller or sorted_d[0].controller
        for _ in range(int(self.size * 0.20)):
            child_net = champ_controller.copy()
            child_net.mutate(mutation_rate=0.05, sigma=0.03, rng=self.rng)
            new_drones.append(MazeDrone(x=0.0, y=0.0, heading=0.0, controller=child_net))

        while len(new_drones) < self.size:
            pa = self.rng.choice(parents)
            pb = self.rng.choice(parents)
            child_net = BatNeuralNet.uniform_crossover(pa.controller, pb.controller, rng=self.rng)
            child_net.mutate(mutation_rate=mutation_rate, sigma=sigma, rng=self.rng)
            new_drones.append(MazeDrone(x=0.0, y=0.0, heading=0.0, controller=child_net))

        self.drones = new_drones
        self.generation = next_gen
        self._sync_vectors()
        self.reset_at_maze(maze)

    def get_best_drone(self) -> MazeDrone:
        return max(self.drones, key=lambda d: (d.max_progress, d.distance_traveled))

    def save_best(self, filepath: str) -> None:
        best = self.get_best_drone()
        best.controller.save(filepath)

    def load_champion(self, filepath: str) -> BatNeuralNet:
        return BatNeuralNet.load(filepath, rng=self.rng)


def run_maze_simulation(
    cols: int = 35,
    rows: int = 22,
    cell_size: float = 40.0,
    pop_size: int = 150,
    max_steps: int = 2500,
    steps_limit: Optional[int] = None,
    weights_path: Optional[str] = None,
    neural_window: bool = False,
    hud_window: bool = False,
    headless: bool = False,
) -> None:
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    pygame.font.init()

    win_w = int(cols * cell_size)
    win_h = int(rows * cell_size)

    try:
        screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Bat Echolocation - Autonomous 3D Overhead Maze Navigation (60 FPS)")
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14, bold=True)
    small_font = pygame.font.SysFont("consolas", 12)

    maze = Maze(cols=cols, rows=rows, cell_size=cell_size)
    pop = MazePopulation(size=pop_size, max_steps=max_steps)
    pop.reset_at_maze(maze)

    resolved_weights = None
    if weights_path and os.path.exists(weights_path):
        resolved_weights = weights_path

    champ = None
    if resolved_weights:
        try:
            champ = pop.load_champion(resolved_weights)
            for d in pop.drones:
                d.controller = champ.copy()
            print(f"[Maze] Loaded weights from {resolved_weights}")
        except Exception:
            pass

    demo_mode = False
    paused = False
    fast_forward = False
    show_waves = False
    show_hud = not hud_window
    ray_modes = ["best", "all", "none"]
    ray_mode_idx = 0

    ipc_writer = ActivationWriter()
    neural_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "neural_window.py"))
    hud_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "maze_hud_window.py"))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    hud_process = None
    if hud_window and not headless:
        try:
            import subprocess
            hud_process = subprocess.Popen([sys.executable, hud_script], cwd=repo_root)
        except Exception:
            pass

    neural_process = None
    if neural_window and not headless:
        try:
            import subprocess
            neural_process = subprocess.Popen([sys.executable, neural_script], cwd=repo_root)
        except Exception:
            pass

    total_frames = 0
    running = True
    while running:
        dt = 1.0 / 60.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    fast_forward = not fast_forward
                elif event.key == pygame.K_e:
                    demo_mode = not demo_mode
                    save_path = "best_weights_maze.npz"
                    if demo_mode:
                        try:
                            pop.save_best(save_path)
                            champ = pop.load_champion(save_path)
                        except Exception:
                            pass
                        pop.reset_at_maze(maze)
                        if champ is not None:
                            for d in pop.drones:
                                d.controller = champ.copy()
                    else:
                        pop.reset_at_maze(maze)
                elif event.key == pygame.K_s:
                    save_path = "best_weights_maze.npz"
                    try:
                        pop.save_best(save_path)
                        champ = pop.load_champion(save_path)
                        print(f"[Maze] Saved best weights to {save_path}")
                    except Exception as e:
                        print(f"[Maze] Error saving: {e}")
                elif event.key == pygame.K_r:
                    maze.generate()
                    pop.all_time_best_progress = 0.0
                    pop.stagnant_count = 0
                    pop.reset_at_maze(maze)
                    if demo_mode and champ is not None:
                        for d in pop.drones:
                            d.controller = champ.copy()
                elif event.key == pygame.K_w:
                    show_waves = not show_waves
                elif event.key == pygame.K_d:
                    ray_mode_idx = (ray_mode_idx + 1) % len(ray_modes)
                elif event.key == pygame.K_h:
                    show_hud = not show_hud
                elif event.key == pygame.K_o:
                    if hud_process is None or hud_process.poll() is not None:
                        try:
                            import subprocess
                            hud_process = subprocess.Popen([sys.executable, hud_script], cwd=repo_root)
                        except Exception:
                            pass
                elif event.key == pygame.K_p:
                    try:
                        import subprocess
                        neural_process = subprocess.Popen([sys.executable, neural_script], cwd=repo_root)
                    except Exception:
                        pass
                elif event.key == pygame.K_UP:
                    for d in pop.drones:
                        d.speed = min(140.0, d.speed + 15.0)
                elif event.key == pygame.K_DOWN:
                    for d in pop.drones:
                        d.speed = max(40.0, d.speed - 15.0)

        if not paused:
            steps_per_frame = 10 if fast_forward else 1
            for _ in range(steps_per_frame):
                done = pop.step(dt, maze)
                if done:
                    if demo_mode and champ is not None:
                        pop.reset_at_maze(maze)
                        for d in pop.drones:
                            d.controller = champ.copy()
                    else:
                        best_curr = pop.get_best_drone()
                        if best_curr.max_progress > 0.20:
                            save_path = "best_weights_maze.npz"
                            try:
                                pop.save_best(save_path)
                                champ = pop.load_champion(save_path)
                            except Exception:
                                pass
                        pop.reproduce(maze)
                    break
            total_frames += steps_per_frame
            if steps_limit and total_frames >= steps_limit:
                running = False

        screen.fill(COLOR_FLOOR)
        maze.draw(screen)

        best_drone = pop.get_best_drone()
        ray_mode = ray_modes[ray_mode_idx]

        if fast_forward:
            for drone in pop.drones:
                if drone.is_alive:
                    if drone is best_drone:
                        drone.draw(screen, draw_waves=False, draw_rays=True, is_champ=True)
                    else:
                        ix, iy = int(round(drone.x)), int(round(drone.y))
                        pygame.draw.circle(screen, COLOR_SWARM_GREEN, (ix, iy), int(drone.radius))
        else:
            for drone in pop.drones:
                is_best = (drone is best_drone)
                draw_r = (ray_mode == "all") or (ray_mode == "best" and is_best)
                drone.draw(screen, draw_waves=show_waves, draw_rays=draw_r, is_champ=is_best)

        alive_count = sum(1 for d in pop.drones if d.is_alive)
        reached_count = sum(1 for d in pop.drones if d.reached_goal)
        best_prog = best_drone.max_progress * 100.0 if best_drone else 0.0
        fps = clock.get_fps() if not fast_forward else 1000.0

        telem_data = {
            "gen": pop.generation,
            "mode": 1 if demo_mode else 0,
            "alive": alive_count,
            "pop_size": pop.size,
            "reached": reached_count,
            "progress": best_prog,
            "step": pop.step_count,
            "max_steps": pop.max_steps,
            "paused": 1 if paused else 0,
            "fast_forward": 1 if fast_forward else 0,
            "fps": fps,
            "ray_mode": ray_mode_idx,
        }

        if best_drone is not None and getattr(best_drone, "controller", None) is not None:
            ctrl = best_drone.controller
            if hasattr(ctrl, "get_last_activations"):
                acts = ctrl.get_last_activations()
                ipc_writer.write(
                    acts,
                    ping_freq=getattr(best_drone, "recent_ping_freq", 10.0),
                    speed=getattr(best_drone, "speed", 85.0),
                    heading=getattr(best_drone, "heading", 0.0),
                    telemetry=telem_data,
                )

        if show_hud:
            card_w, card_h = 360, 210
            card_x = 15
            card_y = win_h - card_h - 15
            hud_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            pygame.draw.rect(hud_surf, (15, 20, 30, 230), (0, 0, card_w, card_h), border_radius=8)
            pygame.draw.rect(hud_surf, (0, 240, 160, 240), (0, 0, card_w, card_h), 1, border_radius=8)

            t_title = font.render("AUTONOMOUS SWARM MAZE NAVIGATION", True, (0, 240, 160))
            hud_surf.blit(t_title, (14, 10))

            mode_str = "CHAMPION DEMO [E]" if demo_mode else "EVOLVING SWARM [E]"
            lines = [
                f"Generation   : {pop.generation} ({mode_str})",
                f"Swarm Alive  : {alive_count} / {pop.size} ({alive_count/pop.size*100:.1f}%)",
                f"Exit Reached : {reached_count} drones",
                f"Best Progress: {best_prog:.1f}% along maze path",
                f"Speed / Pulse: {best_drone.speed:.1f} px/s | {best_drone.recent_ping_freq:.1f} Hz",
                f"Step Count   : {pop.step_count} (unlimited)",
            ]

            yo = 34
            for l in lines:
                txt = small_font.render(l, True, (240, 245, 255))
                hud_surf.blit(txt, (14, yo))
                yo += 18

            bar_x, bar_y, bar_w, bar_h = 14, yo + 4, card_w - 28, 8
            pygame.draw.rect(hud_surf, (30, 40, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            fill_w = int(bar_w * (best_prog / 100.0))
            if fill_w > 0:
                bar_color = (0, 240, 160) if best_prog < 95.0 else (255, 215, 0)
                pygame.draw.rect(hud_surf, bar_color, (bar_x, bar_y, fill_w, bar_h), border_radius=3)

            status_str = "PAUSED [Space]" if paused else ("FAST-FORWARD [F]" if fast_forward else f"RUNNING ({fps:.0f} FPS)")
            t_sub = small_font.render(f"{status_str} | Rays: {ray_mode.upper()} [D] | HUD: [H]/[O]", True, (160, 175, 195))
            hud_surf.blit(t_sub, (14, yo + 18))

            screen.blit(hud_surf, (card_x, card_y))

        pygame.display.flip()
        clock.tick(60)

    if hud_process is not None:
        try:
            hud_process.terminate()
        except Exception:
            pass
    if neural_process is not None:
        try:
            neural_process.terminate()
        except Exception:
            pass

    ipc_writer.close()
    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Swarm Maze Navigation")
    parser.add_argument("--cols", type=int, default=35, help="Maze columns")
    parser.add_argument("--rows", type=int, default=22, help="Maze rows")
    parser.add_argument("--cell-size", type=float, default=40.0, help="Grid cell size")
    parser.add_argument("--pop-size", type=int, default=150, help="Swarm population size")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max steps per episode")
    parser.add_argument("--weights", type=str, default=None, help="Weights file to load")
    parser.add_argument("--neural-window", action="store_true", help="Launch companion neural visualizer window")
    parser.add_argument("--hud-window", action="store_true", help="Launch companion HUD telemetry window")
    parser.add_argument("--steps", type=int, default=None, help="Stop after N total frames")
    parser.add_argument("--headless", action="store_true", help="Run with dummy video driver")
    args = parser.parse_args()

    run_maze_simulation(
        cols=args.cols,
        rows=args.rows,
        cell_size=args.cell_size,
        pop_size=args.pop_size,
        max_steps=args.max_steps,
        steps_limit=args.steps,
        weights_path=args.weights,
        neural_window=args.neural_window,
        hud_window=args.hud_window,
        headless=args.headless,
    )
