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

LOG_FILE_PATH: str = os.path.join(os.path.dirname(__file__), "training.log")


def sim_log(msg: str) -> None:
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{now_str}] {msg}"
    try:
        print(full_msg, flush=True)
    except Exception:
        clean_msg = full_msg.encode("ascii", errors="replace").decode("ascii")
        try:
            print(clean_msg, flush=True)
        except Exception:
            pass
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")
    except Exception:
        pass


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

# Pre-rotated Micro-Drone Sprite Caches (360 discrete headings for 0-overhead blitting)
DRONE_SPRITES_INITIALIZED: bool = False
SPRITE_SWARM_ROT: List[pygame.Surface] = []
SPRITE_CHAMP_ROT: List[pygame.Surface] = []
SPRITE_DEAD_ROT: List[pygame.Surface] = []
SHADOW_SURF: Optional[pygame.Surface] = None


def init_drone_sprites() -> None:
    global DRONE_SPRITES_INITIALIZED, SPRITE_SWARM_ROT, SPRITE_CHAMP_ROT, SPRITE_DEAD_ROT, SHADOW_SURF
    if DRONE_SPRITES_INITIALIZED:
        return
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        assets_dir = os.path.join(base_dir, "assets")

        swarm_p = os.path.join(assets_dir, "drone_swarm.png")
        champ_p = os.path.join(assets_dir, "drone_champ.png")
        dead_p = os.path.join(assets_dir, "drone_dead.png")

        if not os.path.exists(swarm_p):
            swarm_p = os.path.join(assets_dir, "drone_swarm.svg")
        if not os.path.exists(champ_p):
            champ_p = os.path.join(assets_dir, "drone_champ.svg")
        if not os.path.exists(dead_p):
            dead_p = os.path.join(assets_dir, "drone_dead.svg")

        s_swarm = pygame.image.load(swarm_p)
        s_champ = pygame.image.load(champ_p)
        s_dead = pygame.image.load(dead_p)

        target_size = (13, 13)
        s_swarm = pygame.transform.smoothscale(s_swarm, target_size)
        s_champ = pygame.transform.smoothscale(s_champ, target_size)
        s_dead = pygame.transform.smoothscale(s_dead, target_size)

        SPRITE_SWARM_ROT.clear()
        SPRITE_CHAMP_ROT.clear()
        SPRITE_DEAD_ROT.clear()

        for deg in range(360):
            SPRITE_SWARM_ROT.append(pygame.transform.rotate(s_swarm, -deg))
            SPRITE_CHAMP_ROT.append(pygame.transform.rotate(s_champ, -deg))
            SPRITE_DEAD_ROT.append(pygame.transform.rotate(s_dead, -deg))

        SHADOW_SURF = pygame.Surface((15, 9), pygame.SRCALPHA)
        pygame.draw.ellipse(SHADOW_SURF, (0, 0, 0, 90), (0, 0, 15, 9))

        DRONE_SPRITES_INITIALIZED = True
    except Exception:
        DRONE_SPRITES_INITIALIZED = False


class Maze:
    def __init__(self, cols: int = 23, rows: int = 23, cell_size: float = 38.0, seed: Optional[int] = None) -> None:
        c = int(cols)
        r = int(rows)
        self.cols = c if c % 2 == 1 else max(7, c - 1)
        self.rows = r if r % 2 == 1 else max(7, r - 1)
        self.cell_size = float(cell_size)
        self.width = self.cols * self.cell_size
        self.height = self.rows * self.cell_size
        self.grid = np.ones((self.rows, self.cols), dtype=np.uint8)
        self.dist_field = np.full((self.rows, self.cols), -1.0, dtype=np.float64)
        self.dist_start = np.full((self.rows, self.cols), -1.0, dtype=np.float64)
        self.start_to_goal_dist = 1.0
        self.max_corridor_depth = 1.0
        self.start_pos = (2.5 * self.cell_size, 2.5 * self.cell_size)
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

        self.grid[1:4, 1:4] = 0
        self.grid[-3:-1, -3:-1] = 0
        self.grid[1, 4] = 0
        self.grid[2, 4] = 0
        self.grid[3, 4] = 0
        self.grid[4, 1] = 0
        self.grid[4, 2] = 0
        self.grid[4, 3] = 0

        self.start_pos = (2.5 * self.cell_size, 2.5 * self.cell_size)
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
            if dg >= 0.0:
                return max(0.0, min(1.0, (self.start_to_goal_dist - dg) / self.start_to_goal_dist))
        return 0.0

    def get_acoustic_beacon_angle(self, x: float, y: float) -> float:
        gx = int(x / self.cell_size)
        gy = int(y / self.cell_size)
        gx = max(0, min(self.cols - 1, gx))
        gy = max(0, min(self.rows - 1, gy))

        current_dg = self.dist_field[gy, gx]
        best_target_x = self.goal_pos[0]
        best_target_y = self.goal_pos[1]

        if current_dg > 0:
            best_dg = current_dg
            best_cell = None
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.grid[ny, nx] == 0 and self.dist_field[ny, nx] >= 0:
                        if self.dist_field[ny, nx] < best_dg:
                            best_dg = self.dist_field[ny, nx]
                            best_cell = (nx, ny)
            if best_cell is not None:
                best_target_x = (best_cell[0] + 0.5) * self.cell_size
                best_target_y = (best_cell[1] + 0.5) * self.cell_size

        return math.atan2(best_target_y - y, best_target_x - x)

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
    def __init__(
        self,
        x: float,
        y: float,
        heading: float,
        controller: Optional[BatNeuralNet] = None,
        speed: float = 85.0,
        drone_id: int = 0,
        corridor_bias: float = 0.0,
        speed_multiplier: float = 1.0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.speed = float(speed)
        self.drone_id = int(drone_id)
        self.corridor_bias = float(corridor_bias)
        self.speed_multiplier = float(speed_multiplier)
        self.controller = controller if controller is not None else BatNeuralNet()
        if hasattr(self.controller, 'reset_state'):
            self.controller.reset_state()
        self.radius = 6.0
        self.max_range = 240.0
        self.turning_rate_scale = 5.5
        self.trail: deque[Tuple[float, float]] = deque(maxlen=4)
        self.pos_history: deque[Tuple[float, float]] = deque(maxlen=30)
        self.cumulative_rotation: float = 0.0

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
        self.cumulative_rotation = 0.0
        self.pos_history.clear()
        self.ping_freq = 10.0
        self.recent_ping_freq = 10.0
        self.ping_timer = 0.0
        self.urgency = 0.0
        self.delta_theta = 0.0
        self.last_dists = [self.max_range] * 7
        self.wavefronts.clear()
        self.trail.clear()
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

            target_ang = maze.get_acoustic_beacon_angle(self.x, self.y)
            bearing = (target_ang - self.heading + math.pi) % (2.0 * math.pi) - math.pi
            norm_bearing = bearing / math.pi
            norm_v = self.speed / 85.0

            raw_in = np.array(r_norm + [norm_bearing, norm_v], dtype=np.float64)
            out = self.controller.forward(raw_in)

            target_steer = float(out[0]) + self.corridor_bias * 0.12
            self.delta_theta = 0.55 * self.delta_theta + 0.45 * target_steer
            self.urgency = float(out[1])
            throttle = float(out[2]) if len(out) > 2 else 0.5
            r2_val = dists[2] * inv_max
            r3_val = dists[3] * inv_max
            r4_val = dists[4] * inv_max
            fwd_dist = min(r3_val, r2_val * 1.15, r4_val * 1.15)
            brake = 0.30 + 0.70 * min(1.0, fwd_dist * 2.2)
            self.speed = (20.0 + 65.0 * throttle) * brake * self.speed_multiplier
            self.recent_ping_freq = 10.0 + 140.0 * self.urgency

        self.heading = (self.heading + self.delta_theta * self.turning_rate_scale * dt) % (2.0 * math.pi)
        step_dist = self.speed * dt
        self.x += step_dist * math.cos(self.heading)
        self.y += step_dist * math.sin(self.heading)
        self.distance_traveled += step_dist

        if self.time_alive_steps % 3 == 0:
            self.trail.append((self.x, self.y))

        prog = maze.get_progress(self.x, self.y)
        if prog > self.max_progress:
            self.max_progress = prog

        if self.max_progress > self.last_progress_check + 0.005:
            self.last_progress_check = self.max_progress
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1
            if self.stagnant_steps > 700:
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

        # 1. 3D Hovering Drop Shadow (ground depth separation)
        if not DRONE_SPRITES_INITIALIZED:
            init_drone_sprites()

        if self.is_alive and SHADOW_SURF is not None:
            surface.blit(SHADOW_SURF, (ix - 7 + 2, iy - 4 + 4))

        # 2. Leader-Only Precision Motion Spark (no swarm trail smudging)
        if is_champ and self.is_alive and len(self.trail) > 1:
            pts = list(self.trail)
            for k, (tx, ty) in enumerate(pts[:-1]):
                dot_r = 1 if k < 2 else 2
                pygame.draw.circle(surface, (255, 200, 50), (int(round(tx)), int(round(ty))), dot_r)

        # 2. Sonar Wavefronts
        if draw_waves and self.is_alive:
            for w in self.wavefronts:
                alpha = max(0, int(130 * (1.0 - w / self.max_range)))
                wave_surf = pygame.Surface((int(w * 2 + 4), int(w * 2 + 4)), pygame.SRCALPHA)
                pygame.draw.circle(wave_surf, (COLOR_SWARM_GREEN[0], COLOR_SWARM_GREEN[1], COLOR_SWARM_GREEN[2], alpha),
                                   (int(w + 2), int(w + 2)), int(w), 1)
                surface.blit(wave_surf, (ix - int(w + 2), iy - int(w + 2)))

        # 3. Sonar Echolocation Rays
        if draw_rays and self.is_alive:
            ray_offsets = (-math.pi * 0.50, -math.pi * 0.33, -math.pi * 0.17, 0.0, math.pi * 0.17, math.pi * 0.33, math.pi * 0.50)
            for d, off in zip(self.last_dists, ray_offsets):
                ang = self.heading + off
                ex = self.x + d * math.cos(ang)
                ey = self.y + d * math.sin(ang)
                r_col = (255, 210, 50) if is_champ else (COLOR_RAY_GREEN if d >= self.max_range * 0.4 else (255, 160, 40))
                width = 2 if is_champ else 1
                pygame.draw.line(surface, r_col, (ix, iy), (int(ex), int(ey)), width)
                if d < self.max_range:
                    pygame.draw.circle(surface, COLOR_RAY_HIT, (int(ex), int(ey)), 2)

        # 4. Rotated Quadcopter Micro-Drone Sprite
        if not DRONE_SPRITES_INITIALIZED:
            init_drone_sprites()

        angle_deg = int(round(-math.degrees(self.heading))) % 360

        if self.is_alive:
            if DRONE_SPRITES_INITIALIZED and SPRITE_SWARM_ROT:
                spr = SPRITE_CHAMP_ROT[angle_deg] if is_champ else SPRITE_SWARM_ROT[angle_deg]
                rect = spr.get_rect(center=(ix, iy))
                surface.blit(spr, rect)
            else:
                r = int(self.radius)
                pygame.draw.circle(surface, COLOR_SWARM_GREEN, (ix, iy), r)
                pygame.draw.circle(surface, COLOR_SWARM_CORE, (ix, iy), max(2, r - 3))

            if is_champ:
                pygame.draw.circle(surface, (255, 215, 0), (ix, iy), 12, 1)
        else:
            if self.collided:
                if DRONE_SPRITES_INITIALIZED and SPRITE_DEAD_ROT:
                    spr = SPRITE_DEAD_ROT[angle_deg]
                    rect = spr.get_rect(center=(ix, iy))
                    surface.blit(spr, rect)
                else:
                    pygame.draw.circle(surface, COLOR_SWARM_DEAD, (ix, iy), int(self.radius - 2))


class MazePopulation:
    def __init__(self, size: int = 80, max_steps: int = 2500, seed: Optional[int] = None, scratch: bool = True) -> None:
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.scratch = bool(scratch)
        self.step_count = 0
        self.generation = 0
        self.stagnant_count = 0
        self.all_time_best_progress = 0.0
        self.gen_start_best_progress = 0.0
        self.new_record_this_gen = False
        self.all_time_best_controller: Optional[BatNeuralNet] = None
        self.last_status_message: str = "INITIALIZING SWARM"
        self.rng = np.random.default_rng(seed)
        self.drones: List[MazeDrone] = []
        self._init_drones()

    def _init_drones(self) -> None:
        self.drones.clear()
        for i in range(self.size):
            net = BatNeuralNet(rng=self.rng)
            if self.scratch:
                net.init_scratch(rng=self.rng)
            else:
                flat = net.get_flat_weights()
                flat += self.rng.normal(0.0, 0.05, size=flat.shape)
                net.set_flat_weights(flat)
            cb = float(self.rng.uniform(-0.16, 0.16))
            sm = float(self.rng.uniform(0.75, 1.25))
            d = MazeDrone(x=0.0, y=0.0, heading=0.0, controller=net, drone_id=i, corridor_bias=cb, speed_multiplier=sm)
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
        self.coords = np.zeros((self.size, 3), dtype=np.float64)
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
            self.coords[i, 0] = d.x
            self.coords[i, 1] = d.y
            self.coords[i, 2] = d.heading

    def reset_at_maze(self, maze: Maze) -> None:
        self.step_count = 0
        self.H_states.fill(0.0)
        self.gen_start_best_progress = self.all_time_best_progress
        self.new_record_this_gen = False
        sx, sy = maze.start_pos
        cols = 10
        num_rows = (len(self.drones) + cols - 1) // cols
        spacing = 3.2
        for i, drone in enumerate(self.drones):
            r = i // cols
            c = i % cols
            ox = (c - (cols - 1) / 2.0) * spacing
            oy = (r - (num_rows - 1) / 2.0) * spacing
            px = sx + ox
            py = sy + oy
            # Guarantee 100% collision-free spawn:
            if maze.check_collision(px, py, drone.radius + 2.0):
                px, py = sx, sy
            head = float(self.rng.uniform(-0.15, 0.15))
            drone.reset(px, py, head)
            self.coords[i, 0] = px
            self.coords[i, 1] = py
            self.coords[i, 2] = head

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

                target_ang = maze.get_acoustic_beacon_angle(d.x, d.y)
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
                target_steer = float(dthetas[k]) + d.corridor_bias * 0.12
                d.delta_theta = 0.55 * d.delta_theta + 0.45 * target_steer
                d.urgency = float(urgs[k])
                d.recent_ping_freq = 10.0 + 140.0 * d.urgency
                r2_val = self.raw_inputs[i, 0, 2]
                r3_val = self.raw_inputs[i, 0, 3]
                r4_val = self.raw_inputs[i, 0, 4]
                fwd_dist = min(r3_val, r2_val * 1.15, r4_val * 1.15)
                brake = 0.30 + 0.70 * min(1.0, fwd_dist * 2.2)
                d.speed = (20.0 + 65.0 * float(throttles[k])) * brake * d.speed_multiplier
                d.controller.h_state = h1[k, 0].copy()
                d.controller.last_inputs = X[k, 0].copy()
                d.controller.last_h1 = h1[k, 0].copy()
                d.controller.last_h2 = h2[k, 0].copy()
                d.controller.last_outputs = np.array([dthetas[k], urgs[k], throttles[k]], dtype=np.float64)

        # 1. Anti-tailgating following-distance brake (wake turbulence avoidance)
        if len(alive_indices) > 1:
            idx_arr = np.array(alive_indices, dtype=np.int32)
            alive_xs = self.coords[idx_arr, 0]
            alive_ys = self.coords[idx_arr, 1]
            heads = self.coords[idx_arr, 2]

            dx_fwd = alive_xs[None, :] - alive_xs[:, None]
            dy_fwd = alive_ys[None, :] - alive_ys[:, None]
            d2_fwd = dx_fwd * dx_fwd + dy_fwd * dy_fwd
            np.fill_diagonal(d2_fwd, 1e9)

            cos_h = np.cos(heads)[:, None]
            sin_h = np.sin(heads)[:, None]
            forward_proj = dx_fwd * cos_h + dy_fwd * sin_h

            is_ahead = (forward_proj > 4.0) & (forward_proj < 24.0) & (d2_fwd < 24.0 * 24.0)
            has_lead = np.any(is_ahead, axis=1)
            for k, i in enumerate(alive_indices):
                if has_lead[k]:
                    self.drones[i].speed *= 0.88

        g_rad2 = maze.goal_radius * maze.goal_radius
        for i in alive_indices:
            d = self.drones[i]
            rot_delta = abs(d.delta_theta * d.turning_rate_scale * dt)
            d.heading = (d.heading + d.delta_theta * d.turning_rate_scale * dt) % (2.0 * math.pi)
            step_dist = d.speed * dt
            d.x += step_dist * math.cos(d.heading)
            d.y += step_dist * math.sin(d.heading)
            d.distance_traveled += step_dist
            d.cumulative_rotation += rot_delta
            self.coords[i, 0] = d.x
            self.coords[i, 1] = d.y
            self.coords[i, 2] = d.heading

            if d.time_alive_steps % 3 == 0:
                d.trail.append((d.x, d.y))

            # Anti-donut loop detector: kill if spinning in a tight circle without translation
            if d.time_alive_steps % 4 == 0:
                d.pos_history.append((d.x, d.y))
                if len(d.pos_history) >= 20:
                    old_x, old_y = d.pos_history[0]
                    disp = math.hypot(d.x - old_x, d.y - old_y)
                    if disp < 18.0 and d.cumulative_rotation > 10.5:
                        d.is_alive = False
                        d.collided = True
                        continue

            prog = maze.get_progress(d.x, d.y)
            if prog > d.max_progress:
                d.max_progress = prog
                if prog > self.all_time_best_progress:
                    old_best = self.all_time_best_progress
                    self.all_time_best_progress = prog
                    self.all_time_best_controller = d.controller.copy()
                    self.new_record_this_gen = True
                    if prog - old_best > 0.012:
                        self.last_status_message = f"RECORD: {prog*100:.1f}% (+{(prog-old_best)*100:.1f}%)"
                        sim_log(f"[NEW RECORD] Gen {self.generation:03d} Step {self.step_count:4d}: Progress shattered {old_best*100:.1f}% -> {prog*100:.1f}% (+{(prog-old_best)*100:.1f}%) by Drone #{d.drone_id}")

            if d.max_progress > d.last_progress_check + 0.005:
                d.last_progress_check = d.max_progress
                d.stagnant_steps = 0
            else:
                d.stagnant_steps += 1
                # 360 steps = ~6.0s grace period for navigation to prevent premature kill
                if d.time_alive_steps > 180 and d.stagnant_steps > 360:
                    d.is_alive = False
                    d.collided = True
                    continue

            gdx = d.x - gx
            gdy = d.y - gy
            if (gdx * gdx + gdy * gdy) <= g_rad2:
                d.reached_goal = True
                d.is_alive = False
                d.max_progress = 1.0
                self.all_time_best_progress = 1.0
                self.all_time_best_controller = d.controller.copy()
                self.new_record_this_gen = True
                self.stagnant_count = 0
                self.last_status_message = "EXIT REACHED! (100.0%)"
                sim_log(f"[GOAL REACHED!] Drone #{d.drone_id} reached the maze exit at step {self.step_count} (Gen {self.generation:03d})!")
                try:
                    self.save_best("best_weights_maze.npz")
                except Exception:
                    pass
                continue

            if maze.check_collision(d.x, d.y, d.radius):
                d.is_alive = False
                d.collided = True

        # 2. Flocking mutual separation (anti-crowding soft repulsion)
        alive_after = [i for i in alive_indices if self.drones[i].is_alive]
        if len(alive_after) > 1:
            aft_arr = np.array(alive_after, dtype=np.int32)
            alive_xs = self.coords[aft_arr, 0]
            alive_ys = self.coords[aft_arr, 1]
            dx = alive_xs[:, None] - alive_xs[None, :]
            dy = alive_ys[:, None] - alive_ys[None, :]
            d2 = dx * dx + dy * dy
            np.fill_diagonal(d2, 1e9)

            sep_r = 14.0
            mask = d2 < (sep_r * sep_r)
            if np.any(mask):
                dists = np.sqrt(np.maximum(d2, 1e-4))
                overlap = (sep_r - dists) * mask
                inv_d = 1.0 / dists
                fx = np.clip(np.sum((dx * inv_d) * overlap, axis=1) * 0.08, -0.6, 0.6)
                fy = np.clip(np.sum((dy * inv_d) * overlap, axis=1) * 0.08, -0.6, 0.6)
                for k, i in enumerate(alive_after):
                    d = self.drones[i]
                    nx = d.x + float(fx[k])
                    ny = d.y + float(fy[k])
                    # Require a 3.0px safe margin away from walls so repulsion never shoves a drone into collision
                    if not maze.check_collision(nx, ny, d.radius + 3.0):
                        d.x = nx
                        d.y = ny
                        self.coords[i, 0] = nx
                        self.coords[i, 1] = ny

        return not any(self.drones[i].is_alive for i in alive_indices)

    def evaluate_fitness(self) -> None:
        for d in self.drones:
            if d.reached_goal:
                # Goal finishers receive a dominant bonus guaranteeing they outrank all non-finishers,
                # with arrival speed bonus rewarding faster completion.
                time_bonus = max(0.0, 2500.0 - d.time_alive_steps * 2.0)
                d.fitness = float(10000.0 + time_bonus)
            else:
                prog_score = (d.max_progress ** 1.6) * 6000.0

                efficiency_bonus = 0.0
                if d.max_progress > 0.05:
                    efficiency_bonus = min(80.0, (d.max_progress * 1500.0) / max(1.0, d.time_alive_steps / 60.0))

                excess_spin = max(0.0, d.cumulative_rotation - 4.5)
                spin_penalty = min(120.0, excess_spin * 10.0)

                coll_penalty = 15.0 if d.collided else 0.0
                d.fitness = float(max(0.1, prog_score + efficiency_bonus - coll_penalty - spin_penalty))

    def reproduce(self, maze: Maze) -> None:
        self.evaluate_fitness()
        sorted_d = sorted(self.drones, key=lambda d: d.fitness, reverse=True)
        best_drone = sorted_d[0]
        finishers = [d for d in sorted_d if d.reached_goal]
        next_gen = self.generation + 1

        if len(finishers) > 0:
            self.all_time_best_progress = 1.0
            self.all_time_best_controller = finishers[0].controller.copy()
            self.stagnant_count = 0
            stagnation_action = f"GOAL CONVERGENCE ({len(finishers)} finishers)"
            sim_log(
                f"[GEN {self.generation:03d} SUMMARY] Steps: {self.step_count:4d} | "
                f"Peak Progress: 100.0% | "
                f"Finishers this Gen: {len(finishers)} | "
                f"Fastest Finisher Time: {finishers[0].time_alive_steps} steps | "
                f"Status: {stagnation_action}"
            )
            try:
                self.save_best("best_weights_maze.npz")
            except Exception:
                pass

            new_drones: List[MazeDrone] = []

            # 1. Direct unmutated clones of champion finisher(s)
            num_exact_clones = max(35, min(self.size - 10, max(len(finishers) * 2, int(self.size * 0.60))))
            for i in range(num_exact_clones):
                source_finisher = finishers[i % len(finishers)]
                new_drones.append(MazeDrone(
                    x=0.0, y=0.0, heading=0.0,
                    controller=source_finisher.controller.copy(),
                    drone_id=len(new_drones),
                    corridor_bias=0.0,
                    speed_multiplier=1.0,
                ))

            # 2. Micro-mutations: fine-tuning flight path with minimal variance and 0 drift
            while len(new_drones) < self.size:
                if len(finishers) > 1 and self.rng.random() < 0.40:
                    fa = self.rng.choice(finishers)
                    fb = self.rng.choice(finishers)
                    child_net = BatNeuralNet.uniform_crossover(fa.controller, fb.controller, rng=self.rng)
                else:
                    fa = self.rng.choice(finishers)
                    child_net = fa.controller.copy()
                child_net.mutate(mutation_rate=0.015, sigma=0.008, rng=self.rng)
                sm = float(self.rng.uniform(0.98, 1.05))
                new_drones.append(MazeDrone(
                    x=0.0, y=0.0, heading=0.0,
                    controller=child_net,
                    drone_id=len(new_drones),
                    corridor_bias=0.0,
                    speed_multiplier=sm,
                ))

            self.last_status_message = f"GOAL CONVERGENCE ({len(finishers)} FINISHERS)"
            sim_log(
                f"[FLOCK MULTIPLICATION] Gen {self.generation:03d} -> Gen {next_gen:03d}: "
                f"Seeded with {len(finishers)} exit champions -> {num_exact_clones} direct clones + {self.size - num_exact_clones} micro-mutants"
            )

        elif self.all_time_best_progress >= 0.999 and self.all_time_best_controller is not None:
            self.stagnant_count = 0
            stagnation_action = "ALL-TIME CHAMPION RE-SEED"
            sim_log(
                f"[GEN {self.generation:03d} SUMMARY] Steps: {self.step_count:4d} | "
                f"Peak Progress: 100.0% | Status: {stagnation_action}"
            )
            new_drones = []
            num_exact_clones = max(35, int(self.size * 0.65))
            for _ in range(num_exact_clones):
                new_drones.append(MazeDrone(
                    x=0.0, y=0.0, heading=0.0,
                    controller=self.all_time_best_controller.copy(),
                    drone_id=len(new_drones),
                    corridor_bias=0.0,
                    speed_multiplier=1.0,
                ))
            while len(new_drones) < self.size:
                child_net = self.all_time_best_controller.copy()
                child_net.mutate(mutation_rate=0.015, sigma=0.008, rng=self.rng)
                sm = float(self.rng.uniform(0.98, 1.05))
                new_drones.append(MazeDrone(
                    x=0.0, y=0.0, heading=0.0,
                    controller=child_net,
                    drone_id=len(new_drones),
                    corridor_bias=0.0,
                    speed_multiplier=sm,
                ))
            self.last_status_message = "RESTORED FROM GOAL CHAMPION"

        else:
            has_new_record = self.new_record_this_gen or (best_drone.max_progress > self.gen_start_best_progress + 0.002)
            if has_new_record:
                if best_drone.max_progress > self.all_time_best_progress:
                    self.all_time_best_progress = best_drone.max_progress
                    self.all_time_best_controller = best_drone.controller.copy()
                self.stagnant_count = 0
                stagnation_action = "NORMAL EVOLUTION"
            else:
                self.stagnant_count += 1
                if self.stagnant_count < 3:
                    stagnation_action = f"MILD ADAPTATION (stagnant {self.stagnant_count})"
                elif self.stagnant_count < 6:
                    stagnation_action = f"DIVERSITY INJECTION (stagnant {self.stagnant_count})"
                else:
                    stagnation_action = f"CATACLYSMIC MUTATION (stagnant {self.stagnant_count})"

            sim_log(
                f"[GEN {self.generation:03d} SUMMARY] Steps: {self.step_count:4d} | "
                f"Peak Progress: {self.all_time_best_progress*100:5.1f}% | "
                f"Best this Gen: {best_drone.max_progress*100:5.1f}% | "
                f"Stagnant: {self.stagnant_count:2d} gens | Status: {stagnation_action}"
            )

            if self.stagnant_count == 0:
                sigma = 0.04
                mutation_rate = 0.07
            elif self.stagnant_count < 3:
                sigma = 0.08
                mutation_rate = 0.12
            elif self.stagnant_count < 6:
                sigma = 0.15
                mutation_rate = 0.24
            else:
                sigma = 0.24
                mutation_rate = 0.38

            new_drones = []
            elite_champ = self.all_time_best_controller or best_drone.controller
            num_elites = 4 if self.stagnant_count < 4 else 2
            for _ in range(num_elites):
                new_drones.append(MazeDrone(
                    x=0.0, y=0.0, heading=0.0,
                    controller=elite_champ.copy(),
                    drone_id=len(new_drones),
                    corridor_bias=0.0,
                    speed_multiplier=1.0,
                ))
            if best_drone.controller is not elite_champ:
                new_drones.append(MazeDrone(
                    x=0.0, y=0.0, heading=0.0,
                    controller=best_drone.controller.copy(),
                    drone_id=len(new_drones),
                    corridor_bias=0.0,
                    speed_multiplier=1.0,
                ))

            if self.stagnant_count >= 3:
                num_immigrants = min(int(self.size * 0.25), 4 + self.stagnant_count * 2)
                for _ in range(num_immigrants):
                    fresh_net = BatNeuralNet(rng=self.rng)
                    fresh_net.init_scratch(rng=self.rng)
                    cb = float(self.rng.uniform(-0.35, 0.35))
                    sm = float(self.rng.uniform(0.70, 1.30))
                    new_drones.append(MazeDrone(
                        x=0.0, y=0.0, heading=0.0,
                        controller=fresh_net,
                        drone_id=len(new_drones),
                        corridor_bias=cb,
                        speed_multiplier=sm,
                    ))
                sim_log(f"[STAGNATION RESCUE] Injected {num_immigrants} fresh exploratory immigrants into Gen {next_gen:03d} (sigma={sigma:.2f}, rate={mutation_rate:.2f})")
                self.last_status_message = f"DIVERSITY INJECTION ({num_immigrants} IMMIGRANTS)"
            else:
                self.last_status_message = f"EVOLVING (Gen {next_gen})"

            pool_size = max(4, int(self.size * 0.20))
            parents = sorted_d[:pool_size]

            clone_count = max(4, int(self.size * 0.15)) if self.stagnant_count < 3 else 2
            for _ in range(clone_count):
                if len(new_drones) >= self.size:
                    break
                child_net = elite_champ.copy()
                child_net.mutate(mutation_rate=mutation_rate * 0.6, sigma=sigma * 0.6, rng=self.rng)
                cb = float(self.rng.uniform(-0.20, 0.20))
                sm = float(self.rng.uniform(0.85, 1.15))
                new_drones.append(MazeDrone(
                    x=0.0, y=0.0, heading=0.0,
                    controller=child_net,
                    drone_id=len(new_drones),
                    corridor_bias=cb,
                    speed_multiplier=sm,
                ))

            while len(new_drones) < self.size:
                pa = self.rng.choice(parents)
                pb = self.rng.choice(parents)
                child_net = BatNeuralNet.uniform_crossover(pa.controller, pb.controller, rng=self.rng)
                child_net.mutate(mutation_rate=mutation_rate, sigma=sigma, rng=self.rng)
                cb = float(self.rng.uniform(-0.30, 0.30))
                sm = float(self.rng.uniform(0.75, 1.25))
                new_drones.append(MazeDrone(
                    x=0.0, y=0.0, heading=0.0,
                    controller=child_net,
                    drone_id=len(new_drones),
                    corridor_bias=cb,
                    speed_multiplier=sm,
                ))

        self.drones = new_drones
        self.generation = next_gen
        self._sync_vectors()
        self.reset_at_maze(maze)

    def get_best_drone(self) -> MazeDrone:
        return max(self.drones, key=lambda d: (1 if d.reached_goal else 0, d.max_progress, d.fitness, d.distance_traveled))

    def save_best(self, filepath: str) -> None:
        if self.all_time_best_controller is not None:
            self.all_time_best_controller.save(filepath)
        else:
            best = self.get_best_drone()
            best.controller.save(filepath)

    def load_champion(self, filepath: str) -> BatNeuralNet:
        return BatNeuralNet.load(filepath, rng=self.rng)


def run_maze_simulation(
    cols: int = 23,
    rows: int = 23,
    cell_size: float = 38.0,
    pop_size: int = 80,
    max_steps: int = 2500,
    steps_limit: Optional[int] = None,
    weights_path: Optional[str] = None,
    scratch: bool = True,
    neural_window: bool = False,
    hud_window: bool = False,
    headless: bool = False,
) -> None:
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if "SDL_VIDEO_WINDOW_POS" not in os.environ and not headless:
        os.environ["SDL_VIDEO_WINDOW_POS"] = "15,50"

    pygame.init()
    pygame.font.init()

    maze = Maze(cols=cols, rows=rows, cell_size=cell_size)
    world_w = int(maze.width)
    world_h = int(maze.height)
    world_surf = pygame.Surface((world_w, world_h))

    # Balanced square window for side-by-side recording on 1080p
    win_w = 930
    win_h = 930

    try:
        screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Bat Echolocation - Autonomous 3D Overhead Maze Navigation (60 FPS)")
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)

    init_drone_sprites()

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14, bold=True)
    small_font = pygame.font.SysFont("consolas", 12)

    pop = MazePopulation(size=pop_size, max_steps=max_steps, scratch=scratch)
    pop.reset_at_maze(maze)

    sim_log("=" * 75)
    sim_log(f"BAT SWARM SIMULATION LAUNCHED | Scratch: {scratch} | Pop: {pop_size} | Cols: {cols} | Rows: {rows}")
    sim_log(f"Active Logfile -> {LOG_FILE_PATH}")
    sim_log("=" * 75)

    resolved_weights = None
    default_archive = os.path.join(os.path.dirname(__file__), "best_weights_maze.npz")
    if weights_path and os.path.exists(weights_path):
        resolved_weights = weights_path
    elif os.path.exists("best_weights_maze.npz") and not scratch:
        resolved_weights = "best_weights_maze.npz"
    elif os.path.exists(default_archive) and not scratch:
        resolved_weights = default_archive

    champ = None
    if resolved_weights:
        try:
            champ = pop.load_champion(resolved_weights)
            pop.all_time_best_controller = champ.copy()
            pop.all_time_best_progress = 1.0
            for d in pop.drones:
                d.controller = champ.copy()
                d.corridor_bias = 0.0
                d.speed_multiplier = 1.0
            pop._sync_vectors()
            print(f"[Maze] Auto-loaded champion weights from {resolved_weights}")
            sim_log(f"[INIT] Auto-loaded champion weights from {resolved_weights}")
        except Exception as e:
            print(f"[Maze] Error loading weights from {resolved_weights}: {e}")

    demo_mode = False
    started = False if (not headless and steps_limit is None) else True
    paused = True if not started else False
    sim_speed = 1
    SPEED_OPTIONS = [1, 2, 3, 5]
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

        card_w, card_h = 360, 246
        card_x = max(15, (win_w - card_w) // 2) if win_w < 500 else 15
        card_y = max(15, win_h - card_h - 15)

        hud_btn_single = pygame.Rect(card_x + 14, card_y + 196, 332, 36)
        hud_btn_pause = pygame.Rect(card_x + 14, card_y + 196, 160, 36)
        hud_btn_reset = pygame.Rect(card_x + 186, card_y + 196, 160, 36)

        c_w, c_h = 440, 160
        c_x = (win_w - c_w) // 2
        c_y = (win_h - c_h) // 2
        center_btn_rect = pygame.Rect(c_x + (c_w - 300) // 2, c_y + 86, 300, 46)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                win_w = max(360, event.w)
                win_h = max(360, event.h)
                screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if not started:
                    if center_btn_rect.collidepoint((mx, my)) or (show_hud and hud_btn_single.collidepoint((mx, my))):
                        started = True
                        paused = False
                else:
                    if paused:
                        if show_hud and hud_btn_single.collidepoint((mx, my)):
                            paused = False
                    else:
                        if show_hud and hud_btn_pause.collidepoint((mx, my)):
                            paused = True
                        elif show_hud and hud_btn_reset.collidepoint((mx, my)):
                            pop = MazePopulation(size=pop_size, max_steps=max_steps, scratch=True)
                            pop.reset_at_maze(maze)
                            pop.all_time_best_progress = 0.0
                            pop.stagnant_count = 0
                            started = True
                            paused = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if not started:
                        started = True
                        paused = False
                    else:
                        paused = not paused
                elif event.key == pygame.K_f:
                    curr_i = SPEED_OPTIONS.index(sim_speed) if sim_speed in SPEED_OPTIONS else 0
                    sim_speed = SPEED_OPTIONS[(curr_i + 1) % len(SPEED_OPTIONS)]
                elif event.key == pygame.K_1:
                    sim_speed = 1
                elif event.key == pygame.K_2:
                    sim_speed = 2
                elif event.key == pygame.K_3:
                    sim_speed = 3
                elif event.key == pygame.K_4:
                    sim_speed = 4
                elif event.key == pygame.K_5:
                    sim_speed = 5
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
                            pop.all_time_best_controller = champ.copy()
                            for d in pop.drones:
                                d.controller = champ.copy()
                                d.corridor_bias = 0.0
                                d.speed_multiplier = 1.0
                            pop._sync_vectors()
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
                        pop.all_time_best_controller = champ.copy()
                        for d in pop.drones:
                            d.controller = champ.copy()
                            d.corridor_bias = 0.0
                            d.speed_multiplier = 1.0
                        pop._sync_vectors()
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
            steps_per_frame = sim_speed
            for _ in range(steps_per_frame):
                done = pop.step(dt, maze)
                if done:
                    if demo_mode and champ is not None:
                        pop.reset_at_maze(maze)
                        for d in pop.drones:
                            d.controller = champ.copy()
                    else:
                        best_curr = pop.get_best_drone()
                        if (best_curr.reached_goal or best_curr.max_progress >= pop.all_time_best_progress) and best_curr.max_progress > 0.15:
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

        world_surf.fill(COLOR_FLOOR)
        maze.draw(world_surf)

        best_drone = pop.get_best_drone()
        ray_mode = ray_modes[ray_mode_idx]

        if sim_speed > 1:
            for drone in pop.drones:
                if drone.is_alive:
                    is_best = (drone is best_drone)
                    drone.draw(world_surf, draw_waves=False, draw_rays=is_best, is_champ=is_best)
        else:
            for drone in pop.drones:
                is_best = (drone is best_drone)
                draw_r = (ray_mode == "all") or (ray_mode == "best" and is_best)
                drone.draw(world_surf, draw_waves=show_waves, draw_rays=draw_r, is_champ=is_best)

        scale = min(win_w / world_w, win_h / world_h)
        scaled_w = max(1, int(round(world_w * scale)))
        scaled_h = max(1, int(round(world_h * scale)))
        scaled_surf = pygame.transform.smoothscale(world_surf, (scaled_w, scaled_h))
        offset_x = (win_w - scaled_w) // 2
        offset_y = (win_h - scaled_h) // 2

        screen.fill(COLOR_FLOOR)
        screen.blit(scaled_surf, (offset_x, offset_y))

        alive_count = sum(1 for d in pop.drones if d.is_alive)
        reached_count = sum(1 for d in pop.drones if d.reached_goal)
        cur_prog = (best_drone.max_progress * 100.0) if best_drone else 0.0
        if best_drone and best_drone.max_progress > pop.all_time_best_progress:
            pop.all_time_best_progress = best_drone.max_progress
            pop.all_time_best_controller = best_drone.controller.copy()
        peak_prog = pop.all_time_best_progress * 100.0
        fps = clock.get_fps()

        telem_data = {
            "gen": pop.generation,
            "mode": 1 if demo_mode else 0,
            "alive": alive_count,
            "pop_size": pop.size,
            "reached": reached_count,
            "progress": peak_prog,
            "cur_progress": cur_prog,
            "step": pop.step_count,
            "max_steps": pop.max_steps,
            "paused": 1 if paused else 0,
            "fast_forward": 1 if sim_speed > 1 else 0,
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
            hud_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            pygame.draw.rect(hud_surf, (15, 20, 30, 230), (0, 0, card_w, card_h), border_radius=8)
            pygame.draw.rect(hud_surf, (0, 240, 160, 240), (0, 0, card_w, card_h), 1, border_radius=8)

            t_title = font.render("AUTONOMOUS SWARM MAZE NAVIGATION", True, (0, 240, 160))
            hud_surf.blit(t_title, (14, 10))

            mode_str = "CHAMPION DEMO [E]" if demo_mode else ("TRAINING FROM SCRATCH [E]" if getattr(pop, "scratch", False) else "EVOLVING SWARM [E]")
            prog_label = f"Best Progress: {peak_prog:.1f}% (Cur: {cur_prog:.1f}%)" if cur_prog < peak_prog else f"Best Progress: {peak_prog:.1f}% along maze path"
            lines = [
                f"Generation   : {pop.generation} ({mode_str})",
                f"Swarm Alive  : {alive_count} / {pop.size} ({alive_count/pop.size*100:.1f}%)",
                f"Exit Reached : {reached_count} drones",
                prog_label,
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
            fill_peak = int(bar_w * (peak_prog / 100.0))
            if fill_peak > 0:
                bar_color = (0, 240, 160) if peak_prog < 95.0 else (255, 215, 0)
                pygame.draw.rect(hud_surf, bar_color, (bar_x, bar_y, fill_peak, bar_h), border_radius=3)
            fill_cur = int(bar_w * (cur_prog / 100.0))
            if fill_cur > 0 and fill_cur < fill_peak:
                pygame.draw.rect(hud_surf, (150, 255, 210), (bar_x, bar_y, fill_cur, bar_h), border_radius=3)

            status_str = "READY [Space]" if not started else ("PAUSED [Space]" if paused else f"SPEED: {sim_speed}x ({fps:.0f} FPS)")
            t_sub = small_font.render(f"{status_str} | {pop.last_status_message}", True, (160, 175, 195))
            hud_surf.blit(t_sub, (14, yo + 16))

            # Clickable Training Control Button(s)
            by = 196
            if not started or paused:
                btn_lbl = "▶  START TRAINING  [Space]" if not started else "▶  RESUME TRAINING  [Space]"
                pygame.draw.rect(hud_surf, (16, 50, 36), (14, by, 332, 36), border_radius=6)
                pygame.draw.rect(hud_surf, (0, 240, 160), (14, by, 332, 36), 2, border_radius=6)
                t_btn = font.render(btn_lbl, True, (0, 240, 160))
                hud_surf.blit(t_btn, (14 + (332 - t_btn.get_width()) // 2, by + 9))
            else:
                pygame.draw.rect(hud_surf, (22, 32, 46), (14, by, 160, 36), border_radius=6)
                pygame.draw.rect(hud_surf, (70, 160, 240), (14, by, 160, 36), 1, border_radius=6)
                t_p = font.render("⏸ PAUSE [Space]", True, (110, 195, 255))
                hud_surf.blit(t_p, (14 + (160 - t_p.get_width()) // 2, by + 9))

                pygame.draw.rect(hud_surf, (45, 22, 28), (186, by, 160, 36), border_radius=6)
                pygame.draw.rect(hud_surf, (255, 80, 80), (186, by, 160, 36), 1, border_radius=6)
                t_r = font.render("↺ RESTART [R]", True, (255, 120, 120))
                hud_surf.blit(t_r, (186 + (160 - t_r.get_width()) // 2, by + 9))

            screen.blit(hud_surf, (card_x, card_y))

        # Floating center hero card on launch
        if not started:
            center_surf = pygame.Surface((c_w, c_h), pygame.SRCALPHA)
            pygame.draw.rect(center_surf, (10, 15, 24, 245), (0, 0, c_w, c_h), border_radius=12)
            pygame.draw.rect(center_surf, (0, 240, 160, 240), (0, 0, c_w, c_h), 2, border_radius=12)

            t_hero = font.render("AUTONOMOUS SWARM NEUROEVOLUTION", True, (0, 240, 160))
            center_surf.blit(t_hero, ((c_w - t_hero.get_width()) // 2, 20))

            t_desc = small_font.render("Generation 0 from Scratch | Ready for Dual Studio Recording", True, (180, 200, 225))
            center_surf.blit(t_desc, ((c_w - t_desc.get_width()) // 2, 48))

            cb_w, cb_h = 300, 46
            cb_x = (c_w - cb_w) // 2
            cb_y = 86
            pygame.draw.rect(center_surf, (0, 200, 130), (cb_x, cb_y, cb_w, cb_h), border_radius=8)
            pygame.draw.rect(center_surf, (255, 255, 255), (cb_x, cb_y, cb_w, cb_h), 2, border_radius=8)

            t_btn_hero = font.render("▶  START TRAINING  (Space)", True, (10, 20, 15))
            center_surf.blit(t_btn_hero, ((c_w - t_btn_hero.get_width()) // 2, cb_y + 13))

            screen.blit(center_surf, (c_x, c_y))

        mx, my = pygame.mouse.get_pos()
        hovering = False
        if not started:
            if center_btn_rect.collidepoint((mx, my)) or (show_hud and hud_btn_single.collidepoint((mx, my))):
                hovering = True
        elif show_hud:
            if paused and hud_btn_single.collidepoint((mx, my)):
                hovering = True
            elif not paused and (hud_btn_pause.collidepoint((mx, my)) or hud_btn_reset.collidepoint((mx, my))):
                hovering = True
        try:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hovering else pygame.SYSTEM_CURSOR_ARROW)
        except Exception:
            pass

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
    parser.add_argument("--cols", type=int, default=23, help="Maze columns (default: 23 for side-by-side recording)")
    parser.add_argument("--rows", type=int, default=23, help="Maze rows (default: 23 for side-by-side recording)")
    parser.add_argument("--cell-size", type=float, default=38.0, help="Grid cell size")
    parser.add_argument("--pop-size", type=int, default=80, help="Swarm population size (default: 80)")
    parser.add_argument("--max-steps", type=int, default=8000, help="Max steps per episode (default: 8000)")
    parser.add_argument("--weights", type=str, default="best_weights_maze.npz", help="Weights file to load (default: best_weights_maze.npz)")
    parser.add_argument("--scratch", action="store_true", help="Start training from random scratch weights")
    parser.add_argument("--preset", action="store_true", help="Start with hand-crafted biological reflexes instead of scratch")
    parser.add_argument("--neural-window", action="store_true", help="Launch companion neural visualizer window")
    parser.add_argument("--hud-window", action="store_true", help="Launch companion HUD telemetry window")
    parser.add_argument("--steps", type=int, default=None, help="Stop after N total frames")
    parser.add_argument("--headless", action="store_true", help="Run with dummy video driver")
    args = parser.parse_args()

    use_scratch = args.scratch or (args.preset is False and args.weights is not None and not os.path.exists(args.weights))
    run_maze_simulation(
        cols=args.cols,
        rows=args.rows,
        cell_size=args.cell_size,
        pop_size=args.pop_size,
        max_steps=args.max_steps,
        steps_limit=args.steps,
        weights_path=args.weights,
        scratch=use_scratch,
        neural_window=args.neural_window,
        hud_window=args.hud_window,
        headless=args.headless,
    )
