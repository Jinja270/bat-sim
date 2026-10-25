from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
SPEED_OF_SOUND: float = 343.0
DEFAULT_PULSE_FREQ_MIN: float = 10.0
PULSE_RATE_MIN: float = DEFAULT_PULSE_FREQ_MIN
DEFAULT_PULSE_FREQ_MAX: float = 150.0
PULSE_RATE_MAX: float = DEFAULT_PULSE_FREQ_MAX
DEFAULT_CARRIER_FREQ: float = 40000.0
DEFAULT_EMIT_FREQ: float = DEFAULT_CARRIER_FREQ
DEFAULT_MAX_RANGE: float = 300.0
DEFAULT_MAX_SPEED: float = 100.0
ARENA_WIDTH: float = 1200.0
DEFAULT_ARENA_WIDTH: float = ARENA_WIDTH
ARENA_HEIGHT: float = 800.0
DEFAULT_ARENA_HEIGHT: float = ARENA_HEIGHT
RAY_ANGLES: Tuple[float, float, float] = (-math.pi / 6.0, 0.0, math.pi / 6.0)
RAY_OFFSETS: Tuple[float, float, float] = RAY_ANGLES
DEFAULT_RAY_OFFSETS: Tuple[float, float, float] = RAY_ANGLES
SENSOR_INPUT_DIM: int = 5
SENSOR_BOUNDS: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0))

def tof_distance(delta_t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    arr = np.asarray(delta_t)
    if np.any(arr < 0.0):
        raise ValueError(f'Time of flight delta_t must be non-negative (>= 0.0), got {delta_t}')
    result = SPEED_OF_SOUND * arr / 2.0
    return float(result) if np.isscalar(delta_t) else result

def tof_time(distance: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    arr = np.asarray(distance)
    if np.any(arr < 0.0):
        raise ValueError(f'Distance must be non-negative (>= 0.0), got {distance}')
    result = 2.0 * arr / SPEED_OF_SOUND
    return float(result) if np.isscalar(distance) else result

def doppler_velocity(f_obs: Union[float, np.ndarray], f_emit: Union[float, np.ndarray]=DEFAULT_CARRIER_FREQ) -> Union[float, np.ndarray]:
    arr_emit = np.asarray(f_emit)
    arr_obs = np.asarray(f_obs)
    if np.any(arr_emit <= 0.0):
        raise ValueError(f'f_emit must be strictly positive (> 0.0), got {f_emit}')
    result = SPEED_OF_SOUND * (arr_obs - arr_emit) / (2.0 * arr_emit)
    if np.isscalar(f_obs) and np.isscalar(f_emit):
        return float(result)
    return result

def doppler_observed_freq(v_closing: Union[float, np.ndarray], f_emit: Union[float, np.ndarray]=DEFAULT_CARRIER_FREQ) -> Union[float, np.ndarray]:
    arr_emit = np.asarray(f_emit)
    arr_vc = np.asarray(v_closing)
    if np.any(arr_emit <= 0.0):
        raise ValueError(f'f_emit must be strictly positive (> 0.0), got {f_emit}')
    result = arr_emit * (1.0 + 2.0 * arr_vc / SPEED_OF_SOUND)
    if np.isscalar(v_closing) and np.isscalar(f_emit):
        return float(result)
    return result

def urgency_to_pulse_rate(urgency: Union[float, np.ndarray], clamp: bool=True) -> Union[float, np.ndarray]:
    arr = np.asarray(urgency)
    if not clamp and (np.any(arr < 0.0) or np.any(arr > 1.0)):
        raise ValueError(f'Urgency must be in [0.0, 1.0], got {urgency}')
    delta_rate = PULSE_RATE_MAX - PULSE_RATE_MIN
    clamped_u = np.clip(arr, 0.0, 1.0) if clamp else arr
    result = PULSE_RATE_MIN + delta_rate * clamped_u
    return float(result) if np.isscalar(urgency) else result

def pulse_rate_to_urgency(pulse_rate: Union[float, np.ndarray], clamp: bool=True) -> Union[float, np.ndarray]:
    delta_rate = PULSE_RATE_MAX - PULSE_RATE_MIN
    arr = np.asarray(pulse_rate)
    result = (arr - PULSE_RATE_MIN) / delta_rate
    if clamp:
        result = np.clip(result, 0.0, 1.0)
    return float(result) if np.isscalar(pulse_rate) else result

def compute_ray_directions(heading_angle: float, ray_offsets: Optional[Sequence[float]]=None) -> np.ndarray:
    if ray_offsets is None:
        offsets = np.asarray(RAY_ANGLES, dtype=np.float64)
    else:
        offsets = np.asarray(ray_offsets, dtype=np.float64)
    angles = float(heading_angle) + offsets
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)

def cast_ray_circle(origin: Union[Sequence[float], np.ndarray], direction: Union[Sequence[float], np.ndarray], circle_center: Union[Sequence[float], np.ndarray], radius: float, max_range: float=DEFAULT_MAX_RANGE) -> float:
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    c = np.asarray(circle_center, dtype=np.float64)
    r = float(radius)
    dir_norm = float(np.linalg.norm(d))
    if dir_norm < 1e-09:
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
    eps = 1e-06
    if t1 > eps:
        t = t1
    elif t2 > eps:
        t = t2
    else:
        return float(max_range)
    return float(min(t, max_range))

def cast_ray_rect(origin: Union[Sequence[float], np.ndarray], direction: Union[Sequence[float], np.ndarray], rect: Union[Sequence[float], Tuple, np.ndarray, Any], max_range: float=DEFAULT_MAX_RANGE) -> float:
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    dir_norm = float(np.linalg.norm(d))
    if dir_norm < 1e-09:
        return float(max_range)
    d = d / dir_norm
    if hasattr(rect, 'center') and hasattr(rect, 'width') and hasattr(rect, 'height'):
        cx, cy = (float(rect.center[0]), float(rect.center[1]))
        w, h = (float(rect.width), float(rect.height))
        angle = getattr(rect, 'angle', 0.0)
    elif isinstance(rect, (list, tuple, np.ndarray)):
        if len(rect) == 4:
            x, y, w, h = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
            cx = x + w / 2.0
            cy = y + h / 2.0
            angle = 0.0
        elif len(rect) == 5:
            cx, cy, w, h, angle = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]), float(rect[4]))
        else:
            raise ValueError(f'Rectangle sequence must have length 4 or 5, got {len(rect)}')
    elif isinstance(rect, dict):
        if 'center' in rect:
            cx, cy = (float(rect['center'][0]), float(rect['center'][1]))
        else:
            cx = float(rect['x']) + float(rect['w']) / 2.0
            cy = float(rect['y']) + float(rect['h']) / 2.0
        w = float(rect.get('w', rect.get('width', 0.0)))
        h = float(rect.get('h', rect.get('height', 0.0)))
        angle = float(rect.get('angle', 0.0))
    else:
        raise ValueError(f'Unsupported rectangle descriptor type: {type(rect)}')
    hw = w / 2.0
    hh = h / 2.0
    rel_ox = o[0] - cx
    rel_oy = o[1] - cy
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    ox_loc = rel_ox * cos_a + rel_oy * sin_a
    oy_loc = -rel_ox * sin_a + rel_oy * cos_a
    dx_loc = d[0] * cos_a + d[1] * sin_a
    dy_loc = -d[0] * sin_a + d[1] * cos_a
    eps = 1e-09
    if abs(dx_loc) > eps:
        tx1 = (-hw - ox_loc) / dx_loc
        tx2 = (hw - ox_loc) / dx_loc
        tmin_x = min(tx1, tx2)
        tmax_x = max(tx1, tx2)
    else:
        if ox_loc < -hw or ox_loc > hw:
            return float(max_range)
        tmin_x = -float('inf')
        tmax_x = float('inf')
    if abs(dy_loc) > eps:
        ty1 = (-hh - oy_loc) / dy_loc
        ty2 = (hh - oy_loc) / dy_loc
        tmin_y = min(ty1, ty2)
        tmax_y = max(ty1, ty2)
    else:
        if oy_loc < -hh or oy_loc > hh:
            return float(max_range)
        tmin_y = -float('inf')
        tmax_y = float('inf')
    t_enter = max(tmin_x, tmin_y)
    t_exit = min(tmax_x, tmax_y)
    hit_eps = 1e-06
    if t_enter <= t_exit and t_exit > hit_eps:
        t = t_enter if t_enter > hit_eps else t_exit
        return float(min(t, max_range))
    return float(max_range)

def cast_ray_arena(origin: Union[Sequence[float], np.ndarray], direction: Union[Sequence[float], np.ndarray], width: float=ARENA_WIDTH, height: float=ARENA_HEIGHT, max_range: Optional[float]=None) -> float:
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    dir_norm = float(np.linalg.norm(d))
    if dir_norm < 1e-09:
        return float(max_range) if max_range is not None else float('inf')
    dx = d[0] / dir_norm
    dy = d[1] / dir_norm
    ox, oy = (o[0], o[1])
    eps = 1e-09
    t_hits: List[float] = []
    if dx < -eps:
        t = -ox / dx
        if t > 0:
            t_hits.append(t)
    elif dx > eps:
        t = (width - ox) / dx
        if t > 0:
            t_hits.append(t)
    if dy < -eps:
        t = -oy / dy
        if t > 0:
            t_hits.append(t)
    elif dy > eps:
        t = (height - oy) / dy
        if t > 0:
            t_hits.append(t)
    if not t_hits:
        t_min = float('inf')
    else:
        t_min = min(t_hits)
    if max_range is not None:
        return float(min(t_min, max_range))
    return float(t_min)

def _extract_obstacle_geometry(obs: Any) -> Tuple[str, Any, np.ndarray]:
    vel = np.array([0.0, 0.0], dtype=np.float64)
    if hasattr(obs, 'velocity'):
        vel = np.asarray(obs.velocity, dtype=np.float64)
    elif hasattr(obs, 'vx') and hasattr(obs, 'vy'):
        vel = np.array([obs.vx, obs.vy], dtype=np.float64)
    elif isinstance(obs, dict) and 'velocity' in obs:
        vel = np.asarray(obs['velocity'], dtype=np.float64)
    if hasattr(obs, 'radius') or (isinstance(obs, dict) and obs.get('type') == 'circle'):
        if hasattr(obs, 'center'):
            c = obs.center
        elif hasattr(obs, 'pos'):
            c = obs.pos
        elif hasattr(obs, 'position'):
            c = obs.position
        else:
            c = obs['center']
        r = obs.radius if hasattr(obs, 'radius') else obs['radius']
        return ('circle', (np.asarray(c, dtype=np.float64), float(r)), vel)
    if hasattr(obs, 'rect') or (isinstance(obs, dict) and obs.get('type') == 'rect'):
        rect_desc = obs.rect if hasattr(obs, 'rect') else obs['rect']
        return ('rect', rect_desc, vel)
    if isinstance(obs, (tuple, list)) and len(obs) >= 2:
        if obs[0] == 'circle':
            c = np.asarray(obs[1], dtype=np.float64)
            r = float(obs[2])
            v = np.asarray(obs[3], dtype=np.float64) if len(obs) > 3 else vel
            return ('circle', (c, r), v)
        elif obs[0] == 'rect':
            rect_desc = obs[1]
            v = np.asarray(obs[2], dtype=np.float64) if len(obs) > 2 else vel
            return ('rect', rect_desc, v)
    raise ValueError(f'Unrecognized obstacle format: {type(obs)}: {obs}')

def cast_rays_all(origin: Union[Sequence[float], np.ndarray], angle: float, ray_offsets: Sequence[float]=RAY_ANGLES, obstacles: Optional[Sequence[Any]]=None, arena_size: Tuple[float, float]=(ARENA_WIDTH, ARENA_HEIGHT), max_range: float=DEFAULT_MAX_RANGE, drone_velocity: Optional[Union[Sequence[float], np.ndarray]]=None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    orig = np.asarray(origin, dtype=np.float64)
    width, height = arena_size
    v_drone = np.asarray(drone_velocity, dtype=np.float64) if drone_velocity is not None else None
    distances: List[float] = []
    closing_vels: List[float] = []
    for offset in ray_offsets:
        ray_ang = angle + offset
        dir_vec = np.array([math.cos(ray_ang), math.sin(ray_ang)], dtype=np.float64)
        t_wall = cast_ray_arena(orig, dir_vec, width=width, height=height, max_range=max_range)
        t_min = t_wall
        hit_vel = np.array([0.0, 0.0], dtype=np.float64)
        if obstacles:
            for obs in obstacles:
                obs_type, params, obs_vel = _extract_obstacle_geometry(obs)
                if obs_type == 'circle':
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
        return (dist_arr, np.array(closing_vels, dtype=np.float64))
    return dist_arr

def cast_rays_with_doppler(origin: Union[Sequence[float], np.ndarray], angle: float, ray_offsets: Sequence[float], drone_velocity: Union[Sequence[float], np.ndarray], obstacles: Optional[Sequence[Any]]=None, arena_size: Tuple[float, float]=(ARENA_WIDTH, ARENA_HEIGHT), max_range: float=DEFAULT_MAX_RANGE) -> Tuple[np.ndarray, np.ndarray, float]:
    dist_arr, closing_arr = cast_rays_all(origin=origin, angle=angle, ray_offsets=ray_offsets, obstacles=obstacles, arena_size=arena_size, max_range=max_range, drone_velocity=drone_velocity)
    front_idx = 1 if len(ray_offsets) == 3 and abs(ray_offsets[1]) < 0.0001 else int(np.argmin(np.abs(ray_offsets)))
    front_closing = float(closing_arr[front_idx])
    return (dist_arr, closing_arr, front_closing)

def batch_cast_rays_all(origins: np.ndarray, angles: np.ndarray, velocities: np.ndarray, circle_centers: Optional[np.ndarray]=None, circle_radii: Optional[np.ndarray]=None, circle_velocities: Optional[np.ndarray]=None, rect_centers: Optional[np.ndarray]=None, rect_sizes: Optional[np.ndarray]=None, rect_angles: Optional[np.ndarray]=None, rect_velocities: Optional[np.ndarray]=None, arena_size: Tuple[float, float]=(ARENA_WIDTH, ARENA_HEIGHT), ray_offsets: Optional[Sequence[float]]=None, max_range: float=DEFAULT_MAX_RANGE, obstacles: Optional[Sequence[Any]]=None) -> Tuple[np.ndarray, np.ndarray]:
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
    ray_angles = angles_arr[:, None] + offsets[None, :]
    dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)
    px = origins_arr[:, 0:1]
    py = origins_arr[:, 1:2]
    dx = dirs[:, :, 0]
    dy = dirs[:, :, 1]
    safe_dx = np.where(np.abs(dx) > 1e-06, dx, 1.0)
    safe_dy = np.where(np.abs(dy) > 1e-06, dy, 1.0)
    t_left = np.where(dx < -1e-06, -px / safe_dx, np.inf)
    t_right = np.where(dx > 1e-06, (width - px) / safe_dx, np.inf)
    t_top = np.where(dy < -1e-06, -py / safe_dy, np.inf)
    t_bottom = np.where(dy > 1e-06, (height - py) / safe_dy, np.inf)
    t_min = np.minimum(np.minimum(t_left, t_right), np.minimum(t_top, t_bottom))
    t_min = np.minimum(t_min, max_range)
    hit_vel = np.zeros((N, K, 2), dtype=np.float64)
    if obstacles is not None and circle_centers is None and (rect_centers is None):
        c_centers_list, c_radii_list, c_vels_list = ([], [], [])
        r_centers_list, r_sizes_list, r_angles_list, r_vels_list = ([], [], [], [])
        for obs in obstacles:
            obs_type, params, ovel = _extract_obstacle_geometry(obs)
            if obs_type == 'circle':
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
    if circle_centers is not None and len(circle_centers) > 0:
        c_centers = np.asarray(circle_centers, dtype=np.float64)
        c_radii = np.asarray(circle_radii, dtype=np.float64)
        c_vels = np.asarray(circle_velocities, dtype=np.float64) if circle_velocities is not None else np.zeros((len(c_centers), 2))
        m = origins_arr[:, None, :] - c_centers[None, :, :]
        b = np.einsum('ikd,ijd->ikj', dirs, m)
        c_eq = np.sum(m ** 2, axis=-1) - (c_radii ** 2)[None, :]
        disc = b ** 2 - c_eq[:, None, :]
        sqrt_disc = np.sqrt(np.maximum(0.0, disc))
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc
        valid1 = (disc >= 0) & (t1 > 1e-06)
        valid2 = (disc >= 0) & ~valid1 & (t2 > 1e-06)
        t_circ = np.where(valid1, t1, np.where(valid2, t2, np.inf))
        min_circ_idx = np.argmin(t_circ, axis=-1)
        min_circ_dist = np.take_along_axis(t_circ, min_circ_idx[:, :, None], axis=-1).squeeze(-1)
        circle_hit_mask = min_circ_dist < t_min
        t_min = np.where(circle_hit_mask, min_circ_dist, t_min)
        chosen_c_vel = c_vels[min_circ_idx]
        hit_vel = np.where(circle_hit_mask[:, :, None], chosen_c_vel, hit_vel)
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
        eps = 1e-09
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
        hit_mask = (t_enter <= t_exit) & (t_exit > 1e-06)
        t_rect = np.where(hit_mask, np.where(t_enter > 1e-06, t_enter, t_exit), np.inf)
        min_rect_idx = np.argmin(t_rect, axis=-1)
        min_rect_dist = np.take_along_axis(t_rect, min_rect_idx[:, :, None], axis=-1).squeeze(-1)
        rect_hit_mask = min_rect_dist < t_min
        t_min = np.where(rect_hit_mask, min_rect_dist, t_min)
        chosen_r_vel = r_vels[min_rect_idx]
        hit_vel = np.where(rect_hit_mask[:, :, None], chosen_r_vel, hit_vel)
    distances = np.minimum(t_min, max_range)
    v_rel = vels_arr[:, None, :] - hit_vel
    v_closing = np.sum(v_rel * dirs, axis=-1)
    v_closing = np.where(distances < max_range - 0.0001, v_closing, 0.0)
    front_idx = 1 if K == 3 and abs(offsets[1]) < 0.0001 else int(np.argmin(np.abs(offsets)))
    front_closing = v_closing[:, front_idx]
    return (distances, front_closing)

def normalize_sensor_readings(left_dist: Union[float, np.ndarray], right_dist: Optional[Union[float, np.ndarray]]=None, front_dist: Optional[Union[float, np.ndarray]]=None, speed: Optional[Union[float, np.ndarray]]=None, closing_vel: Optional[Union[float, np.ndarray]]=None, max_range: float=DEFAULT_MAX_RANGE, max_speed: float=DEFAULT_MAX_SPEED) -> np.ndarray:
    if max_range <= 0.0:
        raise ValueError(f'max_range must be positive (> 0.0), got {max_range}')
    if max_speed <= 0.0:
        raise ValueError(f'max_speed must be positive (> 0.0), got {max_speed}')
    if right_dist is None and isinstance(left_dist, np.ndarray):
        return normalize_sensor_array(left_dist, max_range=max_range, max_speed=max_speed)
    if right_dist is None or front_dist is None or speed is None or (closing_vel is None):
        raise ValueError('All 5 sensor reading arguments must be provided when not passing a pre-assembled array')
    is_scalar = all((np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0) for x in (left_dist, right_dist, front_dist, speed, closing_vel)))
    if is_scalar:
        nl = float(np.clip(float(left_dist) / max_range, 0.0, 1.0))
        nr = float(np.clip(float(right_dist) / max_range, 0.0, 1.0))
        nf = float(np.clip(float(front_dist) / max_range, 0.0, 1.0))
        ns = float(np.clip(float(speed) / max_speed, 0.0, 1.0))
        nc = float(np.clip(float(closing_vel) / max_speed, -1.0, 1.0))
        return np.array([nl, nr, nf, ns, nc], dtype=np.float64)
    arrs = [np.asarray(x, dtype=np.float64) for x in (left_dist, right_dist, front_dist, speed, closing_vel)]
    b_l, b_r, b_f, b_s, b_c = np.broadcast_arrays(*arrs)
    nl = np.clip(b_l / max_range, 0.0, 1.0)
    nr = np.clip(b_r / max_range, 0.0, 1.0)
    nf = np.clip(b_f / max_range, 0.0, 1.0)
    ns = np.clip(b_s / max_speed, 0.0, 1.0)
    nc = np.clip(b_c / max_speed, -1.0, 1.0)
    return np.stack([nl, nr, nf, ns, nc], axis=-1)

def normalize_sensor_array(raw_inputs: np.ndarray, max_range: float=DEFAULT_MAX_RANGE, max_speed: float=DEFAULT_MAX_SPEED) -> np.ndarray:
    if max_range <= 0.0:
        raise ValueError(f'max_range must be positive (> 0.0), got {max_range}')
    if max_speed <= 0.0:
        raise ValueError(f'max_speed must be positive (> 0.0), got {max_speed}')
    arr = np.asarray(raw_inputs, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != SENSOR_INPUT_DIM:
            raise ValueError(f'Expected 1D array of shape ({SENSOR_INPUT_DIM},), got {arr.shape}')
        norm = np.empty(SENSOR_INPUT_DIM, dtype=np.float64)
        norm[0:3] = np.clip(arr[0:3] / max_range, 0.0, 1.0)
        norm[3] = np.clip(arr[3] / max_speed, 0.0, 1.0)
        norm[4] = np.clip(arr[4] / max_speed, -1.0, 1.0)
        return norm
    elif arr.ndim == 2:
        if arr.shape[1] != SENSOR_INPUT_DIM:
            raise ValueError(f'Expected 2D array of shape (N, {SENSOR_INPUT_DIM}), got {arr.shape}')
        norm = np.empty_like(arr, dtype=np.float64)
        norm[:, 0:3] = np.clip(arr[:, 0:3] / max_range, 0.0, 1.0)
        norm[:, 3] = np.clip(arr[:, 3] / max_speed, 0.0, 1.0)
        norm[:, 4] = np.clip(arr[:, 4] / max_speed, -1.0, 1.0)
        return norm
    else:
        raise ValueError(f'Expected 1D or 2D array, got {arr.ndim}D array of shape {arr.shape}')

def denormalize_sensor_readings(normalized: np.ndarray, max_range: float=DEFAULT_MAX_RANGE, max_speed: float=DEFAULT_MAX_SPEED) -> np.ndarray:
    arr = np.asarray(normalized, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != SENSOR_INPUT_DIM:
            raise ValueError(f'Expected 1D array of shape ({SENSOR_INPUT_DIM},), got {arr.shape}')
        denorm = np.empty(SENSOR_INPUT_DIM, dtype=np.float64)
        denorm[0:3] = arr[0:3] * max_range
        denorm[3] = arr[3] * max_speed
        denorm[4] = arr[4] * max_speed
        return denorm
    elif arr.ndim == 2:
        if arr.shape[1] != SENSOR_INPUT_DIM:
            raise ValueError(f'Expected 2D array of shape (N, {SENSOR_INPUT_DIM}), got {arr.shape}')
        denorm = np.empty_like(arr, dtype=np.float64)
        denorm[:, 0:3] = arr[:, 0:3] * max_range
        denorm[:, 3] = arr[:, 3] * max_speed
        denorm[:, 4] = arr[:, 4] * max_speed
        return denorm
    else:
        raise ValueError(f'Expected 1D or 2D array, got {arr.ndim}D array of shape {arr.shape}')
