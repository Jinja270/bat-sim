"""
bat_sim/main.py
===============
Interactive 60 FPS Pygame Simulation Loop and Telemetry HUD for Bat Echolocation Neuroevolution.

Controls:
- Space       : Pause / Resume simulation
- F           : Toggle Fast-Forward mode (unthrottled maximum CPU stepping)
- C           : Toggle Comparison mode (evolved bio-drones vs traditional 10 Hz threshold drones)
- R           : Reset current generation episode
- D           : Cycle acoustic sensor ray rendering mode ('best' -> 'all' -> 'none')
- Left Click  : Click obstacle to drag, or click empty space to spawn circular obstacle
- Right Click : Click obstacle to remove, or click empty space to spawn rectangular obstacle
- Esc         : Graceful exit and telemetry summary
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bat_sim.physics as physics
from bat_sim.drone import Drone, TraditionalDrone, urgency_to_color
from bat_sim.environment import CircularObstacle, Environment, RectangularObstacle
from bat_sim.evolution import Population
from bat_sim.neural_ipc import ActivationWriter
from bat_sim.neural_visualizer import NeuralNetworkVisualizer


# ==============================================================================
# DISPLAY & COLOR CONSTANTS
# ==============================================================================

WINDOW_WIDTH: int = 1200
WINDOW_HEIGHT: int = 800
TARGET_FPS: int = 60

COLOR_BG: Tuple[int, int, int] = (14, 17, 24)
COLOR_GRID: Tuple[int, int, int] = (22, 26, 36)
COLOR_WALL: Tuple[int, int, int] = (70, 85, 105)
COLOR_TEXT_PRIMARY: Tuple[int, int, int] = (240, 245, 255)
COLOR_TEXT_SECONDARY: Tuple[int, int, int] = (160, 175, 195)
COLOR_TEXT_ACCENT: Tuple[int, int, int] = (46, 204, 113)
COLOR_TRADITIONAL_BLUE: Tuple[int, int, int] = (52, 152, 219)
COLOR_ECHO_HIT: Tuple[int, int, int] = (255, 220, 0)


# ==============================================================================
# DISPLAY INITIALIZATION WITH HEADLESS FALLBACK
# ==============================================================================

def init_display(
    width: int = WINDOW_WIDTH,
    height: int = WINDOW_HEIGHT,
    headless: bool = False,
) -> Tuple[Any, Any]:
    """
    Initialize Pygame and display surface with automatic headless dummy driver fallback.

    Parameters
    ----------
    width : int, default=1200
        Window width in pixels.
    height : int, default=800
        Window height in pixels.
    headless : bool, default=False
        If True or if SDL_VIDEODRIVER='dummy' in environment, forces headless dummy video driver.

    Returns
    -------
    Tuple[Any, Any]
        (pygame_module, display_surface)
    """
    if headless or os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    import pygame

    pygame.init()
    pygame.font.init()

    try:
        screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Bat Echolocation & Terminal Buzz Neuroevolution (60 FPS)")
    except pygame.error:
        # Graceful fallback to dummy driver in environments without physical displays
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        try:
            pygame.display.quit()
        except Exception:
            pass
        pygame.display.init()
        screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

    return pygame, screen


# ==============================================================================
# SENSOR RAY RENDERING
# ==============================================================================

def render_sensor_rays(
    surface: Any,
    drones: Sequence[Any],
    obstacles: Sequence[Any],
    arena_size: Tuple[float, float],
    ray_mode: str = "best",
    best_drone: Optional[Any] = None,
) -> None:
    """
    Render acoustic echolocation sensor rays (Left -30°, Front 0°, Right +30°).

    Parameters
    ----------
    surface : Any
        Pygame rendering surface.
    drones : Sequence[Any]
        List of drones in the arena.
    obstacles : Sequence[Any]
        List of obstacles.
    arena_size : Tuple[float, float]
        (width, height) of arena.
    ray_mode : str
        'best' (draws highlighted rays for the best performing alive drone),
        'all' (draws subtle rays for all alive drones),
        'none' (suppresses ray rendering).
    best_drone : Optional[Any]
        Reference to top drone agent.
    """
    if surface is None or ray_mode == "none":
        return

    import pygame

    if ray_mode == "best":
        target_drones = [best_drone] if best_drone is not None and best_drone.is_alive else []
    else:
        target_drones = [d for d in drones if d is not None and d.is_alive]

    for drone in target_drones:
        is_champion = (drone is best_drone)
        dists = physics.cast_rays_all(
            origin=(drone.x, drone.y),
            angle=drone.heading,
            ray_offsets=physics.RAY_ANGLES,
            obstacles=obstacles,
            arena_size=arena_size,
            max_range=drone.max_range,
        )

        for d, offset in zip(dists, physics.RAY_ANGLES):
            ang = drone.heading + offset
            ex = drone.x + d * math.cos(ang)
            ey = drone.y + d * math.sin(ang)

            if is_champion:
                ray_col = urgency_to_color(drone.urgency) if d < drone.max_range else (90, 105, 125)
                width = 2
                pygame.draw.line(surface, ray_col, (drone.x, drone.y), (ex, ey), width)
                if d < drone.max_range:
                    pygame.draw.circle(surface, COLOR_ECHO_HIT, (int(round(ex)), int(round(ey))), 3)
            else:
                ray_col = (50, 60, 75) if d >= drone.max_range else (80, 100, 120)
                pygame.draw.line(surface, ray_col, (drone.x, drone.y), (ex, ey), 1)


# ==============================================================================
# TELEMETRY HUD OVERLAYS
# ==============================================================================

def render_hud(
    surface: Any,
    font: Any,
    small_font: Any,
    telemetry: Dict[str, Any],
    paused: bool,
    fast_forward: bool,
    ray_mode: str,
    fps: float,
    pop_size: int = 100,
    max_steps: int = 500,
    show_wavefronts: bool = False,
    demo_mode: bool = True,
    speed: float = 75.0,
) -> None:
    """Render translucent HUD card with real-time evolutionary metrics."""
    if surface is None:
        return

    import pygame

    card_w, card_h = 360, 265
    hud_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(hud_surf, (18, 22, 32, 215), (0, 0, card_w, card_h), border_radius=8)
    pygame.draw.rect(hud_surf, (55, 68, 88, 255), (0, 0, card_w, card_h), width=1, border_radius=8)

    # Title
    t_title = font.render("BAT ECHOLOCATION HUD", True, COLOR_TEXT_ACCENT)
    hud_surf.blit(t_title, (14, 10))

    # Metrics
    alive = telemetry.get("alive_count", 0)
    surv_pct = telemetry.get("survival_rate", 0.0) * 100.0
    best_fit = telemetry.get("best_fitness", 0.0)
    avg_fit = telemetry.get("avg_fitness", 0.0)
    avg_ping = telemetry.get("avg_ping_rate", 10.0)
    coll_count = telemetry.get("collision_count", 0)
    step = telemetry.get("step", 0)

    mode_label = "CHAMPION DEMO [E]" if demo_mode else "EVOLVING POPULATION [E]"
    gen_txt = f"Generation    : {telemetry.get('generation', 0)} ({mode_label})"
    alive_txt = f"Alive Drones  : {alive} / {pop_size} ({surv_pct:.1f}%)"
    fit_txt = f"Best Fitness  : {best_fit:.1f}  (Avg: {avg_fit:.1f})"
    ping_txt = f"Avg Ping Rate : {avg_ping:.1f} Hz  [10-150 Hz]"
    spd_txt = f"Flight Speed  : {speed:.0f} px/s [UP/DOWN keys]"
    coll_txt = f"Collisions    : {coll_count}"
    step_txt = f"Step / Episode: {step} / {max_steps}"

    lines = [gen_txt, alive_txt, fit_txt, ping_txt, spd_txt, coll_txt, step_txt]
    y_off = 36
    for line in lines:
        t_line = small_font.render(line, True, COLOR_TEXT_PRIMARY)
        hud_surf.blit(t_line, (14, y_off))
        y_off += 20

    # Controls & mode indicator badge
    status_str = "PAUSED [Space]" if paused else ("FAST-FORWARD [F]" if fast_forward else f"RUNNING ({fps:.0f} FPS)")
    waves_str = "ON" if show_wavefronts else "OFF"
    t_status = small_font.render(f"State: {status_str} | Save: [S] | Rays: {ray_mode.upper()} [D] | Waves: {waves_str} [W] | Mode: [E]", True, COLOR_TEXT_SECONDARY)
    hud_surf.blit(t_status, (14, y_off + 6))

    surface.blit(hud_surf, (15, 15))


def render_comparison_hud(
    surface: Any,
    font: Any,
    small_font: Any,
    trad_alive: int,
    trad_coll: int,
    trad_total: int = 100,
) -> None:
    """Render comparison mode HUD contrasting bio-inspired vs traditional drones."""
    if surface is None:
        return

    import pygame

    card_w, card_h = 330, 150
    comp_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
    pygame.draw.rect(comp_surf, (22, 30, 45, 220), (0, 0, card_w, card_h), border_radius=8)
    pygame.draw.rect(comp_surf, (52, 152, 219, 255), (0, 0, card_w, card_h), width=1, border_radius=8)

    t_header = font.render("TRADITIONAL 10 Hz BASELINE", True, COLOR_TRADITIONAL_BLUE)
    comp_surf.blit(t_header, (14, 10))

    alive_pct = (trad_alive / trad_total * 100.0) if trad_total > 0 else 0.0
    t_alive = small_font.render(f"Alive Drones  : {trad_alive} / {trad_total} ({alive_pct:.1f}%)", True, COLOR_TEXT_PRIMARY)
    t_coll = small_font.render(f"Collisions    : {trad_coll}", True, COLOR_TEXT_PRIMARY)
    t_rate = small_font.render(f"Pulse Rate    : 10.0 Hz (Fixed)", True, COLOR_TEXT_PRIMARY)
    t_eval = small_font.render(f"Echolocation  : High collision rate under drift", True, COLOR_TEXT_SECONDARY)

    comp_surf.blit(t_alive, (14, 38))
    comp_surf.blit(t_coll, (14, 60))
    comp_surf.blit(t_rate, (14, 82))
    comp_surf.blit(t_eval, (14, 108))

    surface.blit(comp_surf, (15, 250))


# ==============================================================================
# MAIN SIMULATION RUNNER
# ==============================================================================

def run_simulation(
    pop_size: int = 100,
    max_steps: int = 500,
    headless: bool = False,
    steps_limit: Optional[int] = None,
    generations_limit: Optional[int] = None,
    weights_path: Optional[str] = None,
    comparison: bool = False,
    fast_forward: bool = False,
    waves: bool = False,
    neural_window: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute the interactive 60 FPS Pygame simulation loop.

    Parameters
    ----------
    pop_size : int, default=100
        Number of drones in population.
    max_steps : int, default=500
        Steps per generation episode.
    headless : bool, default=False
        Whether to run with dummy video driver.
    steps_limit : Optional[int], default=None
        Stop simulation after N total steps/frames.
    generations_limit : Optional[int], default=None
        Stop simulation after N generations.
    weights_path : Optional[str], default=None
        Path to .npz weights archive to load.
    comparison : bool, default=False
        Start with comparison mode enabled.
    fast_forward : bool, default=False
        Start with fast-forward unthrottled stepping enabled.
    seed : Optional[int], default=None
        Random number generator seed.

    Returns
    -------
    Dict[str, Any]
        Final telemetry dictionary.
    """
    pygame, screen = init_display(headless=headless)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 14, bold=True)
    small_font = pygame.font.SysFont("consolas", 12)

    # World and populations
    env = Environment(arena_width=WINDOW_WIDTH, arena_height=WINDOW_HEIGHT)
    pop = Population(size=pop_size, max_steps=max_steps, arena_size=(WINDOW_WIDTH, WINDOW_HEIGHT), seed=seed)

    champ = None
    resolved_path = None
    candidates = []
    if weights_path:
        candidates.append(os.path.join(os.path.dirname(__file__), weights_path))
        candidates.append(weights_path)
    candidates.append(os.path.join(os.path.dirname(__file__), "best_weights.npz"))

    for c in candidates:
        if c and os.path.exists(c):
            resolved_path = os.path.abspath(c)
            break

    if resolved_path:
        try:
            champ = pop.load_champion(resolved_path)
            for d in pop.drones:
                d.controller = champ.copy()
            print(f"[Main] Loaded champion weights from {resolved_path} into population.")
        except Exception as e:
            print(f"[Main] Warning loading weights: {e}")

    demo_mode = True if champ is not None else False
    current_speed = 75.0
    for d in pop.drones:
        d.speed = current_speed

    # Comparison mode state
    comparison_mode = comparison
    trad_drones: List[TraditionalDrone] = []
    if comparison_mode:
        trad_drones = [
            TraditionalDrone(x=d.x, y=d.y, heading=d.heading, speed=current_speed, track_wavefronts=False)
            for d in pop.drones
        ]

    # Live Neural Network Activation Tree Visualizer
    neural_vis = NeuralNetworkVisualizer(x=740, y=15, width=445, height=425)
    ipc_writer = ActivationWriter()

    neural_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "neural_window.py"))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Launch separate dedicated neural network window if requested
    if neural_window:
        try:
            subprocess.Popen([sys.executable, neural_script], cwd=repo_root)
            neural_vis.mode = "hidden"
        except Exception as e:
            print(f"[Main] Notice: could not launch neural window subprocess: {e}")

    # Simulation control flags
    paused = False
    is_fast_forward = fast_forward
    show_wavefronts = waves
    ray_modes = ["best", "all", "none"]
    ray_mode_idx = 0
    running = True

    total_frames = 0
    total_generations = 0

    while running:
        dt = 1.0 / TARGET_FPS

        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                win_w = max(640, event.w)
                win_h = max(480, event.h)
                screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
                env.set_arena_size(win_w, win_h)
                pop.set_arena_size(win_w, win_h)
                pop.reset_generation(spawn_center=(win_w * 0.5, win_h * 0.5), env=env)
                if demo_mode and champ is not None:
                    for d in pop.drones:
                        d.controller = champ.copy()
                        d.speed = current_speed
                if comparison_mode:
                    trad_drones = [
                        TraditionalDrone(x=d.x, y=d.y, heading=d.heading, speed=current_speed, track_wavefronts=False)
                        for d in pop.drones
                    ]
                neural_vis.card_x = max(10, win_w - neural_vis.card_w - 20)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    is_fast_forward = not is_fast_forward
                elif event.key == pygame.K_s:
                    save_target = resolved_path or os.path.join(os.path.dirname(__file__), "best_weights.npz")
                    try:
                        pop.save_best(save_target)
                        champ = pop.load_champion(save_target)
                        print(f"[Main] Manually saved best weights to {save_target}")
                    except Exception as e:
                        print(f"[Main] Warning saving weights: {e}")
                elif event.key == pygame.K_c:
                    comparison_mode = not comparison_mode
                    if comparison_mode and not trad_drones:
                        trad_drones = [
                            TraditionalDrone(x=d.x, y=d.y, heading=d.heading, speed=current_speed, track_wavefronts=False)
                            for d in pop.drones
                        ]
                elif event.key == pygame.K_r:
                    env.reset_default_obstacles()
                    pop.reset_generation(spawn_center=(env.arena_width * 0.5, env.arena_height * 0.5), env=env)
                    if demo_mode and champ is not None:
                        for d in pop.drones:
                            d.controller = champ.copy()
                            d.speed = current_speed
                    if comparison_mode:
                        trad_drones = [
                            TraditionalDrone(x=d.x, y=d.y, heading=d.heading, speed=current_speed, track_wavefronts=False)
                            for d in pop.drones
                        ]
                elif event.key == pygame.K_d:
                    ray_mode_idx = (ray_mode_idx + 1) % len(ray_modes)
                elif event.key == pygame.K_w:
                    show_wavefronts = not show_wavefronts
                elif event.key in (pygame.K_n, pygame.K_t):
                    neural_vis.cycle_mode()
                elif event.key == pygame.K_p:
                    try:
                        subprocess.Popen([sys.executable, neural_script], cwd=repo_root)
                        neural_vis.mode = "hidden"
                    except Exception as e:
                        print(f"[Main] Notice: could not launch neural window subprocess: {e}")
                elif event.key == pygame.K_e:
                    demo_mode = not demo_mode
                    save_target = resolved_path or os.path.join(os.path.dirname(__file__), "best_weights.npz")
                    if demo_mode:
                        try:
                            pop.save_best(save_target)
                            champ = pop.load_champion(save_target)
                        except Exception:
                            if os.path.exists(save_target):
                                champ = pop.load_champion(save_target)
                        env.reset_default_obstacles()
                        pop.reset_generation(spawn_center=(env.arena_width * 0.5, env.arena_height * 0.5), env=env)
                        if champ is not None:
                            for d in pop.drones:
                                d.controller = champ.copy()
                                d.speed = current_speed
                    else:
                        env.reset_default_obstacles()
                        pop.reset_generation(spawn_center=(env.arena_width * 0.5, env.arena_height * 0.5), env=env)
                        for d in pop.drones:
                            d.speed = current_speed
                elif event.key == pygame.K_UP:
                    current_speed = min(130.0, current_speed + 15.0)
                    for d in pop.drones:
                        d.speed = current_speed
                    for td in trad_drones:
                        td.speed = current_speed
                elif event.key == pygame.K_DOWN:
                    current_speed = max(40.0, current_speed - 15.0)
                    for d in pop.drones:
                        d.speed = current_speed
                    for td in trad_drones:
                        td.speed = current_speed

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:  # Left Click
                    hit = env.start_drag(mx, my)
                    if hit is None:
                        env.spawn_circle(mx, my, radius=30.0)
                elif event.button == 3:  # Right Click
                    hit = env.get_obstacle_at(mx, my)
                    if hit is not None:
                        env.remove_obstacle(hit)
                    else:
                        env.spawn_rect(mx, my, width=60.0, height=45.0)

            elif event.type == pygame.MOUSEMOTION:
                if env.dragged_obstacle is not None:
                    env.drag_to(event.pos[0], event.pos[1])

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    env.stop_drag()

        # 2. Physics & Population Stepping
        if not paused:
            steps_per_frame = 20 if is_fast_forward else 1
            for _ in range(steps_per_frame):
                env.step(dt)
                done = pop.step(dt, env)

                if comparison_mode:
                    for td in trad_drones:
                        if td.is_alive:
                            td.step(dt, env)

                if done:
                    env.reset_default_obstacles()
                    if demo_mode and champ is not None:
                        pop.reset_generation(spawn_center=(env.arena_width * 0.5, env.arena_height * 0.5), env=env)
                        for d in pop.drones:
                            d.controller = champ.copy()
                            d.speed = current_speed
                    else:
                        save_target = resolved_path or os.path.join(os.path.dirname(__file__), "best_weights.npz")
                        try:
                            pop.save_best(save_target)
                            champ = pop.load_champion(save_target)
                        except Exception as e:
                            pass
                        pop.reproduce(env=env)
                        for d in pop.drones:
                            d.speed = current_speed
                    total_generations += 1
                    if comparison_mode:
                        trad_drones = [
                            TraditionalDrone(x=d.x, y=d.y, heading=d.heading, speed=current_speed, track_wavefronts=False)
                            for d in pop.drones
                        ]
                    if generations_limit and total_generations >= generations_limit:
                        running = False
                        break

            total_frames += steps_per_frame
            if steps_limit and total_frames >= steps_limit:
                running = False

        # 3. Graphics Rendering
        screen.fill(COLOR_BG)

        cur_w = screen.get_width()
        cur_h = screen.get_height()

        # Draw subtle grid
        for gx in range(0, cur_w, 100):
            pygame.draw.line(screen, COLOR_GRID, (gx, 0), (gx, cur_h), 1)
        for gy in range(0, cur_h, 100):
            pygame.draw.line(screen, COLOR_GRID, (0, gy), (cur_w, gy), 1)

        # Draw arena boundary line
        pygame.draw.rect(screen, COLOR_WALL, (0, 0, cur_w, cur_h), width=2)

        # Draw obstacles
        env.draw(screen)

        # Draw acoustic sensor rays
        best_drone = pop.get_best_drone() if pop.drones else None
        render_sensor_rays(screen, pop.drones, env.obstacles, env.arena_size, ray_modes[ray_mode_idx], best_drone)

        # Draw traditional baseline drones (if in comparison mode)
        if comparison_mode:
            for td in trad_drones:
                td.draw(screen, draw_wavefronts=show_wavefronts)

        # Draw bio-inspired drones (wavefronts disabled by default for zero clutter)
        for drone in pop.drones:
            drone.draw(screen, draw_wavefronts=show_wavefronts)

        # Draw Telemetry HUD Card
        current_fps = clock.get_fps() if not is_fast_forward else 1000.0
        render_hud(
            screen,
            font,
            small_font,
            pop.get_telemetry(),
            paused,
            is_fast_forward,
            ray_modes[ray_mode_idx],
            current_fps,
            pop_size=pop_size,
            max_steps=max_steps,
            show_wavefronts=show_wavefronts,
            demo_mode=demo_mode,
            speed=current_speed,
        )

        # Draw Comparison HUD Card (if in comparison mode)
        if comparison_mode:
            trad_alive = sum(1 for td in trad_drones if td.is_alive)
            trad_coll = sum(1 for td in trad_drones if td.collided)
            render_comparison_hud(screen, font, small_font, trad_alive, trad_coll, len(trad_drones))

        # Export live activations for companion neural window
        if best_drone is not None and getattr(best_drone, "controller", None) is not None:
            ctrl = best_drone.controller
            if hasattr(ctrl, "get_last_activations"):
                acts = ctrl.get_last_activations()
                ipc_writer.write(
                    acts,
                    ping_freq=getattr(best_drone, "recent_ping_freq", 10.0),
                    speed=getattr(best_drone, "speed", 75.0),
                    heading=getattr(best_drone, "heading", 0.0),
                )

        # Draw Live Neural Network Activation Tree
        neural_vis.render(screen, font, small_font, drone=best_drone, dt=dt)

        pygame.display.flip()

        if not is_fast_forward:
            clock.tick(TARGET_FPS)

    ipc_writer.close()
    final_telemetry = pop.get_telemetry()
    pygame.quit()
    return final_telemetry


def main() -> None:
    """CLI entry point for bat echolocation simulation."""
    parser = argparse.ArgumentParser(description="Autonomous Multi-Drone Bat Echolocation Simulation")
    parser.add_argument("--headless", action="store_true", help="Run with dummy video driver")
    parser.add_argument("--comparison", action="store_true", help="Start directly in comparison mode")
    parser.add_argument("--fast-forward", action="store_true", help="Start in unthrottled fast-forward mode")
    parser.add_argument("--steps", type=int, default=None, help="Stop after N total frames/steps")
    parser.add_argument("--generations", type=int, default=None, help="Stop after N generations")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size")
    parser.add_argument("--max-steps", type=int, default=500, help="Steps per generation episode")
    parser.add_argument("--weights", type=str, default="best_weights.npz", help="Weights file to load")
    parser.add_argument("--waves", action="store_true", help="Display radar waves (default: False)")
    parser.add_argument("--neural-window", action="store_true", help="Launch neural network in a separate dedicated window")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    run_simulation(
        pop_size=args.pop_size,
        max_steps=args.max_steps,
        headless=args.headless,
        steps_limit=args.steps,
        generations_limit=args.generations,
        weights_path=args.weights,
        comparison=args.comparison,
        fast_forward=args.fast_forward,
        waves=args.waves,
        neural_window=args.neural_window,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
