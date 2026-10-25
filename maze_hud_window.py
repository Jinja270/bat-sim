from __future__ import annotations
import os
import sys
import time
from typing import Any, Dict, Optional
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bat_sim.neural_ipc import ActivationReader

WINDOW_WIDTH: int = 460
WINDOW_HEIGHT: int = 420
TARGET_FPS: int = 60

COLOR_BG = (12, 17, 26)
COLOR_CARD_BG = (18, 25, 38)
COLOR_CARD_BORDER = (35, 48, 70)
COLOR_CYAN = (0, 240, 160)
COLOR_BLUE = (60, 160, 255)
COLOR_GOLD = (255, 215, 0)
COLOR_WHITE = (240, 245, 255)
COLOR_MUTED = (140, 155, 175)
COLOR_RED = (255, 75, 75)
COLOR_DARK_BAR = (28, 38, 52)


def run_maze_hud_window(steps_limit: Optional[int] = None) -> None:
    import pygame

    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    pygame.font.init()

    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Swarm Maze Navigation - Live Telemetry HUD")
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)

    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("consolas", 15, bold=True)
    body_font = pygame.font.SysFont("consolas", 13, bold=True)
    small_font = pygame.font.SysFont("consolas", 11)
    stat_font = pygame.font.SysFont("consolas", 20, bold=True)

    reader = ActivationReader()
    running = True
    frames = 0
    history_prog = []

    last_seq = -1.0
    stale_timer = 0.0

    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0
        frames += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                new_w = max(400, event.w)
                new_h = max(360, event.h)
                screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)

        data = reader.read()
        is_live = False
        telem: Dict[str, Any] = {}
        ping_freq = 10.0
        speed = 85.0

        if data is not None:
            seq = float(data.get("seq", 0.0))
            if seq != last_seq:
                last_seq = seq
                stale_timer = 0.0
            else:
                stale_timer += dt

            is_live = stale_timer < 2.0
            telem = data.get("telemetry", {})
            ping_freq = float(data.get("ping_freq", 10.0))
            speed = float(data.get("speed", 85.0))

        gen = telem.get("gen", 0)
        mode = telem.get("mode", 0)
        alive = telem.get("alive", 0)
        pop_size = max(1, telem.get("pop_size", 100))
        reached = telem.get("reached", 0)
        progress = telem.get("progress", 0.0)
        step = telem.get("step", 0)
        max_steps = telem.get("max_steps", 2500)
        paused = telem.get("paused", False)
        fast_forward = telem.get("fast_forward", False)
        fps = telem.get("fps", 60.0)
        ray_mode_int = telem.get("ray_mode", 0)
        ray_modes = ["BEST", "ALL", "NONE"]
        ray_str = ray_modes[ray_mode_int] if 0 <= ray_mode_int < len(ray_modes) else "BEST"

        if is_live and (not history_prog or history_prog[-1][0] != gen):
            history_prog.append((gen, progress))
            if len(history_prog) > 50:
                history_prog.pop(0)

        w, h = screen.get_size()
        screen.fill(COLOR_BG)

        # 1. Header Card
        header_h = 44
        pygame.draw.rect(screen, COLOR_CARD_BG, (10, 10, w - 20, header_h), border_radius=6)
        pygame.draw.rect(screen, COLOR_CARD_BORDER, (10, 10, w - 20, header_h), 1, border_radius=6)

        t_hud = title_font.render("SWARM MAZE TELEMETRY", True, COLOR_CYAN)
        screen.blit(t_hud, (22, 22))

        # Connection status badge
        dot_col = COLOR_CYAN if is_live else COLOR_RED
        dot_x = w - 120
        pygame.draw.circle(screen, dot_col, (dot_x, 32), 5)
        status_lbl = small_font.render("LIVE (60 FPS)" if is_live else "WAITING SIM", True, COLOR_WHITE if is_live else COLOR_MUTED)
        screen.blit(status_lbl, (dot_x + 10, 26))

        # 2. Main Stats Grid
        grid_y = 62
        card_h = 240
        pygame.draw.rect(screen, COLOR_CARD_BG, (10, grid_y, w - 20, card_h), border_radius=6)
        pygame.draw.rect(screen, COLOR_CARD_BORDER, (10, grid_y, w - 20, card_h), 1, border_radius=6)

        # Row 1: Generation & Mode
        mode_text = "CHAMPION DEMO" if mode == 1 else "EVOLVING SWARM"
        mode_col = COLOR_GOLD if mode == 1 else COLOR_CYAN
        t_gen = body_font.render(f"Generation  : {gen:03d}", True, COLOR_WHITE)
        t_mode = small_font.render(f"[{mode_text}]", True, mode_col)
        screen.blit(t_gen, (24, grid_y + 14))
        screen.blit(t_mode, (180, grid_y + 14))

        # Row 2: Sim Status
        sim_state = "PAUSED" if paused else ("FAST-FORWARD (20x)" if fast_forward else f"RUNNING ({fps:.0f} FPS)")
        state_col = COLOR_BLUE if fast_forward else (COLOR_CYAN if not paused else COLOR_GOLD)
        t_status = body_font.render(f"Sim State   : {sim_state}", True, state_col)
        screen.blit(t_status, (24, grid_y + 36))

        # Row 3: Swarm Population
        alive_pct = (alive / pop_size) * 100.0
        alive_col = COLOR_CYAN if alive_pct > 30.0 else COLOR_RED
        t_alive = body_font.render(f"Swarm Alive : {alive} / {pop_size} ({alive_pct:.1f}%)", True, alive_col)
        screen.blit(t_alive, (24, grid_y + 58))

        # Row 4: Goal Reached
        reached_col = COLOR_GOLD if reached > 0 else COLOR_MUTED
        t_reached = body_font.render(f"Exit Beacon : {reached} drones reached", True, reached_col)
        screen.blit(t_reached, (24, grid_y + 80))

        # Row 5: Path Progress + Big Bar
        t_prog_label = body_font.render(f"Path Progress: {progress:.1f}% along shortest path", True, COLOR_WHITE)
        screen.blit(t_prog_label, (24, grid_y + 104))

        bar_x = 24
        bar_y = grid_y + 126
        bar_w = w - 68
        bar_h = 16
        pygame.draw.rect(screen, COLOR_DARK_BAR, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        fill_w = int(bar_w * (min(100.0, max(0.0, progress)) / 100.0))
        if fill_w > 0:
            fill_col = COLOR_CYAN if progress < 90.0 else COLOR_GOLD
            pygame.draw.rect(screen, fill_col, (bar_x, bar_y, fill_w, bar_h), border_radius=4)
            # Gloss highlight
            pygame.draw.rect(screen, (255, 255, 255, 70), (bar_x, bar_y, fill_w, 4), border_radius=2)
        pygame.draw.rect(screen, COLOR_CARD_BORDER, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

        # Row 6: Echolocation & Kinematics
        t_echo = body_font.render(f"Pulse Rate  : {ping_freq:.1f} Hz  |  Speed: {speed:.1f} px/s", True, COLOR_CYAN)
        screen.blit(t_echo, (24, grid_y + 152))

        # Row 7: Steps
        step_pct = (step / max(1, max_steps)) * 100.0
        t_steps = body_font.render(f"Step / Limit: {step} / {max_steps} ({step_pct:.0f}%)", True, COLOR_MUTED)
        screen.blit(t_steps, (24, grid_y + 174))

        # Row 8: Ray Mode & Shortcuts summary
        t_rays = small_font.render(f"Sensors: Rays={ray_str} [D] | Wavefronts=[W] | Regenerate=[R]", True, COLOR_MUTED)
        screen.blit(t_rays, (24, grid_y + 202))

        # 3. Controls Card
        ctrl_y = grid_y + card_h + 8
        ctrl_h = h - ctrl_y - 10
        if ctrl_h > 30:
            pygame.draw.rect(screen, COLOR_CARD_BG, (10, ctrl_y, w - 20, ctrl_h), border_radius=6)
            pygame.draw.rect(screen, COLOR_CARD_BORDER, (10, ctrl_y, w - 20, ctrl_h), 1, border_radius=6)

            t_ctrl_title = small_font.render("SIMULATION KEY CONTROLS", True, COLOR_BLUE)
            screen.blit(t_ctrl_title, (24, ctrl_y + 8))

            keys_line1 = small_font.render("[Space] Pause/Run   [F] 20x Fast-Forward   [E] Demo Mode", True, COLOR_WHITE)
            keys_line2 = small_font.render("[D] Cycle Rays      [W] Sonar Wavefronts   [H] Toggle In-Game HUD", True, COLOR_WHITE)
            screen.blit(keys_line1, (24, ctrl_y + 28))
            screen.blit(keys_line2, (24, ctrl_y + 46))

        pygame.display.flip()

        if steps_limit and frames >= steps_limit:
            running = False

    reader.close()
    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Swarm Maze Telemetry Standalone HUD")
    parser.add_argument("--steps", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--headless", action="store_true", help="Run with dummy video driver")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    run_maze_hud_window(steps_limit=args.steps)
