from __future__ import annotations
import os
import sys
import time
from typing import Any, Dict, Optional
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bat_sim.neural_ipc import ActivationReader
from bat_sim.neural_visualizer import NeuralNetworkVisualizer

WINDOW_WIDTH: int = 760
WINDOW_HEIGHT: int = 720
TARGET_FPS: int = 60


class MockDroneWithActivations:
    def __init__(self, activations: Dict[str, Any]) -> None:
        self.activations = activations
        self.recent_ping_freq: float = float(activations.get("ping_freq", 10.0))
        self.ping_freq: float = self.recent_ping_freq
        self.speed: float = float(activations.get("speed", 75.0))
        self.heading: float = float(activations.get("heading", 0.0))
        out = activations.get("outputs", np.zeros(2, dtype=np.float64))
        self.delta_theta: float = float(out[0]) if len(out) > 0 else 0.0
        self.urgency: float = float(out[1]) if len(out) > 1 else 0.0

    @property
    def controller(self) -> Any:
        return self

    def get_last_activations(self) -> Dict[str, np.ndarray]:
        return {
            "inputs": self.activations.get("inputs", np.zeros(5, dtype=np.float64)),
            "h1": self.activations.get("h1", np.zeros(32, dtype=np.float64)),
            "h2": self.activations.get("h2", np.zeros(32, dtype=np.float64)),
            "outputs": self.activations.get("outputs", np.zeros(2, dtype=np.float64)),
        }


def run_neural_window() -> None:
    import pygame

    if "SDL_VIDEO_WINDOW_POS" not in os.environ and os.environ.get("SDL_VIDEODRIVER") != "dummy":
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1135,50"

    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    pygame.font.init()

    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Bat Echolocation - Neural Network Live Architecture (Dedicated View)")
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14, bold=True)
    small_font = pygame.font.SysFont("consolas", 12)

    visualizer = NeuralNetworkVisualizer(x=0, y=0, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    visualizer.mode = "window"

    reader = ActivationReader()
    running = True

    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                new_w = max(520, event.w)
                new_h = max(400, event.h)
                screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
                visualizer.card_w = new_w
                visualizer.card_h = new_h

        acts = reader.read()
        proxy_drone: Optional[MockDroneWithActivations] = None
        if acts is not None:
            proxy_drone = MockDroneWithActivations(acts)

        screen.fill((10, 14, 22))
        visualizer.render(screen, font, small_font, drone=proxy_drone, dt=dt)
        pygame.display.flip()

    reader.close()
    pygame.quit()


if __name__ == "__main__":
    run_neural_window()
