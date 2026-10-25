"""
bat_sim/neural_visualizer.py
============================
Live Real-Time Neural Network Architecture & Synaptic Firing Visualizer.

Implements the exact textbook multi-layer perceptron architecture diagram:
- Bold column headers: 'Input Layer', 'Hidden Layer', 'Output Layer'.
- Large prominent circular nodes ('Node ●') with live membrane potential glows.
- Full all-to-all dense synaptic line mesh ('Weight —') connecting adjacent layers.
- Animated action potentials ('blinking and shit') streaming along active synapses.
- Bottom-right diagram legend:
    Weight  —
    Node    ●
- Real-time bio-acoustic telemetry:
    * Distance inputs (dL, dR, dF) shift green -> amber -> pulsing red on obstacle proximity.
    * Speed and Doppler shift inputs (v, vc) drive feedforward transmission.
    * Hidden ReLU units ignite in cascading layers.
    * Output steering Δθ swings cyan (left) / amber (right).
    * Output urgency λ strobes with high-frequency Terminal Buzz pulses (10 Hz -> 150 Hz).
- Supports docked top-right HUD mode, cinematic center-stage overlay mode, and toggle hide via [N] or [T].
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import bat_sim.physics as physics
from bat_sim.neural_net import BatNeuralNet


# ==============================================================================
# COLOR PALETTE & VISUAL STYLING
# ==============================================================================

BG_CARD_COLOR: Tuple[int, int, int, int] = (15, 20, 30, 240)
BG_CARD_BORDER: Tuple[int, int, int, int] = (0, 200, 255, 220)

HEADER_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
LABEL_TEXT_COLOR: Tuple[int, int, int] = (200, 220, 245)
DIM_TEXT_COLOR: Tuple[int, int, int] = (130, 150, 175)

NODE_DEFAULT_BORDER: Tuple[int, int, int] = (255, 255, 255)
NODE_DEFAULT_FILL: Tuple[int, int, int] = (20, 28, 42)

SYNAPSE_LINE_BASE: Tuple[int, int, int, int] = (50, 65, 88, 110)
SYNAPSE_LINE_ACTIVE: Tuple[int, int, int, int] = (0, 220, 255, 180)


# ==============================================================================
# ACTION POTENTIAL PACKET (FIRING PULSE)
# ==============================================================================

class ActionPotentialParticle:
    """Animated action potential pulse traveling along a synaptic axon."""

    def __init__(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        speed: float = 3.0,
        color: Tuple[int, int, int] = (0, 240, 255),
        radius: float = 2.5,
    ) -> None:
        self.start_x, self.start_y = float(start_pos[0]), float(start_pos[1])
        self.end_x, self.end_y = float(end_pos[0]), float(end_pos[1])
        self.progress: float = 0.0
        self.speed: float = float(speed)
        self.color: Tuple[int, int, int] = color
        self.radius: float = float(radius)
        self.is_alive: bool = True

    def update(self, dt: float) -> bool:
        """Advance packet position along synapse."""
        self.progress += self.speed * dt
        if self.progress >= 1.0:
            self.is_alive = False
        return self.is_alive

    @property
    def current_pos(self) -> Tuple[float, float]:
        """Interpolated current coordinates."""
        t = min(max(self.progress, 0.0), 1.0)
        cx = self.start_x + (self.end_x - self.start_x) * t
        cy = self.start_y + (self.end_y - self.start_y) * t
        return (cx, cy)


# ==============================================================================
# NEURAL NETWORK ARCHITECTURE DIAGRAM VISUALIZER
# ==============================================================================

class NeuralNetworkVisualizer:
    """
    Renders the exact textbook neural network architecture tree diagram
    with live activations, dense synaptic mesh, and animated action potentials.
    """

    def __init__(
        self,
        x: int = 740,
        y: int = 15,
        width: int = 445,
        height: int = 425,
    ) -> None:
        self.card_x = int(x)
        self.card_y = int(y)
        self.card_w = int(width)
        self.card_h = int(height)

        # Display mode: 'docked' (side card), 'expanded' (center screen), 'hidden'
        self.mode: str = "docked"
        self.particles: List[ActionPotentialParticle] = []
        self.spawn_timer: float = 0.0
        self.clock: float = 0.0

        # Layer dimensions displayed in the diagram
        self.n_in: int = 5
        self.n_h1: int = 8    # 8 prominent nodes representing Hidden Layer 1 (32 units)
        self.n_h2: int = 8    # 8 prominent nodes representing Hidden Layer 2 (32 units)
        self.n_out: int = 2   # Output Layer (Δθ steering, λ terminal buzz)

        self.input_keys: List[str] = ["d_L", "d_R", "d_F", "v", "v_c"]
        self.input_descs: List[str] = ["Left Ray", "Right Ray", "Front Ray", "Speed", "Doppler"]

    @property
    def visible(self) -> bool:
        """True if the visualizer is currently visible."""
        return self.mode != "hidden"

    def toggle_visibility(self) -> bool:
        """Toggle HUD card visibility on/off."""
        if self.mode == "hidden":
            self.mode = "docked"
        else:
            self.mode = "hidden"
        return self.visible

    def cycle_mode(self) -> str:
        """Cycle display mode: docked -> expanded (center overlay) -> hidden -> docked."""
        if self.mode == "docked":
            self.mode = "expanded"
        elif self.mode == "expanded":
            self.mode = "hidden"
        else:
            self.mode = "docked"
        return self.mode

    def update(self, dt: float) -> None:
        """Advance internal animation clock and traveling action potential packets."""
        self.clock += dt
        self.spawn_timer += dt

        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.is_alive]

    def _spawn_firing_particles(
        self,
        in_coords: List[Tuple[float, float]],
        h1_coords: List[Tuple[float, float]],
        h2_coords: List[Tuple[float, float]],
        out_coords: List[Tuple[float, float]],
        activations: Dict[str, np.ndarray],
        curr_freq: float = 10.0,
    ) -> None:
        """Spawn traveling action potential packets along active synaptic connections."""
        freq_factor = min(1.0, max(0.0, (curr_freq - 10.0) / 140.0))
        min_interval = 0.04 - 0.024 * freq_factor
        if self.spawn_timer < min_interval:
            return
        self.spawn_timer = 0.0

        rng = np.random.default_rng()
        u = activations.get("inputs", np.zeros(5))
        h1 = activations.get("h1", np.zeros(32))
        h2 = activations.get("h2", np.zeros(32))
        out = activations.get("outputs", np.zeros(2))

        speed_scale = 1.0 + 1.2 * freq_factor
        is_large = self.mode in ("window", "expanded")

        # 1. In -> H1 packets (Sample across ALL 5 inputs so every sensor visibly fires!)
        num_in_packets = 3 if is_large else 2
        for _ in range(num_in_packets):
            src_in = int(rng.integers(0, min(len(in_coords), 5)))
            dst_h1 = int(rng.integers(0, len(h1_coords)))
            val = float(u[src_in]) if src_in < len(u) else 0.5

            if src_in in (0, 1, 2):  # Proximity [dL, dR, dF]
                proximity = 1.0 - val
                p_color = (255, 140, 40) if proximity > 0.4 else (0, 240, 220)
            elif src_in == 3:  # Cruise speed
                p_color = (80, 180, 255)
            else:  # Doppler closing velocity
                p_color = (255, 60, 90) if val > 0.1 else (0, 255, 180)

            self.particles.append(
                ActionPotentialParticle(
                    start_pos=in_coords[src_in],
                    end_pos=h1_coords[dst_h1],
                    speed=rng.uniform(3.0, 5.0) * speed_scale,
                    color=p_color,
                    radius=2.6 if is_large else 2.2,
                )
            )

        # 2. H1 -> H2 packets
        num_h1_packets = 3 if is_large else 2
        for _ in range(num_h1_packets):
            src_h1 = int(rng.integers(0, len(h1_coords)))
            dst_h2 = int(rng.integers(0, len(h2_coords)))
            h1_val = float(h1[min(src_h1 * 4, len(h1) - 1)]) if len(h1) > 0 else 0.0
            p_col = (0, 240, 220) if h1_val > 0.3 else (100, 255, 160)
            self.particles.append(
                ActionPotentialParticle(
                    start_pos=h1_coords[src_h1],
                    end_pos=h2_coords[dst_h2],
                    speed=rng.uniform(3.2, 5.2) * speed_scale,
                    color=p_col,
                    radius=2.4 if is_large else 2.0,
                )
            )

        # 3. H2 -> Out packet (Urgency buzz or steering)
        num_out_packets = 2 if is_large else 1
        for _ in range(num_out_packets):
            src_h2 = int(rng.integers(0, len(h2_coords)))
            urgency = float(out[1]) if len(out) > 1 else 0.0
            steer = float(out[0]) if len(out) > 0 else 0.0

            target_out = 1 if (urgency > 0.4 or rng.random() > 0.5) else 0
            if target_out == 1:
                p_color = (255, 50, 50) if urgency > 0.5 else (0, 230, 255)
                p_rad = 3.2 if urgency > 0.5 else 2.5
            else:
                p_color = (0, 200, 255) if steer < -0.1 else ((255, 180, 40) if steer > 0.1 else (80, 240, 160))
                p_rad = 2.6

            self.particles.append(
                ActionPotentialParticle(
                    start_pos=h2_coords[src_h2],
                    end_pos=out_coords[target_out],
                    speed=rng.uniform(3.5, 5.4) * speed_scale,
                    color=p_color,
                    radius=p_rad,
                )
            )

        # Cap max active particles
        if len(self.particles) > 80:
            self.particles = self.particles[-80:]

    def render(
        self,
        surface: Any,
        font: Any,
        small_font: Any,
        drone: Optional[Any] = None,
        dt: float = 1.0 / 60.0,
    ) -> None:
        """
        Draw the live neural network architecture diagram on screen.

        Parameters
        ----------
        surface : Any
            Target Pygame surface.
        font : Any
            Standard HUD font.
        small_font : Any
            Small telemetric font.
        drone : Optional[Any]
            Active drone agent being visualized.
        dt : float, default=1/60
            Simulation frame time delta in seconds.
        """
        if surface is None or self.mode == "hidden":
            return

        import pygame

        self.update(dt)

        # Determine canvas bounds based on mode
        sw, sh = surface.get_size()
        if self.mode == "window":
            # Dedicated window fills 100% of the canvas dynamically
            cw, ch = sw, sh
            dest_x, dest_y = 0, 0
        elif self.mode == "expanded":
            # Centered full diagram overlay
            cw = min(sw - 40, max(760, int(sw * 0.85)))
            ch = min(sh - 40, max(560, int(sh * 0.85)))
            dest_x, dest_y = (sw - cw) // 2, (sh - ch) // 2
        else:
            # Docked side card
            dest_x, dest_y = self.card_x, self.card_y
            cw, ch = self.card_w, self.card_h

        # Extract live activations
        activations = {
            "inputs": np.array([0.7, 0.7, 0.7, 0.75, 0.0]),
            "h1": np.zeros(32),
            "h2": np.zeros(32),
            "outputs": np.array([0.0, 0.0]),
        }

        if drone is not None and getattr(drone, "controller", None) is not None:
            ctrl = drone.controller
            if hasattr(ctrl, "get_last_activations"):
                activations = ctrl.get_last_activations()

        u = activations.get("inputs", np.zeros(5))
        h1 = activations.get("h1", np.zeros(32))
        h2 = activations.get("h2", np.zeros(32))
        out = activations.get("outputs", np.zeros(2))

        urgency_val = float(out[1]) if len(out) > 1 else 0.0
        steer_val = float(out[0]) if len(out) > 0 else 0.0
        curr_freq = float(getattr(drone, "recent_ping_freq", 10.0 + 140.0 * urgency_val)) if drone is not None else (10.0 + 140.0 * urgency_val)

        # 1. Main Background Surface
        card_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, BG_CARD_COLOR, (0, 0, cw, ch), border_radius=12)
        pygame.draw.rect(card_surf, BG_CARD_BORDER, (0, 0, cw, ch), width=2, border_radius=12)

        # 2. Main Title Bar
        mode_hint = "[N: Center/Hide]" if self.mode == "docked" else ("[Esc: Exit]" if self.mode == "window" else "[N: Close Overlay]")
        t_title = font.render(f"NEURAL NETWORK LIVE ARCHITECTURE  {mode_hint}", True, (0, 230, 255))
        card_surf.blit(t_title, (18, 12))

        # 3. Layer Columns Geometry
        # Layer X positions:
        # Col 0: Input Layer
        # Col 1: Hidden Layer 1
        # Col 2: Hidden Layer 2
        # Col 3: Output Layer
        left_margin = 60.0
        right_margin = cw - 65.0
        span_x = right_margin - left_margin

        x_in = left_margin
        x_h1 = left_margin + span_x * 0.33
        x_h2 = left_margin + span_x * 0.66
        x_out = right_margin

        # Vertical spans
        top_y = max(65.0, ch * 0.13)
        bottom_y = ch - max(48.0, ch * 0.09)
        avail_h = bottom_y - top_y

        # Node coordinates calculation
        # Inputs: 5 nodes
        in_coords: List[Tuple[float, float]] = []
        for i in range(self.n_in):
            ny = top_y + (i / max(1, self.n_in - 1)) * avail_h
            in_coords.append((x_in, ny))

        # Hidden 1: 8 nodes
        h1_coords: List[Tuple[float, float]] = []
        for i in range(self.n_h1):
            ny = top_y + (i / max(1, self.n_h1 - 1)) * avail_h
            h1_coords.append((x_h1, ny))

        # Hidden 2: 8 nodes
        h2_coords: List[Tuple[float, float]] = []
        for i in range(self.n_h2):
            ny = top_y + (i / max(1, self.n_h2 - 1)) * avail_h
            h2_coords.append((x_h2, ny))

        # Output: 2 nodes
        out_coords: List[Tuple[float, float]] = []
        for i in range(self.n_out):
            ny = top_y + (0.28 + i * 0.44) * avail_h
            out_coords.append((x_out, ny))

        # Spawn traveling packets
        self._spawn_firing_particles(in_coords, h1_coords, h2_coords, out_coords, activations, curr_freq=curr_freq)

        # 4. Prominent Column Header Titles (Matching User's Reference Diagram!)
        # "Input Layer"
        t_head_in = font.render("Input Layer", True, HEADER_TEXT_COLOR)
        card_surf.blit(t_head_in, (int(x_in - t_head_in.get_width() / 2), int(top_y - 32)))

        # "Hidden Layer" (Centered over both hidden columns)
        hidden_center_x = (x_h1 + x_h2) / 2.0
        t_head_hid = font.render("Hidden Layer", True, HEADER_TEXT_COLOR)
        card_surf.blit(t_head_hid, (int(hidden_center_x - t_head_hid.get_width() / 2), int(top_y - 32)))

        # "Output Layer"
        t_head_out = font.render("Output Layer", True, HEADER_TEXT_COLOR)
        card_surf.blit(t_head_out, (int(x_out - t_head_out.get_width() / 2), int(top_y - 32)))

        # 5. Full Dense Synaptic Connection Mesh ('Weight —')
        # All-to-all cross lines exactly matching the textbook reference diagram!

        # In -> H1 (5 x 8 = 40 lines)
        for i, ip in enumerate(in_coords):
            in_val = float(u[i]) if i < len(u) else 0.5
            proximity = 0.0
            if i in (0, 1, 2):  # Proximity [dL, dR, dF]
                proximity = max(0.0, min(1.0, 1.0 - in_val))
                channel_active = proximity > 0.25
                active_color = (255, 140, 40) if proximity > 0.45 else (0, 230, 240)
            elif i == 3:  # Cruise speed
                channel_active = in_val > 0.2
                active_color = (80, 180, 255)
            else:  # Doppler closing velocity
                channel_active = abs(in_val) > 0.08
                active_color = (255, 60, 90) if in_val > 0.08 else (0, 230, 220)

            for j, hp in enumerate(h1_coords):
                # Highlight active synaptic pathways across hidden nodes
                is_active = channel_active and ((i + j) % 2 == 0 or (proximity > 0.45 if i in (0, 1, 2) else False))
                line_col = active_color if is_active else SYNAPSE_LINE_BASE
                pygame.draw.line(card_surf, line_col, (int(ip[0]), int(ip[1])), (int(hp[0]), int(hp[1])), 1)

        # H1 -> H2 (8 x 8 = 64 lines)
        for i, hp1 in enumerate(h1_coords):
            h1_val = float(h1[min(i * 4, len(h1) - 1)]) if len(h1) > 0 else 0.0
            for j, hp2 in enumerate(h2_coords):
                h2_val = float(h2[min(j * 4, len(h2) - 1)]) if len(h2) > 0 else 0.0
                if h1_val > 0.35 and h2_val > 0.3:
                    line_col = (0, 240, 200, 210)
                elif h1_val > 0.2:
                    line_col = SYNAPSE_LINE_ACTIVE
                else:
                    line_col = SYNAPSE_LINE_BASE
                pygame.draw.line(card_surf, line_col, (int(hp1[0]), int(hp1[1])), (int(hp2[0]), int(hp2[1])), 1)

        # H2 -> Out (8 x 2 = 16 lines)
        for i, hp2 in enumerate(h2_coords):
            h2_val = float(h2[min(i * 4, len(h2) - 1)]) if len(h2) > 0 else 0.0
            for j, op in enumerate(out_coords):
                if j == 1:  # Terminal Buzz Urgency
                    if urgency_val > 0.35 and h2_val > 0.2:
                        line_col = (255, 60, 60, int(150 + 105 * urgency_val))
                    else:
                        line_col = SYNAPSE_LINE_BASE
                else:  # Steering
                    if steer_val < -0.12 and h2_val > 0.2:
                        line_col = (0, 220, 255, 190)
                    elif steer_val > 0.12 and h2_val > 0.2:
                        line_col = (255, 180, 40, 190)
                    else:
                        line_col = SYNAPSE_LINE_BASE
                pygame.draw.line(card_surf, line_col, (int(hp2[0]), int(hp2[1])), (int(op[0]), int(op[1])), 1)

        # 6. Action Potential Firing Pulses ("blinking and shit")
        for p in self.particles:
            px, py = p.current_pos
            pygame.draw.circle(card_surf, p.color, (int(round(px)), int(round(py))), int(round(p.radius)))

        # 7. Render Input Layer Nodes ('Node ●')
        node_radius = 12 if (self.mode in ("expanded", "window") and cw >= 700) else 9
        for i, (ix, iy) in enumerate(in_coords):
            val = float(u[i]) if i < len(u) else 0.5
            node_cx, node_cy = int(round(ix)), int(round(iy))

            # Dynamic bio-acoustic color mapping
            if i in (0, 1, 2):  # Proximity [dL, dR, dF]
                proximity = max(0.0, min(1.0, 1.0 - val))
                r = int(round(46.0 + proximity * (255.0 - 46.0)))
                g = int(round(204.0 * (1.0 - proximity)))
                b = int(round(113.0 * (1.0 - proximity)))
                fill_col = (r, g, b)
                if proximity > 0.5:
                    # Alarming pulsating ring
                    ring_r = node_radius + int(3.0 + 3.0 * math.sin(self.clock * 14.0))
                    pygame.draw.circle(card_surf, (255, 50, 50, 140), (node_cx, node_cy), ring_r, width=1)
            elif i == 3:  # Cruise speed
                fill_col = (52, 152, 219)  # Blue
            else:  # Doppler closing
                fill_col = (235, 75, 95) if val > 0.1 else (46, 204, 113)

            # Draw prominent solid circular node
            pygame.draw.circle(card_surf, fill_col, (node_cx, node_cy), node_radius)
            pygame.draw.circle(card_surf, NODE_DEFAULT_BORDER, (node_cx, node_cy), node_radius, width=2)

            # Node Key Label on the left
            t_k = small_font.render(self.input_keys[i], True, LABEL_TEXT_COLOR)
            card_surf.blit(t_k, (node_cx - t_k.get_width() - 14, node_cy - 6))

        # 8. Render Hidden Layer 1 Nodes (8 nodes representing 32 units)
        for i, (hx, hy) in enumerate(h1_coords):
            h_idx = min(i * 4, len(h1) - 1)
            act = float(h1[h_idx]) if len(h1) > h_idx else 0.0
            node_cx, node_cy = int(round(hx)), int(round(hy))

            # Intensity driven by ReLU activation
            intensity = min(1.0, max(0.0, act / 1.5))
            glow_r = int(round(20.0 + intensity * (0.0 - 20.0)))
            glow_g = int(round(40.0 + intensity * (240.0 - 40.0)))
            glow_b = int(round(60.0 + intensity * (230.0 - 60.0)))

            pygame.draw.circle(card_surf, (glow_r, glow_g, glow_b), (node_cx, node_cy), node_radius - 1)
            pygame.draw.circle(card_surf, NODE_DEFAULT_BORDER, (node_cx, node_cy), node_radius - 1, width=2)
            if intensity > 0.35:
                # Activation halo
                pygame.draw.circle(card_surf, (0, 240, 200, int(160 * intensity)), (node_cx, node_cy), node_radius + 3, width=1)

        # 9. Render Hidden Layer 2 Nodes (8 nodes representing 32 units)
        for i, (hx, hy) in enumerate(h2_coords):
            h_idx = min(i * 4, len(h2) - 1)
            act = float(h2[h_idx]) if len(h2) > h_idx else 0.0
            node_cx, node_cy = int(round(hx)), int(round(hy))

            intensity = min(1.0, max(0.0, act / 1.5))
            glow_r = int(round(25.0 + intensity * (160.0 - 25.0)))
            glow_g = int(round(40.0 + intensity * (255.0 - 40.0)))
            glow_b = int(round(75.0 + intensity * (120.0 - 75.0)))

            pygame.draw.circle(card_surf, (glow_r, glow_g, glow_b), (node_cx, node_cy), node_radius - 1)
            pygame.draw.circle(card_surf, NODE_DEFAULT_BORDER, (node_cx, node_cy), node_radius - 1, width=2)
            if intensity > 0.35:
                pygame.draw.circle(card_surf, (140, 255, 120, int(160 * intensity)), (node_cx, node_cy), node_radius + 3, width=1)

        # 10. Render Output Layer Nodes (Δθ steering, λ terminal buzz)
        # Node 0: Steering Δθ in [-1.0, 1.0]
        steer_val = float(out[0]) if len(out) > 0 else 0.0
        sx, sy = int(round(out_coords[0][0])), int(round(out_coords[0][1]))

        if steer_val < -0.1:
            steer_col = (0, 200, 255)  # Left (Cyan)
        elif steer_val > 0.1:
            steer_col = (255, 170, 20)  # Right (Gold)
        else:
            steer_col = (60, 220, 120)  # Straight (Green)

        pygame.draw.circle(card_surf, steer_col, (sx, sy), node_radius + 3)
        pygame.draw.circle(card_surf, NODE_DEFAULT_BORDER, (sx, sy), node_radius + 3, width=2)

        # Direction needle inside steering node
        needle_angle = steer_val * (math.pi / 2.0)
        nx = sx + int(round(7.0 * math.sin(needle_angle)))
        ny = sy - int(round(7.0 * math.cos(needle_angle)))
        pygame.draw.line(card_surf, (15, 20, 30), (sx, sy), (nx, ny), 2)

        # Steering Text
        steer_tag = "L" if steer_val < -0.1 else ("R" if steer_val > 0.1 else "S")
        t_s = small_font.render(f"Δθ: {steer_val:+.2f} ({steer_tag})", True, LABEL_TEXT_COLOR)
        card_surf.blit(t_s, (sx + node_radius + 10, sy - 6))

        # Node 1: Terminal Buzz Urgency λ in [0.0, 1.0]
        urgency_val = float(out[1]) if len(out) > 1 else 0.0
        ux, uy = int(round(out_coords[1][0])), int(round(out_coords[1][1]))
        curr_freq = 10.0 + 140.0 * urgency_val

        # Blinking beacon animation at actual acoustic pulse frequency
        strobe_phase = self.clock * curr_freq * 2.0 * math.pi
        strobe_alpha = (math.sin(strobe_phase) + 1.0) / 2.0

        urg_col = physics.urgency_to_color(urgency_val) if hasattr(physics, "urgency_to_color") else (
            (231, 76, 60) if urgency_val > 0.6 else (46, 204, 113)
        )

        # Shockwave halo when terminal buzz is firing
        if urgency_val > 0.45:
            halo_r = node_radius + 3 + int(5.0 * strobe_alpha)
            halo_a = int(180 * strobe_alpha * urgency_val)
            pygame.draw.circle(card_surf, (255, 40, 40, halo_a), (ux, uy), halo_r, width=2)

        pygame.draw.circle(card_surf, urg_col, (ux, uy), node_radius + 3)
        pygame.draw.circle(card_surf, NODE_DEFAULT_BORDER, (ux, uy), node_radius + 3, width=2)

        buzz_str = "BUZZ!" if urgency_val > 0.65 else f"{curr_freq:.0f}Hz"
        t_u = small_font.render(f"λ: {urgency_val*100.0:.0f}% [{buzz_str}]", True, (255, 120, 120) if urgency_val > 0.65 else LABEL_TEXT_COLOR)
        card_surf.blit(t_u, (ux + node_radius + 10, uy - 6))

        # 11. Diagram Legend (Bottom-Right, Exactly Like Reference Diagram!)
        leg_x = cw - 120
        leg_y = ch - 38

        # "Weight —"
        t_leg_w = small_font.render("Weight", True, LABEL_TEXT_COLOR)
        card_surf.blit(t_leg_w, (leg_x, leg_y))
        pygame.draw.line(card_surf, (200, 220, 245), (leg_x + 60, leg_y + 6), (leg_x + 90, leg_y + 6), 2)

        # "Node ●"
        t_leg_n = small_font.render("Node", True, LABEL_TEXT_COLOR)
        card_surf.blit(t_leg_n, (leg_x, leg_y + 16))
        pygame.draw.circle(card_surf, (255, 255, 255), (leg_x + 75, leg_y + 22), 5)

        # 12. Bottom Left Summary Status
        status_tag = "[LIVE SYNC - 60 FPS]" if drone is not None else "[WAITING FOR SIMULATION]"
        status_color = (0, 240, 200) if drone is not None else (255, 160, 40)
        t_live = small_font.render(status_tag, True, status_color)
        card_surf.blit(t_live, (18, ch - 22))

        t_status = small_font.render(
            f"Ping: {curr_freq:.1f} Hz | Front Echo: {u[2]*100.0:.0f}% | Closing Vel: {u[4]*100.0:+.0f}% | Steer: {steer_val:+.2f}",
            True,
            DIM_TEXT_COLOR,
        )
        card_surf.blit(t_status, (18 + t_live.get_width() + 14, ch - 22))

        # Blit final canvas onto simulation screen
        surface.blit(card_surf, (dest_x, dest_y))


# Global visualizer singleton instance
_visualizer_instance: Optional[NeuralNetworkVisualizer] = None


def get_neural_visualizer() -> NeuralNetworkVisualizer:
    """Return or initialize global neural network visualizer instance."""
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = NeuralNetworkVisualizer()
    return _visualizer_instance
