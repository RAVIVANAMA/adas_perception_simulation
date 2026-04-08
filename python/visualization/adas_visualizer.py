#!/usr/bin/env python3
"""
adas_visualizer.py – Real-time ADAS Perception Demo Visualizer
================================================================
A 1280×720 Pygame window with four panels:

  ┌──────────────────────────────────┬─────────────────────────┐
  │   A: CAMERA VIEW (760×450)       │  B: BIRD'S EYE (520×450)│
  │   • Detection bounding boxes     │  • Top-down ego + objects│
  │   • Lane lines (L/R boundaries)  │  • Radar arc             │
  │   • Labels, confidence, distance │  • Predicted trajectories│
  │   • Trajectory arrows            │  • Distance rings        │
  ├──────────────────────────────────┼─────────────────────────┤
  │   C: DASHBOARD (640×270)         │  D: TIME-SERIES (640×270)│
  │   • Speedometer dial             │  • Rolling speed plot    │
  │   • Brake / throttle bars        │  • Rolling TTC plot      │
  │   • AEB / ACC / LKA / TL lights  │  • Lead-vehicle distance │
  │   • Steering wheel indicator     │                          │
  └──────────────────────────────────┴─────────────────────────┘

Run:
    pip install pygame numpy
    python adas_visualizer.py

Optional flags:
    --fps   30     Target frame-rate (default 30)
    --seed  42     RNG seed for reproducibility
    --width 1280   Window width
    --height 720   Window height
    --no-fullscreen
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    sys.exit("pygame not found. Install: pip install pygame")

# ═══════════════════════════════════════════════════════════════════════════════
# ── Colour palette ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

BLACK      = (  5,   5,   5)
DARK_GRAY  = ( 25,  27,  30)
PANEL_BG   = ( 18,  20,  24)
GRID_COLOR = ( 35,  40,  48)
WHITE      = (240, 245, 250)
LIGHT_GRAY = (160, 168, 180)
MID_GRAY   = ( 80,  88,  98)

CYAN      = (  0, 220, 220)
TEAL      = (  0, 180, 160)
GREEN     = ( 40, 220,  80)
LIME      = (120, 230,  20)
YELLOW    = (240, 210,   0)
ORANGE    = (255, 140,   0)
RED       = (230,  40,  40)
DARK_RED  = (160,  20,  20)
BLUE      = ( 60, 130, 255)
PURPLE    = (160,  60, 220)
PINK      = (240,  80, 150)
GOLD      = (255, 200,  40)

# Object-class colours
CLASS_COLORS = {
    "Car":          CYAN,
    "Truck":        ORANGE,
    "Pedestrian":   GREEN,
    "Cyclist":      YELLOW,
    "Motorcycle":   PURPLE,
    "TrafficLight": GOLD,
    "TrafficSign":  PINK,
    "Unknown":      LIGHT_GRAY,
}

# AEB state colours
AEB_COLORS = {
    "Inactive":     GREEN,
    "Warning":      YELLOW,
    "PartialBrake": ORANGE,
    "FullBrake":    RED,
}

HISTORY = 180   # number of data points kept in rolling plots (~6 s at 30 fps)

# ═══════════════════════════════════════════════════════════════════════════════
# ── Data structures ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BBox2D:
    x: float; y: float; w: float; h: float

@dataclass
class Vec2:
    x: float; y: float

@dataclass
class DetectedObject:
    id:          int
    label:       str
    confidence:  float
    bbox:        BBox2D
    distance:    float          # metres
    velocity:    float          # m/s
    lat_offset:  float          # signed lateral offset in metres
    traj:        List[Vec2]     # future trajectory points (image-space)
    tl_color:    str = "unknown"    # "red" / "amber" / "green" / "unknown"

@dataclass
class LaneLine:
    points: List[Vec2]          # image-space polyline
    lateral_error: float = 0.0  # m
    heading_error: float = 0.0  # rad
    departing:     bool  = False

@dataclass
class RadarTarget:
    range:    float   # m
    azimuth:  float   # rad
    velocity: float   # m/s (closing negative)

@dataclass
class PerceptionFrame:
    frame_no:        int
    timestamp:       float

    objects:         List[DetectedObject]
    lane_left:       Optional[LaneLine]
    lane_right:      Optional[LaneLine]
    radar_targets:   List[RadarTarget]

    ego_speed:       float   # m/s
    ego_accel:       float
    ego_yaw_rate:    float
    steering:        float   # rad, positive = left

    throttle:        float   # [0,1]
    brake:           float   # [0,1]
    aeb_state:       str     # "Inactive" / "Warning" / "PartialBrake" / "FullBrake"
    acc_active:      bool
    lka_active:      bool
    leading_ttc:     float   # seconds (>100 = no threat)
    lead_distance:   float   # m (0 = no lead)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Synthetic ADAS data generator ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticADAS:
    """Produces realistic simulated ADAS data for demonstration."""

    IMG_W = 760
    IMG_H = 450

    def __init__(self, seed: int = 42):
        self._rng   = random.Random(seed)
        self._np    = np.random.default_rng(seed)
        self._t     = 0.0
        self._frame = 0
        self.dt     = 1 / 30.0

        # Ego state
        self._ego_speed = 25.0      # m/s
        self._ego_accel = 0.0
        self._steering  = 0.0

        # Lead vehicle
        self._lead_dist = 55.0      # m ahead
        self._lead_vel  = 18.0      # m/s initial
        self._lead_lat  = 0.0       # lateral m

        # Lateral driving scenario (oscillate then depart)
        self._lat_phase    = 0.0
        self._lat_drift    = 0.0

        # Traffic light cycle: 12s green → 3s amber → 15s red
        self._tl_dist      = 80.0   # m ahead (static in scene)
        self._tl_cycle     = 0.0

        # AEB scenario counter (triggers every ~25 s)
        self._aeb_timer    = 20.0
        self._aeb_active   = False
        self._aeb_state    = "Inactive"
        self._aeb_ttc      = 999.0

        # Pedestrian (appears briefly on shoulder)
        self._ped_visible  = False
        self._ped_timer    = 8.0
        self._ped_x        = 3.5    # lateral m (right shoulder)

        # Noise generators
        self._det_noise    = lambda: self._np.normal(0, 1.5)

    # ─── public ──────────────────────────────────────────────────────────────

    def step(self) -> PerceptionFrame:
        self._t     += self.dt
        self._frame += 1
        self._update_ego()
        self._update_lead()
        self._update_tl()
        self._update_aeb()
        self._update_ped()
        self._update_lane()
        return self._build_frame()

    # ─── private update steps ────────────────────────────────────────────────

    def _update_ego(self):
        # Smooth speed variation around 25 m/s (slight ACC hunting)
        target = 25.0 if not self._aeb_active else max(0.0, self._ego_speed - 8 * self.dt)
        self._ego_accel = (target - self._ego_speed) * 0.3
        self._ego_speed = max(0.0, self._ego_speed + self._ego_accel * self.dt)

        # Steering from lane keeping
        self._steering = -self._lat_drift * 0.12 + math.sin(self._t * 0.4) * 0.015

    def _update_lead(self):
        # Lead vehicle oscillates between 18 and 28 m/s
        self._lead_vel = 23.0 + 5.0 * math.sin(self._t * 0.15)
        # Occasional braking event (triggers AEB)
        if self._aeb_active:
            self._lead_vel = max(0.0, self._lead_vel - 12 * self.dt)

        closing = self._ego_speed - self._lead_vel
        self._lead_dist = max(3.0, self._lead_dist - closing * self.dt)

        # Slowly drift laterally
        self._lead_lat = 0.3 * math.sin(self._t * 0.08)

    def _update_tl(self):
        self._tl_cycle = (self._tl_cycle + self.dt) % 30.0

    @property
    def _tl_state(self) -> str:
        c = self._tl_cycle
        if   c < 12:  return "green"
        elif c < 15:  return "amber"
        else:         return "red"

    def _update_aeb(self):
        self._aeb_timer -= self.dt
        if self._aeb_timer <= 0 and not self._aeb_active:
            self._aeb_active = True
            self._aeb_timer  = 35.0     # next AEB scenario

        if self._aeb_active:
            closing = self._ego_speed - self._lead_vel
            if closing > 0 and self._lead_dist > 0:
                ttc = self._lead_dist / max(closing, 0.01)
            else:
                ttc = 999.0
            self._aeb_ttc = ttc

            if   ttc < 1.2: self._aeb_state = "FullBrake"
            elif ttc < 1.8: self._aeb_state = "PartialBrake"
            elif ttc < 2.5: self._aeb_state = "Warning"
            else:           self._aeb_state = "Inactive"

            # Cancel AEB when safe gap restored
            if self._lead_dist > 50.0 or ttc > 5.0:
                self._aeb_active = False
                self._aeb_state  = "Inactive"
                self._aeb_ttc    = 999.0
        else:
            self._aeb_state = "Inactive"
            self._aeb_ttc   = 999.0

    def _update_ped(self):
        self._ped_timer -= self.dt
        if self._ped_timer <= 0:
            self._ped_visible = not self._ped_visible
            self._ped_timer   = self._rng.uniform(5, 12)

    def _update_lane(self):
        # Gentle sinusoidal lateral drift with occasional departure
        self._lat_phase += self.dt * 0.3
        self._lat_drift = 0.08 * math.sin(self._lat_phase) + \
                          0.25 * math.sin(self._lat_phase * 0.2)

    # ─── frame builder ────────────────────────────────────────────────────────

    def _project_to_image(self, dist_m: float, lat_m: float,
                           height_m: float = 0.5) -> Optional[Vec2]:
        """Simple pinhole projection. Returns None if behind camera."""
        if dist_m < 0.5:
            return None
        f = 700.0   # focal length in pixels
        cx, cy = self.IMG_W / 2, self.IMG_H / 2
        u = cx - (lat_m / dist_m) * f
        v = cy + (height_m / dist_m) * f * 0.6 + (1 - 10.0 / dist_m) * 120
        return Vec2(u, v)

    def _build_frame(self) -> PerceptionFrame:
        objects: List[DetectedObject] = []
        ego_v = self._ego_speed

        # ── Lead car ─────────────────────────────────────────────────────────
        lead_pt = self._project_to_image(self._lead_dist, self._lead_lat, 0.75)
        if lead_pt:
            scale   = max(0.2, min(1.0, 25.0 / self._lead_dist))
            bw, bh  = 130 * scale + self._det_noise(), 90 * scale + self._det_noise()
            bx      = lead_pt.x - bw / 2
            by      = lead_pt.y - bh
            closing = ego_v - self._lead_vel
            ttc_    = self._lead_dist / max(closing, 0.01) if closing > 0 else 999.0

            # Trajectory: predict 5 future positions
            traj = []
            for step in range(1, 6):
                fut_dist = self._lead_dist - closing * step * 0.3
                fut_pt   = self._project_to_image(max(5, fut_dist), self._lead_lat, 0.75)
                if fut_pt:
                    traj.append(fut_pt)

            objects.append(DetectedObject(
                id=1, label="Car",
                confidence=0.89 + self._det_noise() * 0.02,
                bbox=BBox2D(bx, by, bw, bh),
                distance=self._lead_dist + self._det_noise(),
                velocity=self._lead_vel + self._det_noise() * 0.3,
                lat_offset=self._lead_lat,
                traj=traj,
            ))

        # ── Truck (appears far ahead, then passes) ────────────────────────────
        truck_dist = 120.0 - (self._t % 40) * 1.5
        if 30 < truck_dist < 115:
            truck_pt = self._project_to_image(truck_dist, -3.5, 1.5)
            if truck_pt:
                sc   = max(0.15, 22.0 / truck_dist)
                tw   = 160 * sc; th = 120 * sc
                objects.append(DetectedObject(
                    id=2, label="Truck",
                    confidence=0.82 + self._det_noise() * 0.02,
                    bbox=BBox2D(truck_pt.x - tw/2, truck_pt.y - th, tw, th),
                    distance=truck_dist, velocity=20.0, lat_offset=-3.5,
                    traj=[],
                ))

        # ── Pedestrian (shoulder) ─────────────────────────────────────────────
        if self._ped_visible:
            ped_dist = 18.0 + 2 * math.sin(self._t)
            ped_pt   = self._project_to_image(ped_dist, self._ped_x, 0.9)
            if ped_pt:
                sc   = max(0.3, 12.0 / ped_dist)
                pw   = 40 * sc; ph = 100 * sc
                objects.append(DetectedObject(
                    id=3, label="Pedestrian",
                    confidence=0.77 + self._det_noise() * 0.03,
                    bbox=BBox2D(ped_pt.x - pw/2, ped_pt.y - ph, pw, ph),
                    distance=ped_dist, velocity=1.2, lat_offset=self._ped_x,
                    traj=[],
                ))

        # ── Traffic light ─────────────────────────────────────────────────────
        tl_pt = self._project_to_image(self._tl_dist, 0.5, -2.5)  # high up
        if tl_pt and self.IMG_H * 0.1 < tl_pt.y < self.IMG_H * 0.45:
            objects.append(DetectedObject(
                id=4, label="TrafficLight",
                confidence=0.91,
                bbox=BBox2D(tl_pt.x - 20, tl_pt.y - 60, 40, 70),
                distance=self._tl_dist, velocity=0.0, lat_offset=0.5,
                traj=[], tl_color=self._tl_state,
            ))

        # ── Lane lines ────────────────────────────────────────────────────────
        lateral_err = self._lat_drift
        heading_err = self._steering * 0.5
        departing   = abs(lateral_err) > 0.28

        def _lane_pts(offset: float, steps: int = 14) -> List[Vec2]:
            pts = []
            for i in range(steps):
                d = 5 + i * 9
                p = self._project_to_image(d, offset + lateral_err * 0.5, -0.05)
                if p:
                    pts.append(p)
            return pts

        left  = LaneLine(_lane_pts(-1.85), lateral_err, heading_err, departing)
        right = LaneLine(_lane_pts( 1.85), lateral_err, heading_err, departing)

        # ── Radar targets ─────────────────────────────────────────────────────
        radar: List[RadarTarget] = [
            RadarTarget(self._lead_dist + self._det_noise(),
                        self._lead_lat / self._lead_dist,
                        -(self._ego_speed - self._lead_vel) + self._det_noise()),
        ]
        for _ in range(2):
            r = RadarTarget(
                range    = self._rng.uniform(30, 100),
                azimuth  = self._rng.uniform(-0.4, 0.4),
                velocity = self._rng.uniform(-15, 5),
            )
            radar.append(r)

        # ── Control ───────────────────────────────────────────────────────────
        if self._aeb_state == "FullBrake":
            brake, throttle = 1.0, 0.0
        elif self._aeb_state == "PartialBrake":
            brake, throttle = 0.45, 0.0
        elif self._tl_state == "red" and self._tl_dist < 35:
            brake    = min(1.0, (35 - self._tl_dist) / 35 * 0.8)
            throttle = 0.0
        else:
            speed_err = 25.0 - self._ego_speed
            throttle  = max(0.0, min(1.0, speed_err * 0.15 + 0.3))
            brake     = max(0.0, min(1.0, -speed_err * 0.1))

        return PerceptionFrame(
            frame_no      = self._frame,
            timestamp     = self._t,
            objects       = objects,
            lane_left     = left,
            lane_right    = right,
            radar_targets = radar,
            ego_speed     = ego_v,
            ego_accel     = self._ego_accel,
            ego_yaw_rate  = math.sin(self._t * 0.3) * 0.02,
            steering      = self._steering,
            throttle      = throttle,
            brake         = brake,
            aeb_state     = self._aeb_state,
            acc_active    = True,
            lka_active    = abs(lateral_err) > 0.05,
            leading_ttc   = self._aeb_ttc,
            lead_distance = self._lead_dist,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ── Low-level drawing helpers ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def draw_rounded_rect(surf: pygame.Surface, color,
                      rect: pygame.Rect, radius: int = 8, alpha: int = 255):
    if alpha < 255:
        tmp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(tmp, (*color, alpha), tmp.get_rect(), border_radius=radius)
        surf.blit(tmp, rect.topleft)
    else:
        pygame.draw.rect(surf, color, rect, border_radius=radius)


def draw_text(surf: pygame.Surface, text: str, pos: Tuple[int, int],
              font: pygame.font.Font, color=WHITE,
              anchor: str = "topleft") -> pygame.Rect:
    img  = font.render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


def draw_dashed_line(surf, color, start, end, dash=8, gap=6, width=1):
    x1, y1 = start
    x2, y2 = end
    length  = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy  = (x2 - x1) / length, (y2 - y1) / length
    pos     = 0.0
    drawing = True
    while pos < length:
        seg_len = min(dash if drawing else gap, length - pos)
        if drawing:
            sx = x1 + dx * pos;  sy = y1 + dy * pos
            ex = x1 + dx * (pos + seg_len); ey = y1 + dy * (pos + seg_len)
            pygame.draw.line(surf, color, (int(sx), int(sy)), (int(ex), int(ey)), width)
        pos    += seg_len
        drawing = not drawing


def draw_arrow(surf, color, start: Vec2, end: Vec2, width=2, head_size=8):
    sx, sy = int(start.x), int(start.y)
    ex, ey = int(end.x),   int(end.y)
    if abs(ex - sx) + abs(ey - sy) < 4:
        return
    pygame.draw.line(surf, color, (sx, sy), (ex, ey), width)
    angle = math.atan2(ey - sy, ex - sx)
    for side in (+0.4, -0.4):
        hx = ex - head_size * math.cos(angle - side)
        hy = ey - head_size * math.sin(angle - side)
        pygame.draw.line(surf, color, (ex, ey), (int(hx), int(hy)), width)


def lerp_color(c1, c2, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def alpha_surface(size: Tuple[int, int]) -> pygame.Surface:
    s = pygame.Surface(size, pygame.SRCALPHA)
    s.fill((0, 0, 0, 0))
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# ── Panel A: Camera view ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class CameraPanel:
    """Simulated front-camera image with perception overlays."""

    W, H = 760, 450

    def __init__(self, fonts: dict):
        self._fonts = fonts
        # Pre-build sky/road gradient background
        self._bg = self._make_bg()

    def _make_bg(self) -> pygame.Surface:
        surf = pygame.Surface((self.W, self.H))
        sky_top  = ( 22,  30,  55)
        sky_bot  = ( 55,  65,  95)
        road_top = ( 52,  55,  60)
        road_bot = ( 35,  38,  42)
        horizon  = int(self.H * 0.42)

        # Sky gradient
        for y in range(horizon):
            t   = y / horizon
            col = lerp_color(sky_top, sky_bot, t)
            pygame.draw.line(surf, col, (0, y), (self.W, y))

        # Road gradient
        for y in range(horizon, self.H):
            t   = (y - horizon) / (self.H - horizon)
            col = lerp_color(road_top, road_bot, t)
            pygame.draw.line(surf, col, (0, y), (self.W, y))

        # Road centre dashes (static markings)
        cx = self.W // 2
        for y in range(horizon + 20, self.H, 40):
            pygame.draw.rect(surf, (80, 82, 85), (cx - 3, y, 6, 18))

        return surf

    def render(self, surf: pygame.Surface, frame: PerceptionFrame,
               ox: int = 0, oy: int = 0):
        # Draw background
        surf.blit(self._bg, (ox, oy))

        # ── Animated road dashes ────────────────────────────────────────────
        dash_offset = int((frame.timestamp * frame.ego_speed * 8) % 40)
        cx = ox + self.W // 2
        horizon_y = oy + int(self.H * 0.42)
        for y in range(horizon_y + dash_offset, oy + self.H, 40):
            pygame.draw.rect(surf, (90, 92, 96), (cx - 3, y, 6, 18))

        # ── Lane lines ─────────────────────────────────────────────────────
        for lane in (frame.lane_left, frame.lane_right):
            if not lane or len(lane.points) < 2:
                continue
            color = RED if lane.departing else TEAL
            pts   = [(ox + int(p.x), oy + int(p.y)) for p in lane.points]
            if len(pts) >= 2:
                pygame.draw.lines(surf, color, False, pts, 3)

        # ── Detection bounding boxes ────────────────────────────────────────
        for obj in frame.objects:
            self._draw_detection(surf, obj, ox, oy)

        # ── Trajectory arrows ───────────────────────────────────────────────
        for obj in frame.objects:
            if len(obj.traj) >= 2:
                prev = obj.traj[0]
                for nxt in obj.traj[1:]:
                    c = CLASS_COLORS.get(obj.label, LIGHT_GRAY)
                    draw_arrow(surf, (*c, 180),
                               Vec2(ox + prev.x, oy + prev.y),
                               Vec2(ox + nxt.x,  oy + nxt.y),
                               width=2, head_size=7)
                    prev = nxt

        # ── Lane departure overlay ──────────────────────────────────────────
        if frame.lane_left and frame.lane_left.departing:
            warn_surf = alpha_surface((self.W, self.H))
            pygame.draw.rect(warn_surf, (230, 40, 40, 28), warn_surf.get_rect())
            surf.blit(warn_surf, (ox, oy))
            draw_text(surf, "⚠  LANE DEPARTURE",
                      (ox + self.W // 2, oy + int(self.H * 0.88)),
                      self._fonts["bold_sm"], RED, anchor="center")

        # ── AEB overlay pulse ───────────────────────────────────────────────
        if frame.aeb_state != "Inactive":
            col = AEB_COLORS[frame.aeb_state]
            blink = int(frame.timestamp * 4) % 2 == 0
            if blink or frame.aeb_state == "FullBrake":
                brk_surf = alpha_surface((self.W, self.H))
                alpha    = 60 if frame.aeb_state == "FullBrake" else 30
                pygame.draw.rect(brk_surf, (*col, alpha), brk_surf.get_rect())
                surf.blit(brk_surf, (ox, oy))
            draw_text(surf, f"⬛ AEB — {frame.aeb_state.upper()}",
                      (ox + self.W // 2, oy + 24),
                      self._fonts["bold_sm"], col, anchor="center")

        # ── Panel title ─────────────────────────────────────────────────────
        draw_text(surf, "CAMERA VIEW",
                  (ox + 10, oy + 8), self._fonts["tiny"], LIGHT_GRAY)
        draw_text(surf, f"Frame {frame.frame_no:05d}  |  {frame.timestamp:.2f}s",
                  (ox + self.W - 10, oy + 8), self._fonts["tiny"],
                  LIGHT_GRAY, anchor="topright")

        # ── Border ──────────────────────────────────────────────────────────
        pygame.draw.rect(surf, GRID_COLOR, (ox, oy, self.W, self.H), 1)

    def _draw_detection(self, surf, obj: DetectedObject, ox: int, oy: int):
        x = ox + int(obj.bbox.x)
        y = oy + int(obj.bbox.y)
        w = int(obj.bbox.w)
        h = int(obj.bbox.h)

        col  = CLASS_COLORS.get(obj.label, LIGHT_GRAY)
        col2 = lerp_color(col, BLACK, 0.55)

        # Main box
        pygame.draw.rect(surf, col, (x, y, w, h), 2, border_radius=4)

        # Corner decorations
        corner = min(14, w // 4, h // 4)
        for cx2, cy2, sx, sy in [
            (x, y, 1, 1), (x+w, y, -1, 1),
            (x, y+h, 1, -1), (x+w, y+h, -1, -1),
        ]:
            pygame.draw.line(surf, WHITE, (cx2, cy2), (cx2+sx*corner, cy2), 2)
            pygame.draw.line(surf, WHITE, (cx2, cy2), (cx2, cy2+sy*corner), 2)

        # Traffic-light indicator inside box
        if obj.label == "TrafficLight":
            tl_col = {"green": GREEN, "amber": ORANGE, "red": RED}.get(obj.tl_color, MID_GRAY)
            pygame.draw.circle(surf, tl_col, (x + w//2, y + h//2), min(10, w//3))

        # Label background
        font   = self._fonts["tiny"]
        label  = f"{obj.label}  {obj.confidence*100:.0f}%"
        dist_t = f"{obj.distance:.1f}m"
        lw     = max(font.size(label)[0], font.size(dist_t)[0]) + 8
        lh     = 32
        ly     = y - lh - 2 if y - lh - 2 > oy else y + h + 2
        draw_rounded_rect(surf, col2, pygame.Rect(x, ly, lw, lh),
                          radius=4, alpha=220)

        draw_text(surf, label, (x + 4, ly + 2),    font, WHITE)
        draw_text(surf, dist_t, (x + 4, ly + 16),   font, LIME)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Panel B: Bird's-Eye View  ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class BirdEyePanel:
    """Top-down map view of the environment around the ego vehicle."""

    W, H  = 520, 450
    SCALE = 3.5   # pixels per metre

    def __init__(self, fonts: dict):
        self._fonts = fonts

    def render(self, surf: pygame.Surface, frame: PerceptionFrame,
               ox: int = 0, oy: int = 0):
        # Background
        pygame.draw.rect(surf, PANEL_BG, (ox, oy, self.W, self.H))

        # Ego vehicle position in panel coords (lower-centre)
        ex = ox + self.W // 2
        ey = oy + int(self.H * 0.82)

        def world_to_panel(dist_m: float, lat_m: float) -> Tuple[int, int]:
            px = int(ex - lat_m  * self.SCALE)
            py = int(ey - dist_m * self.SCALE)
            return px, py

        # ── Concentric distance rings ────────────────────────────────────────
        for r_m in [10, 20, 40, 60, 80, 100]:
            r_px = int(r_m * self.SCALE)
            pygame.draw.circle(surf, GRID_COLOR, (ex, ey), r_px, 1)
            draw_text(surf, f"{r_m}m",
                      (ex + r_px + 3, ey - 8),
                      self._fonts["tiny"], GRID_COLOR)

        # ── Road lanes (white lines either side) ────────────────────────────
        for lat in (-1.85, 1.85):
            pygame.draw.line(surf, (55, 60, 70),
                             world_to_panel(0, lat),
                             world_to_panel(110, lat), 1)

        # ── Radar arc ───────────────────────────────────────────────────────
        radar_range = 120
        r_surf = alpha_surface((self.W, self.H))
        arc_rect = pygame.Rect(ex - int(radar_range*self.SCALE),
                               ey - int(radar_range*self.SCALE),
                               int(radar_range*self.SCALE*2),
                               int(radar_range*self.SCALE*2))
        try:
            pygame.draw.arc(r_surf, (*TEAL, 25),
                            arc_rect,
                            math.radians(80), math.radians(100), 1)
            # Fill sector
            pygame.draw.arc(r_surf, (0, 180, 160, 12),
                            arc_rect,
                            math.radians(80), math.radians(100),
                            int(radar_range * self.SCALE))
        except Exception:
            pass
        surf.blit(r_surf, (ox, oy))

        # ── Radar targets ───────────────────────────────────────────────────
        for tgt in frame.radar_targets:
            rx, ry = world_to_panel(tgt.range,
                                    tgt.range * math.tan(tgt.azimuth))
            if ox <= rx <= ox + self.W and oy <= ry <= oy + self.H:
                pygame.draw.circle(surf, TEAL, (rx, ry), 4)
                draw_dashed_line(surf, TEAL, (ex, ey), (rx, ry), dash=5, gap=5)

        # ── Tracked objects ──────────────────────────────────────────────────
        for obj in frame.objects:
            if obj.distance > 110:
                continue
            ox2, oy2 = world_to_panel(obj.distance, obj.lat_offset)
            if not (ox <= ox2 <= ox + self.W and oy <= oy2 <= oy + self.H):
                continue
            col  = CLASS_COLORS.get(obj.label, LIGHT_GRAY)
            size = 12 if obj.label == "Truck" else 9

            pygame.draw.rect(surf, col,
                             (ox2 - size//2, oy2 - size//2, size, size), 0,
                             border_radius=3)

            # Velocity vector
            vx = ex + int(-obj.lat_offset * self.SCALE)
            speed_px = int(obj.velocity * self.SCALE * 0.6)
            pygame.draw.line(surf, col,
                             (ox2, oy2),
                             (ox2, oy2 - speed_px), 2)

            # Label
            draw_text(surf, f"ID:{obj.id} {obj.distance:.0f}m",
                      (ox2 + 8, oy2 - 6), self._fonts["tiny"], col)

            # Predicted trajectory in BEV
            if obj.traj:
                prev_bev = (ox2, oy2)
                for i, tp in enumerate(obj.traj[:4]):
                    # Convert image-space traj point back to approx world
                    fut_dist = obj.distance - (i + 1) * 4
                    tp_bev   = world_to_panel(max(5, fut_dist), obj.lat_offset)
                    if ox <= tp_bev[0] <= ox+self.W and oy <= tp_bev[1] <= oy+self.H:
                        pygame.draw.line(surf, (*col, 120), prev_bev, tp_bev, 1)
                        prev_bev = tp_bev

        # ── Ego vehicle ──────────────────────────────────────────────────────
        car_w, car_h = 18, 32
        car_surf = alpha_surface((car_w, car_h))
        pygame.draw.rect(car_surf, (*BLUE, 240), car_surf.get_rect(), border_radius=5)
        # Windscreen
        pygame.draw.rect(car_surf, (*CYAN, 200),
                         (3, 8, car_w-6, 10), border_radius=2)
        rot  = math.degrees(frame.steering * 3)
        rcar = pygame.transform.rotate(car_surf, rot)
        surf.blit(rcar, (ex - rcar.get_width()//2, ey - rcar.get_height()//2))

        # Heading arrow
        arrow_len = 30
        dx = math.sin(frame.steering * 2) * arrow_len
        dy = -(math.cos(frame.steering * 2)) * arrow_len
        pygame.draw.line(surf, CYAN, (ex, ey), (int(ex+dx), int(ey+dy)), 2)

        # ── Speed annotation under ego ───────────────────────────────────────
        draw_text(surf, f"{frame.ego_speed*3.6:.1f} km/h",
                  (ex, ey + 28), self._fonts["small"], CYAN, anchor="center")

        # ── Panel label ──────────────────────────────────────────────────────
        draw_text(surf, "BIRD'S-EYE VIEW",
                  (ox + 10, oy + 8), self._fonts["tiny"], LIGHT_GRAY)

        # ── Lead TTC meter ───────────────────────────────────────────────────
        ttc_txt = f"TTC: {frame.leading_ttc:.1f}s" \
            if frame.leading_ttc < 50 else "TTC: --"
        ttc_col = RED if frame.leading_ttc < 1.5 else \
                  ORANGE if frame.leading_ttc < 2.5 else GREEN
        draw_text(surf, ttc_txt,
                  (ox + self.W - 10, oy + 8),
                  self._fonts["small"], ttc_col, anchor="topright")

        pygame.draw.rect(surf, GRID_COLOR, (ox, oy, self.W, self.H), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Panel C: Dashboard ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardPanel:
    """Vehicle control dashboard with gauges and ADAS status lights."""

    W, H = 640, 270

    def __init__(self, fonts: dict):
        self._fonts = fonts

    def render(self, surf: pygame.Surface, frame: PerceptionFrame,
               ox: int = 0, oy: int = 0):
        pygame.draw.rect(surf, PANEL_BG, (ox, oy, self.W, self.H))

        # ── Speedometer ─────────────────────────────────────────────────────
        self._draw_speedometer(surf, frame.ego_speed * 3.6,
                               ox + 120, oy + 140, 100)

        # ── Brake / Throttle bars ────────────────────────────────────────────
        self._draw_bar(surf, "THROTTLE", frame.throttle, GREEN,
                       ox + 255, oy + 55, 22, 170)
        self._draw_bar(surf, "BRAKE",    frame.brake,    RED,
                       ox + 295, oy + 55, 22, 170)

        # ── Steering wheel ───────────────────────────────────────────────────
        self._draw_steering(surf, frame.steering,
                            ox + 395, oy + 140, 70)

        # ── ADAS status lights ───────────────────────────────────────────────
        lx = ox + 490
        self._draw_status_light(surf, "ACC",  frame.acc_active,      lx, oy + 55)
        self._draw_status_light(surf, "LKA",  frame.lka_active,      lx, oy + 100)

        aeb_color = AEB_COLORS.get(frame.aeb_state, MID_GRAY)
        aeb_on    = frame.aeb_state != "Inactive"
        self._draw_status_light(surf, "AEB",  aeb_on, lx, oy + 145,
                                on_color=aeb_color)
        self._draw_status_light(surf, "TL-STOP",
                                any(o.tl_color in ("red","amber") and o.distance < 35
                                    for o in frame.objects),
                                lx, oy + 190, on_color=RED)

        # ── Traffic-light icon ───────────────────────────────────────────────
        self._draw_traffic_light(surf, frame, ox + 605, oy + 95)

        # ── Numeric readouts ─────────────────────────────────────────────────
        draw_text(surf, f"{frame.ego_speed*3.6:>6.1f} km/h",
                  (ox + 120, oy + 18), self._fonts["bold_sm"], CYAN, anchor="center")
        draw_text(surf, f"Accel: {frame.ego_accel:+.2f} m/s²",
                  (ox + 120, oy + 240), self._fonts["tiny"], LIGHT_GRAY, anchor="center")

        # AEB state text
        aeb_c = AEB_COLORS.get(frame.aeb_state, LIGHT_GRAY)
        draw_text(surf, frame.aeb_state,
                  (ox + self.W // 2, oy + 248),
                  self._fonts["small"], aeb_c, anchor="center")

        draw_text(surf, "DASHBOARD",
                  (ox + 10, oy + 8), self._fonts["tiny"], LIGHT_GRAY)

        pygame.draw.rect(surf, GRID_COLOR, (ox, oy, self.W, self.H), 1)

    # ─── Sub-components ───────────────────────────────────────────────────────

    def _draw_speedometer(self, surf, speed_kmh, cx, cy, r):
        # Arc background
        pygame.draw.circle(surf, (30, 35, 42), (cx, cy), r)
        pygame.draw.circle(surf, GRID_COLOR,   (cx, cy), r, 2)

        max_speed = 180.0
        start_deg = 220;  sweep = 280

        # Coloured arc segments: green → yellow → red
        for i in range(sweep):
            angle    = math.radians(start_deg - i)
            t        = i / sweep
            seg_col  = lerp_color(GREEN, RED, t)
            x1 = cx + int((r - 6) * math.cos(angle))
            y1 = cy - int((r - 6) * math.sin(angle))
            x2 = cx + int((r - 2) * math.cos(angle))
            y2 = cy - int((r - 2) * math.sin(angle))
            pygame.draw.line(surf, seg_col, (x1, y1), (x2, y2), 2)

        # Tick marks
        for v in range(0, 181, 20):
            a  = math.radians(start_deg - (v / max_speed) * sweep)
            lf = 0.75 if v % 40 == 0 else 0.84
            x1 = cx + int(r * math.cos(a));        y1 = cy - int(r * math.sin(a))
            x2 = cx + int(r * lf * math.cos(a));   y2 = cy - int(r * lf * math.sin(a))
            pygame.draw.line(surf, LIGHT_GRAY, (x1,y1), (x2,y2), 1)
            if v % 40 == 0:
                tx = cx + int((r - 20) * math.cos(a))
                ty = cy - int((r - 20) * math.sin(a))
                draw_text(surf, str(v), (tx, ty), self._fonts["tiny"],
                          LIGHT_GRAY, anchor="center")

        # Needle
        clamped = min(speed_kmh, max_speed)
        needle_angle = math.radians(start_deg - (clamped / max_speed) * sweep)
        nx = cx + int((r - 12) * math.cos(needle_angle))
        ny = cy - int((r - 12) * math.sin(needle_angle))
        pygame.draw.line(surf, WHITE, (cx, cy), (nx, ny), 3)
        pygame.draw.circle(surf, CYAN, (cx, cy), 6)   # hub

        # Digital readout
        draw_text(surf, f"{speed_kmh:.0f}",
                  (cx, cy + 28), self._fonts["bold_sm"], WHITE, anchor="center")
        draw_text(surf, "km/h",
                  (cx, cy + 46), self._fonts["tiny"], MID_GRAY, anchor="center")

    def _draw_bar(self, surf, label, value, color, x, y, bw, bh):
        # Background
        pygame.draw.rect(surf, (30, 35, 42), (x, y, bw, bh), border_radius=4)
        # Fill
        fill_h = int(bh * value)
        fill_y = y + bh - fill_h
        if fill_h > 0:
            mix = lerp_color((30,35,42), color, value)
            pygame.draw.rect(surf, mix,
                             (x, fill_y, bw, fill_h), border_radius=4)
        # Border
        pygame.draw.rect(surf, GRID_COLOR, (x, y, bw, bh), 1, border_radius=4)
        # Label
        draw_text(surf, label, (x + bw//2, y - 14),
                  self._fonts["tiny"], LIGHT_GRAY, anchor="center")
        draw_text(surf, f"{value*100:.0f}%",
                  (x + bw//2, y + bh + 4),
                  self._fonts["tiny"], color, anchor="center")

    def _draw_steering(self, surf, angle_rad, cx, cy, r):
        pygame.draw.circle(surf, (30, 35, 42), (cx, cy), r)
        pygame.draw.circle(surf, GRID_COLOR,   (cx, cy), r, 2)

        # Steering wheel rim
        rim_col = lerp_color(BLUE, RED, min(1, abs(angle_rad) * 4))
        pygame.draw.circle(surf, rim_col, (cx, cy), r - 8, 3)

        # Spokes (rotated by steering angle)
        rot_deg = math.degrees(angle_rad) * 10
        for spoke in (0, 120, 240):
            a = math.radians(rot_deg + spoke)
            sx = cx + int((r - 8) * math.cos(a))
            sy = cy - int((r - 8) * math.sin(a))
            pygame.draw.line(surf, LIGHT_GRAY, (cx, cy), (sx, sy), 3)

        pygame.draw.circle(surf, DARK_GRAY, (cx, cy), 12)

        draw_text(surf, "STEER",
                  (cx, cy + r + 10),
                  self._fonts["tiny"], LIGHT_GRAY, anchor="center")
        draw_text(surf, f"{math.degrees(angle_rad):+.1f}°",
                  (cx, cy - r - 14),
                  self._fonts["tiny"], LIGHT_GRAY, anchor="center")

    def _draw_status_light(self, surf, label, active, x, y,
                            on_color=GREEN, off_color=DARK_GRAY):
        bw, bh = 110, 32
        col    = on_color if active else off_color
        bg     = lerp_color(DARK_GRAY, col, 0.3) if active else (28, 30, 36)
        draw_rounded_rect(surf, bg, pygame.Rect(x, y, bw, bh), radius=6)
        pygame.draw.rect(surf, col, (x, y, bw, bh), 1, border_radius=6)
        # Indicator dot
        dot_x = x + 14; dot_y = y + bh // 2
        pygame.draw.circle(surf, col if active else MID_GRAY, (dot_x, dot_y), 6)
        if active:
            pygame.draw.circle(surf, WHITE, (dot_x, dot_y), 3)
        draw_text(surf, label, (x + 26, y + bh // 2),
                  self._fonts["small"], WHITE if active else MID_GRAY,
                  anchor="midleft")

    def _draw_traffic_light(self, surf, frame, cx, cy):
        tl_state = "unknown"
        for obj in frame.objects:
            if obj.label == "TrafficLight":
                tl_state = obj.tl_color
                break

        bx, by, bw, bh = cx - 18, cy - 55, 36, 110
        draw_rounded_rect(surf, (30, 35, 42), pygame.Rect(bx, by, bw, bh), radius=8)
        pygame.draw.rect(surf, GRID_COLOR, (bx, by, bw, bh), 1, border_radius=8)

        bulb_data = [("red",   RED,   cy - 32),
                     ("amber", ORANGE, cy),
                     ("green", GREEN,  cy + 32)]
        for state, col, by2 in bulb_data:
            active = tl_state == state
            pygame.draw.circle(surf, col if active else lerp_color(col, BLACK, 0.7),
                               (cx, by2), 11)
            if active:
                # Glow
                glow = alpha_surface((50, 50))
                pygame.draw.circle(glow, (*col, 40), (25, 25), 22)
                surf.blit(glow, (cx - 25, by2 - 25))

        draw_text(surf, tl_state.upper() if tl_state != "unknown" else "TL",
                  (cx, by + bh + 10), self._fonts["tiny"],
                  {"red": RED, "amber": ORANGE, "green": GREEN}.get(tl_state, MID_GRAY),
                  anchor="center")


# ═══════════════════════════════════════════════════════════════════════════════
# ── Panel D: Time-series plots ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class PlotsPanel:
    """Rolling time-series charts for speed, TTC, and lead distance."""

    W, H = 640, 270

    def __init__(self, fonts: dict):
        self._fonts   = fonts
        self._speed   : deque[float] = deque(maxlen=HISTORY)
        self._ttc     : deque[float] = deque(maxlen=HISTORY)
        self._lead_d  : deque[float] = deque(maxlen=HISTORY)
        self._brake   : deque[float] = deque(maxlen=HISTORY)
        self._throttle: deque[float] = deque(maxlen=HISTORY)

    def update(self, frame: PerceptionFrame):
        self._speed.append(frame.ego_speed * 3.6)
        self._ttc.append(  min(frame.leading_ttc, 10.0) if frame.leading_ttc < 50 else 10.0)
        self._lead_d.append(min(frame.lead_distance, 100.0))
        self._brake.append(frame.brake * 100)
        self._throttle.append(frame.throttle * 100)

    def render(self, surf: pygame.Surface, frame: PerceptionFrame,
               ox: int = 0, oy: int = 0):
        pygame.draw.rect(surf, PANEL_BG, (ox, oy, self.W, self.H))

        pw = (self.W - 40) // 3     # plot width
        ph = self.H - 50            # plot height
        pad= 10

        plots = [
            ("Speed (km/h)", self._speed,   0,  120,  CYAN,   None),
            ("TTC (s)",       self._ttc,    0,   10,  ORANGE, [(2.5, ORANGE), (1.2, RED)]),
            ("Lead Dist (m)", self._lead_d, 0,  100,  GREEN,  [(15, ORANGE), (8, RED)]),
        ]

        for i, (title, data, lo, hi, color, thresholds) in enumerate(plots):
            px = ox + pad + i * (pw + pad)
            py = oy + 35
            self._draw_plot(surf, data, lo, hi, px, py, pw, ph,
                            color, title, thresholds)

        # Current value readouts
        cv_y = oy + 15
        for i, (label, val, unit, col) in enumerate([
            ("Speed",    self._speed[-1]   if self._speed   else 0, "km/h", CYAN),
            ("TTC",      self._ttc[-1]     if self._ttc     else 0, "s",    ORANGE),
            ("LeadDist", self._lead_d[-1]  if self._lead_d  else 0, "m",    GREEN),
        ]):
            ix = ox + pad + i * (pw + pad) + pw // 2
            draw_text(surf, f"{val:>5.1f} {unit}",
                      (ix, cv_y), self._fonts["small"], col, anchor="center")

        draw_text(surf, "TIME-SERIES PLOTS",
                  (ox + 10, oy + 8), self._fonts["tiny"], LIGHT_GRAY)

        pygame.draw.rect(surf, GRID_COLOR, (ox, oy, self.W, self.H), 1)

    def _draw_plot(self, surf, data: deque, lo, hi, px, py, pw, ph,
                   color, title, thresholds):
        # Background
        pygame.draw.rect(surf, (22, 26, 32), (px, py, pw, ph))
        pygame.draw.rect(surf, GRID_COLOR,   (px, py, pw, ph), 1)

        # Horizontal grid lines + labels
        for v in np.linspace(lo, hi, 5):
            gy = py + int(ph * (1 - (v - lo) / (hi - lo + 1e-9)))
            pygame.draw.line(surf, GRID_COLOR, (px, gy), (px+pw, gy), 1)
            draw_text(surf, f"{v:.0f}", (px - 2, gy), self._fonts["tiny"],
                      MID_GRAY, anchor="midright")

        # Threshold lines
        if thresholds:
            for th_val, th_col in thresholds:
                th_y = py + int(ph * (1 - (th_val - lo) / (hi - lo + 1e-9)))
                if py <= th_y <= py + ph:
                    pygame.draw.line(surf, th_col, (px, th_y), (px+pw, th_y), 1)

        if len(data) < 2:
            return

        # Data polyline
        arr = list(data)
        n   = len(arr)
        pts = []
        for i, v in enumerate(arr):
            cx2 = px + int(pw * i / (HISTORY - 1))
            cy2 = py + int(ph * (1 - (v - lo) / (hi - lo + 1e-9)))
            cy2 = max(py, min(py + ph, cy2))
            pts.append((cx2, cy2))

        # Fill under curve (alpha)
        if len(pts) >= 2:
            fill_pts = [(px, py + ph)] + pts + [(pts[-1][0], py + ph)]
            fs = alpha_surface((pw, ph))
            fp = [(x - px, y - py) for x, y in fill_pts]
            pygame.draw.polygon(fs, (*color, 35), fp)
            surf.blit(fs, (px, py))
            pygame.draw.lines(surf, color, False, pts, 2)

            # Last-point dot
            lx, ly = pts[-1]
            pygame.draw.circle(surf, WHITE, (lx, ly), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main visualizer  ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class ADASVisualizer:
    WIN_W = 1280
    WIN_H = 720
    TITLE  = "ADAS Perception Stack – Live Demo"

    # Panel layout (x, y, w, h)
    LAYOUT = {
        "camera":    (   0,   0, 760, 450),
        "bev":       ( 760,   0, 520, 450),
        "dashboard": (   0, 450, 640, 270),
        "plots":     ( 640, 450, 640, 270),
    }

    def __init__(self, fps: int = 30, seed: int = 42):
        pygame.init()
        pygame.display.set_caption(self.TITLE)

        self._screen = pygame.display.set_mode(
            (self.WIN_W, self.WIN_H), pygame.DOUBLEBUF)

        self._clock  = pygame.time.Clock()
        self._fps    = fps
        self._data   = SyntheticADAS(seed=seed)
        self._fonts  = self._load_fonts()
        self._panels = {
            "camera":    CameraPanel(self._fonts),
            "bev":       BirdEyePanel(self._fonts),
            "dashboard": DashboardPanel(self._fonts),
            "plots":     PlotsPanel(self._fonts),
        }
        self._running   = False
        self._paused    = False
        self._show_help = False
        self._frame     = None

    # ─── public ──────────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        while self._running:
            self._handle_events()
            if not self._paused:
                self._frame = self._data.step()
                self._panels["plots"].update(self._frame)

            self._render()
            self._clock.tick(self._fps)

        pygame.quit()

    # ─── private ─────────────────────────────────────────────────────────────

    def _load_fonts(self) -> dict:
        fonts = {}
        try:
            mono = "Consolas"      # Windows
            sans = "Segoe UI"
            pygame.font.SysFont(mono, 12)   # test availability
        except Exception:
            mono = sans = None     # fall back to pygame default

        sizes = {
            "tiny":     (mono, 11),
            "small":    (mono, 13),
            "medium":   (sans, 15),
            "bold_sm":  (sans, 14),
            "large":    (sans, 20),
            "huge":     (sans, 28),
        }
        for name, (family, size) in sizes.items():
            bold = "bold" in name
            fonts[name] = pygame.font.SysFont(family, size, bold=bold) \
                if family else pygame.font.Font(None, size + 6)
        return fonts

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._running = False
                elif ev.key == pygame.K_SPACE:
                    self._paused = not self._paused
                elif ev.key == pygame.K_h:
                    self._show_help = not self._show_help
                elif ev.key == pygame.K_r:
                    # Reset
                    self._data   = SyntheticADAS()
                    self._panels["plots"] = PlotsPanel(self._fonts)

    def _render(self):
        self._screen.fill(BLACK)

        if self._frame is None:
            return

        # ── Render each panel ────────────────────────────────────────────────
        panel_map = {
            "camera":    (self._panels["camera"],    self._frame),
            "bev":       (self._panels["bev"],       self._frame),
            "dashboard": (self._panels["dashboard"], self._frame),
            "plots":     (self._panels["plots"],     self._frame),
        }
        for key, (panel, data) in panel_map.items():
            x, y, w, h = self.LAYOUT[key]
            panel.render(self._screen, data, x, y)

        # ── Top status bar overlay ───────────────────────────────────────────
        self._draw_status_bar()

        # ── Help overlay (H key) ─────────────────────────────────────────────
        if self._show_help:
            self._draw_help()

        # ── Pause overlay ────────────────────────────────────────────────────
        if self._paused:
            draw_text(self._screen, "⏸  PAUSED  (SPACE to resume)",
                      (self.WIN_W // 2, self.WIN_H // 2),
                      self._fonts["large"], YELLOW, anchor="center")

        pygame.display.flip()

    def _draw_status_bar(self):
        """Thin top status bar across full width."""
        bar_h = 0   # integrated into panel titles; kept for future use

    def _draw_help(self):
        help_surf = alpha_surface((400, 200))
        pygame.draw.rect(help_surf, (10, 12, 18, 230),
                         help_surf.get_rect(), border_radius=12)
        pygame.draw.rect(help_surf, (*GRID_COLOR, 200),
                         help_surf.get_rect(), 1, border_radius=12)
        self._screen.blit(help_surf,
                          (self.WIN_W // 2 - 200, self.WIN_H // 2 - 100))
        lines = [
            "KEYBOARD SHORTCUTS",
            "",
            "  SPACE   Pause / Resume",
            "  R       Reset simulation",
            "  H       Toggle this help",
            "  Q / ESC Quit",
        ]
        for i, line in enumerate(lines):
            col = CYAN if i == 0 else WHITE
            draw_text(self._screen, line,
                      (self.WIN_W // 2, self.WIN_H // 2 - 80 + i * 24),
                      self._fonts["small"], col, anchor="center")


# ═══════════════════════════════════════════════════════════════════════════════
# ── Entry point ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ADAS Perception Visualizer")
    parser.add_argument("--fps",   type=int, default=30,  help="Target frame-rate")
    parser.add_argument("--seed",  type=int, default=42,  help="RNG seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  ADAS Perception Stack – Visualization Demo")
    print("=" * 60)
    print(f"  Resolution : 1280 × 720")
    print(f"  FPS target : {args.fps}")
    print(f"  RNG seed   : {args.seed}")
    print()
    print("  Panels:")
    print("    A (top-left)    Camera view + detections + lanes")
    print("    B (top-right)   Bird's-eye view (top-down map)")
    print("    C (bottom-left) Vehicle dashboard + ADAS lights")
    print("    D (bottom-right)Time-series plots (speed/TTC/dist)")
    print()
    print("  Keys: SPACE=pause  R=reset  H=help  Q=quit")
    print("=" * 60)

    vis = ADASVisualizer(fps=args.fps, seed=args.seed)
    vis.run()


if __name__ == "__main__":
    main()
