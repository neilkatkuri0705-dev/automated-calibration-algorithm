"""
calibration_from_points.py — build the pixel->real-world perspective transform
from 4 clicked road corners (calibration.json produced by the picker).

Tutorial method: 4 SOURCE points on the road (camera pixels) map to a TARGET
rectangle of known real size (feet). Lane width fixes the lateral side
(lanes x 12 ft); dash markings fix the length ((stripes-1) x 40 ft). Long
baseline -> well-conditioned homography -> stable speeds.

The homography maps the WHOLE road plane, so EVERY tracked vehicle gets a speed
(not just those inside the clicked rectangle). Only physically-degenerate steps
are excluded: points at/above the horizon line (denominator ~0) or mapping to
absurd ground distances, and per-step speeds above a plausibility ceiling.

Speed: transform each point to feet, distance between consecutive points /
elapsed time = ft/s, then x 0.681818 = mph.
"""

import os
import json
import numpy as np
import cv2

FT_PER_S_TO_MPH = 0.681818
MAX_GROUND_FT = 2000.0        # steps mapping beyond this are degenerate (near horizon)


def load_calibration(src_dir):
    """Load calibration.json (from the picker). Returns (H, meta) or (None, None)."""
    path = os.path.join(src_dir, "calibration.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        cal = json.load(f)
    src = np.float32(cal["source_px"])              # near_left, near_right, far_right, far_left
    W = float(cal["width_ft"])
    L = float(cal["length_ft"])
    dst = np.float32([[0, 0], [W, 0], [W, L], [0, L]])  # feet
    H = cv2.getPerspectiveTransform(src, dst)
    meta = {"width_ft": W, "length_ft": L,
            "condition_number": float(np.linalg.cond(H)),
            "source_px": cal["source_px"]}
    return H, meta


def to_ground_ft(H, x, y):
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    gx, gy = p[0] / p[2], p[1] / p[2]
    if abs(gx) > MAX_GROUND_FT or abs(gy) > MAX_GROUND_FT:
        return None                      # degenerate (near/behind horizon)
    return float(gx), float(gy)


def avg_speed_mph(points, times, H, ceiling_mph=120.0):
    """
    Average speed (mph) for a trajectory over the whole road plane. Every
    vehicle with >=1 valid step gets a speed. Excludes only degenerate steps
    (horizon/absurd distance) and per-step speeds above the ceiling.
    """
    if H is None or len(points) < 2:
        return None
    vs = []
    for i in range(1, len(points)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        g0 = to_ground_ft(H, points[i - 1][0], points[i - 1][1])
        g1 = to_ground_ft(H, points[i][0], points[i][1])
        if g0 is None or g1 is None:
            continue
        d_ft = np.hypot(g1[0] - g0[0], g1[1] - g0[1])
        mph = (d_ft / dt) * FT_PER_S_TO_MPH
        if mph <= ceiling_mph:
            vs.append(mph)
    if not vs:
        return None
    return float(np.median(vs))          # median = robust to a bad step
