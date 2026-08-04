"""
clean_matrix.py — INTENSE cleaning of trajectories. Strays must be impossible.

A trajectory survives ONLY if it is a real, coherent vehicle path:
  (a) LONG ENOUGH: >= MIN_TRAJ_POINTS points AND travels >= MIN_TRAVEL_PX pixels
      (drops fragments, jitter, and frozen sign/pole tracks).
  (b) COHERENT DIRECTION: its per-step headings don't wander — heading standard
      deviation <= MAX_HEADING_STD_DEG (a real car goes roughly straight; the
      fragmented cross-crossing junk does not).
  (c) ON A DOMINANT FLOW: overall heading within STRAY_TOL_DEG of one of the two
      dominant carriageway directions (drops anything pointing the wrong way).
  (d) SPEED BAND: within the 25-75 percentile of speeds (drops outliers).

Works on the trajectory list (each slice has positions + per-point speeds/times
if present). Writes the cleaned matrix; leaves the raw one intact.
"""

import os
import sys
import glob
import json

import numpy as np

import config

MIN_POINTS = config.MIN_TRAJ_POINTS
MIN_TRAVEL_PX = config.MIN_TRAVEL_PX
STRAY_TOL_DEG = config.STRAY_TOL_DEG
MAX_HEADING_STD = config.MAX_HEADING_STD_DEG
PCTL_LOW = config.TRAJ_PCTL_LOW
PCTL_HIGH = config.TRAJ_PCTL_HIGH


def _ang_diff(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _path_len(pos):
    if len(pos) < 2:
        return 0.0
    p = np.asarray(pos, float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _straight_line_dist(pos):
    if len(pos) < 2:
        return 0.0
    p = np.asarray(pos, float)
    return float(np.linalg.norm(p[-1] - p[0]))


def _heading_std(pos):
    """circular std of per-step headings (deg). Low = straight, high = wiggly."""
    p = np.asarray(pos, float)
    if len(p) < 3:
        return 0.0
    d = np.diff(p, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])              # radians
    # circular std
    s = np.sqrt(-2.0 * np.log(np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang))) + 1e-9))
    return float(np.degrees(s))


def _dominant_directions(headings):
    if len(headings) == 0:
        return []
    hist, edges = np.histogram(headings, bins=36, range=(0, 360))
    centers = (edges[:-1] + edges[1:]) / 2.0
    dirs = []
    for k in np.argsort(hist)[::-1]:
        if hist[k] == 0:
            break
        c = centers[k]
        if all(_ang_diff(c, d) > 45 for d in dirs):
            dirs.append(c)
        if len(dirs) >= 2:
            break
    return dirs


def clean(matrix):
    slices = matrix["slices"]
    n0 = len(slices)
    d_short = d_frozen = d_wiggly = d_stray = d_speed = 0

    survivors = []
    for sl in slices:
        pos = sl["positions"]
        # (a) long enough + actually moves
        if sl.get("n_points", len(pos)) < MIN_POINTS or sl.get("avg_speed_mph") is None:
            d_short += 1; continue
        if _straight_line_dist(pos) < MIN_TRAVEL_PX:
            d_frozen += 1; continue                 # frozen / jitter in place
        # (b) coherent (goes roughly straight)
        if _heading_std(pos) > MAX_HEADING_STD:
            d_wiggly += 1; continue                 # fragmented / wandering
        survivors.append(sl)

    # (c) on a dominant flow direction
    dirs = _dominant_directions(np.array([s["heading_deg"] for s in survivors]))
    if dirs:
        kept = []
        for s in survivors:
            if any(_ang_diff(s["heading_deg"], d) <= STRAY_TOL_DEG for d in dirs):
                kept.append(s)
            else:
                d_stray += 1
        survivors = kept

    # (d) speed band
    lo = hi = None
    if survivors:
        sp = np.array([s["avg_speed_mph"] for s in survivors])
        lo = float(np.percentile(sp, PCTL_LOW)); hi = float(np.percentile(sp, PCTL_HIGH))
        band = []
        for s in survivors:
            if lo <= s["avg_speed_mph"] <= hi:
                band.append(s)
            else:
                d_speed += 1
        survivors = band

    cleaned = dict(matrix)
    cleaned["slices"] = survivors
    cleaned["n"] = len(survivors)
    cleaned["cleaned"] = True
    cleaned["cleaning"] = {
        "n_before": n0, "n_after": len(survivors),
        "dropped_short": d_short, "dropped_frozen": d_frozen,
        "dropped_wiggly": d_wiggly, "dropped_stray": d_stray,
        "dropped_speed_band": d_speed,
        "dominant_directions_deg": [round(d, 1) for d in dirs],
        "params": {"min_points": MIN_POINTS, "min_travel_px": MIN_TRAVEL_PX,
                   "max_heading_std_deg": MAX_HEADING_STD, "stray_tol_deg": STRAY_TOL_DEG,
                   "speed_pctl": [PCTL_LOW, PCTL_HIGH]},
        "speed_band_mph": [round(lo, 1), round(hi, 1)] if lo is not None else None,
    }
    return cleaned


def clean_file(path):
    with open(path) as f:
        matrix = json.load(f)
    cleaned = clean(matrix)
    out = path.replace("_velocity_matrix.json", "_velocity_matrix_cleaned.json")
    if out == path:
        out = path.replace(".json", "_cleaned.json")
    with open(out, "w") as f:
        json.dump(cleaned, f, separators=(",", ":"))
    c = cleaned["cleaning"]
    print(f"{os.path.basename(path)}: n {c['n_before']} -> {c['n_after']} "
          f"(short {c['dropped_short']}, frozen {c['dropped_frozen']}, "
          f"wiggly {c['dropped_wiggly']}, stray {c['dropped_stray']}, "
          f"speed {c['dropped_speed_band']}); dirs {c['dominant_directions_deg']} "
          f"band {c['speed_band_mph']}")
    print(f"  wrote {out}")


def main():
    if len(sys.argv) > 1:
        files = [sys.argv[1]]
    else:
        files = [f for f in glob.glob(os.path.join(config.OUTPUT_DIR, "*",
                 "*_velocity_matrix.json")) if "cleaned" not in f]
    if not files:
        print("No matrices found. Run run_pipeline.py first.")
        return
    for f in files:
        clean_file(f)


if __name__ == "__main__":
    main()
