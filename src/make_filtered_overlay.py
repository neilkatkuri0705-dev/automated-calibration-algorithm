"""
make_filtered_overlay.py — render a trajectory overlay from the CLEANED matrix
(only the kept trajectories), colored by direction. This is the "filtered"
version of <camera>_trajectories.png (which shows ALL raw trajectories).

    python make_filtered_overlay.py                 # all cameras
    python make_filtered_overlay.py <camera>

Writes outputs/<camera>/<camera>_trajectories_filtered.png
Needs the cleaned matrix + the representative frame (both from run_pipeline).
"""

import os
import sys
import glob
import json

import cv2
import numpy as np

import config

DOWN_COLOR = (0, 0, 255)   # red  (BGR) = moving down-image
UP_COLOR = (255, 0, 0)     # blue (BGR) = moving up-image


def render(camera):
    out_dir = os.path.join(config.OUTPUT_DIR, camera)
    cleaned_path = os.path.join(out_dir, f"{camera}_velocity_matrix_cleaned.json")
    if not os.path.exists(cleaned_path):
        print(f"{camera}: no cleaned matrix, skipping")
        return
    with open(cleaned_path) as f:
        matrix = json.load(f)

    # background: prefer the representative frame; fall back to the raw overlay
    rep = os.path.join(out_dir, "llm_inputs", "representative.png")
    base = cv2.imread(rep) if os.path.exists(rep) else None
    if base is None:
        raw_overlay = os.path.join(out_dir, f"{camera}_trajectories.png")
        base = cv2.imread(raw_overlay)
    if base is None:
        base = np.zeros((matrix.get("H", 270), matrix.get("W", 480), 3), np.uint8)

    vis = base.copy()
    drawn = 0
    for sl in matrix["slices"]:
        pts = sl["positions"]
        if len(pts) < 2:
            continue
        color = DOWN_COLOR if np.sin(np.radians(sl["heading_deg"])) >= 0 else UP_COLOR
        arr = np.array(pts, dtype=np.int32)
        for j in range(len(arr) - 1):
            cv2.line(vis, tuple(arr[j]), tuple(arr[j + 1]), color, 1, cv2.LINE_AA)
        cv2.arrowedLine(vis, tuple(arr[-2]), tuple(arr[-1]), color, 1, cv2.LINE_AA, tipLength=0.4)
        drawn += 1

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, f"{camera}: {drawn} cleaned trajectories (red=away, blue=toward)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    out = os.path.join(out_dir, f"{camera}_trajectories_filtered.png")
    cv2.imwrite(out, vis)
    print(f"{camera}: {drawn} cleaned trajectories -> {out}")


def main():
    if len(sys.argv) > 1:
        render(sys.argv[1])
        return
    for p in glob.glob(os.path.join(config.OUTPUT_DIR, "*",
                                    "*_velocity_matrix_cleaned.json")):
        render(json.load(open(p))["camera"])


if __name__ == "__main__":
    main()
