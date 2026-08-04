"""
export_raw_trajectories.py — trajectory data with BOTH raw pixel speed AND the
algorithm's real-world speed (mph), for sending to a mentor/collaborator.

To avoid any timing mistakes, this reads the pipeline's authoritative
matrix_<camera>.csv (which already has the correct real timestamps and
speed_mph) and simply ADDS a speed_px_per_s column computed from the same
real timestamps. One source of truth -> no stride/timestamp bugs.

    python export_raw_trajectories.py            # all cameras
    python export_raw_trajectories.py <camera>

Reads  outputs/<camera>/matrix_<camera>.csv
Writes outputs/<camera>/raw_trajectories_<camera>.csv with columns:
    id, x_px, y_px, timestamp_s, speed_px_per_s, speed_mph, direction_deg
  - speed_px_per_s : pixel speed (NOT comparable across the frame)
  - speed_mph      : the algorithm's real-world speed (pixel->real conversion)
  - direction_deg  : image-space heading (0=right,90=down,180=left,270=up)
"""

import os
import sys
import csv
import glob

import numpy as np

import config


def export(camera):
    out_dir = os.path.join(config.OUTPUT_DIR, camera)
    src = os.path.join(out_dir, f"matrix_{camera}.csv")
    if not os.path.exists(src):
        print(f"{camera}: no matrix_{camera}.csv, skipping")
        return

    with open(src, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    ii, xi, yi = idx["id"], idx["x_px"], idx["y_px"]
    ti, spi, di = idx["timestamp_s"], idx["speed_mph"], idx["direction_deg"]

    # group by id to compute pixel speed from consecutive points (real dt)
    out = os.path.join(out_dir, f"raw_trajectories_{camera}.csv")
    n = 0
    prev = {}   # id -> (x, y, t)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_px", "y_px", "timestamp_s",
                    "speed_px_per_s", "speed_mph", "direction_deg"])
        for r in data:
            vid = r[ii]
            x, y, t = float(r[xi]), float(r[yi]), float(r[ti])
            speed_px = ""
            if vid in prev:
                x0, y0, t0 = prev[vid]
                dt = t - t0
                if dt > 0:
                    speed_px = round(np.hypot(x - x0, y - y0) / dt, 2)
            prev[vid] = (x, y, t)
            w.writerow([vid, r[xi], r[yi], r[ti], speed_px, r[spi], r[di]])
            n += 1
    print(f"{camera}: {n} points -> {out}")


def main():
    if len(sys.argv) > 1:
        export(sys.argv[1])
        return
    for p in glob.glob(os.path.join(config.OUTPUT_DIR, "*", "matrix_*.csv")):
        if p.endswith("_clean.csv"):
            continue
        cam = os.path.basename(os.path.dirname(p))
        export(cam)


if __name__ == "__main__":
    main()
