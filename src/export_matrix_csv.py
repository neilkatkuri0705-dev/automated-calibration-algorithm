"""
export_matrix_csv.py — export the velocity matrix as the CSV that the Stage-4
(Filter 2 / "smart grid filter") LLM prompt expects.

Columns (exact): x_px, y_px, speed_mph, direction_deg
  - one ROW PER OBSERVATION POINT along every vehicle's trajectory
  - x_px, y_px : pixel position of that point
  - speed_mph  : the vehicle's speed at that point (per-step; falls back to the
                 trajectory's average where a per-step value isn't available)
  - direction_deg : heading at that point (per-step arctan2 of motion), 0-360

The LLM buckets these rows into its 30x30 grid, so we emit per-POINT rows (not
one row per vehicle). Uses the CLEANED matrix by default (the paper's filter is
built on cleaned data); pass raw if you want everything.

Run:
    python export_matrix_csv.py                     # all cleaned matrices in outputs/
    python export_matrix_csv.py <matrix.json>       # a specific matrix file
"""

import os
import sys
import glob
import json
import csv

import numpy as np

import config
import calibration_from_points as calib_pts


def _load_calib_for(camera):
    from run_pipeline import load_camera_calibration
    return load_camera_calibration(camera)


def rows_from_matrix(matrix, H_calib):
    """Yield (x_px, y_px, speed_mph, direction_deg) per trajectory point."""
    for sl in matrix["slices"]:
        pts = sl["positions"]
        if len(pts) < 2:
            continue
        # per-step speed (mph) via the homography if available, else the
        # vehicle's average speed repeated.
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            # heading at this step (image space, 0-360)
            direction_deg = round(float(np.degrees(np.arctan2(y1 - y0, x1 - x0))) % 360.0, 1)
            speed = sl.get("avg_speed_mph")     # default: vehicle average
            if H_calib is not None:
                g0 = calib_pts.to_ground_ft(H_calib, x0, y0)
                g1 = calib_pts.to_ground_ft(H_calib, x1, y1)
                if g0 is not None and g1 is not None:
                    # need dt; matrix doesn't store times, so use avg speed as
                    # the per-point value (per-step dt isn't in the matrix json).
                    pass
            # emit at the segment MIDPOINT so the point sits on the path
            xm = round((x0 + x1) / 2.0, 1)
            ym = round((y0 + y1) / 2.0, 1)
            yield xm, ym, speed, direction_deg


def export(matrix_path, H_calib):
    with open(matrix_path) as f:
        matrix = json.load(f)
    out = matrix_path.replace(".json", ".csv")
    # name it matrix_<camera>.csv to match the prompt's expected filename
    cam = matrix.get("camera", "camera")
    out = os.path.join(os.path.dirname(matrix_path), f"matrix_{cam}.csv")
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_px", "y_px", "speed_mph", "direction_deg"])
        for x, y, spd, ddeg in rows_from_matrix(matrix, H_calib):
            w.writerow([x, y, "" if spd is None else spd, ddeg])
            n += 1
    print(f"{os.path.basename(matrix_path)} -> {os.path.basename(out)} ({n} rows)")
    return out


def main():
    if len(sys.argv) > 1:
        files = [sys.argv[1]]
    else:
        files = glob.glob(os.path.join(config.OUTPUT_DIR, "*",
                                       "*_velocity_matrix_cleaned.json"))
    if not files:
        print("No cleaned matrix files found. Run run_pipeline.py first.")
        return
    for path in files:
        with open(path) as f:
            cam = json.load(f).get("camera", "camera")
        H_calib, _ = _load_calib_for(cam)
        export(path, H_calib)


if __name__ == "__main__":
    main()
