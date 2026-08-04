"""
clean_csv.py — clean the per-observation velocity-matrix CSV rows.

For EACH camera independently:
  1. DROP rows with a blank speed_mph or direction_deg.
  2. DROP speed rows outside that camera's own [25, 75] percentile band.

This narrows each camera's spread around its OWN real median (e.g. a camera that
measures 63 stays ~63, one that measures 48 stays ~48) — it does not force all
cameras to a common value.

Writes matrix_<camera>_clean.csv next to the original.

    python clean_csv.py            # all matrix_*.csv in outputs/
    python clean_csv.py <path/to/matrix_camera.csv>
"""

import os
import sys
import csv
import glob

import numpy as np

import config

PCTL_LOW = config.TRAJ_PCTL_LOW      # 25
PCTL_HIGH = config.TRAJ_PCTL_HIGH    # 75


def clean_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print(f"{os.path.basename(path)}: empty")
        return
    header, data = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    si = idx.get("speed_mph"); di = idx.get("direction_deg")
    if si is None or di is None:
        print(f"{os.path.basename(path)}: missing speed/direction columns, skip")
        return

    # 1. drop blanks
    kept, speeds, n_blank = [], [], 0
    for r in data:
        sv = r[si].strip() if si < len(r) else ""
        dv = r[di].strip() if di < len(r) else ""
        if sv == "" or dv == "":
            n_blank += 1
            continue
        try:
            speeds.append(float(sv)); kept.append(r)
        except ValueError:
            n_blank += 1

    # 2. drop rows outside this camera's own 25-75 percentile speed band
    n_out = 0
    lo = hi = None
    if speeds:
        lo = float(np.percentile(speeds, PCTL_LOW))
        hi = float(np.percentile(speeds, PCTL_HIGH))
        final = []
        for r in kept:
            v = float(r[si])
            if lo <= v <= hi:
                final.append(r)
            else:
                n_out += 1
        kept = final

    out = path.replace(".csv", "_clean.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

    med = np.median([float(r[si]) for r in kept]) if kept else float("nan")
    band = f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "n/a"
    print(f"{os.path.basename(path)}: {len(data)} -> {len(kept)} rows "
          f"(dropped {n_blank} blank, {n_out} outside {PCTL_LOW}-{PCTL_HIGH} band {band}); "
          f"median {med:.0f} mph")
    print(f"  wrote {os.path.basename(out)}")


def main():
    if len(sys.argv) > 1:
        files = [sys.argv[1]]
    else:
        files = [f for f in glob.glob(os.path.join(config.OUTPUT_DIR, "*", "matrix_*.csv"))
                 if not f.endswith("_clean.csv")]
    if not files:
        print("No matrix_*.csv found. Run run_pipeline.py first.")
        return
    for f in files:
        clean_csv(f)


if __name__ == "__main__":
    main()
