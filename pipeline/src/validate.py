"""
validate.py — validation of the pixel-to-real-world conversion + figure
generation. Run AFTER run_pipeline.py has produced the matrices.

    python validate.py            # all cameras with a velocity matrix
    python validate.py <camera>

The conversion's contribution is the homography-based speed. Validation must be
INDEPENDENT of what we calibrated with (lane width 12 ft, dash 40 ft), or it is
circular. So we validate two independent ways:

  1. VEHICLE-WIDTH check (independent of calibration):
     Predict each detected vehicle's real-world width by mapping its box's
     bottom-left and bottom-right (ground-contact corners) through the
     homography and measuring the ground distance in feet. Compare the class
     median to known widths (car body ~6 ft, truck/bus ~8.5 ft). We calibrated
     with lane geometry, NOT individual vehicle widths, so this is independent
     evidence the scale is right.

  2. SPEED-DISTRIBUTION check:
     The distribution of vehicle speeds should sit around expected highway
     speeds (Interstate ~55-70 mph). We report median / IQR and plot it.

Requires the RAW matrix (has vehicle boxes via positions? no) -> we recompute
widths from the annotated detections stored during the run. Since the matrix
only stores center-bottom positions, vehicle-width validation needs the box
widths, which we re-derive by re-detecting a sample of frames. To keep it fast
and not re-run YOLO, this script uses the WIDTHS saved in
<camera>_vehicle_widths.json if present (written by run_pipeline when enabled);
otherwise it does a quick re-detection pass on a sample of frames.

Outputs per camera (outputs/<camera>/validation/):
  speed_distribution.png
  vehicle_width_validation.png
  validation.json    (numbers for the paper)
"""

import os
import sys
import glob
import json

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

import config
import calibration_from_points as calib_pts

VEHICLE = config.VEHICLE_CLASSES
CONF = 0.15
IMGSZ = config.DETECT_IMGSZ

# known real-world vehicle body widths (feet) — INDEPENDENT of calibration
KNOWN_WIDTH_FT = {"car": 6.0, "truck": 8.5, "bus": 8.5, "motorcycle": 3.0}


def _load_calib(camera):
    from run_pipeline import load_camera_calibration
    return load_camera_calibration(camera)


def _rep_video(camera):
    single = sorted(glob.glob(os.path.join(config.VIDEO_SINGLE_DIR, "*.mp4")))
    if camera == "single_camera" and single:
        return single[0]
    for v in sorted(glob.glob(os.path.join(config.VIDEO_MULTI_DIR, "*"))):
        if os.path.splitext(os.path.basename(v))[0] == camera:
            return v
    return None


def measure_vehicle_widths(video_path, model, H, sample_stride=8, max_frames=1500):
    """Re-detect a sample of frames; for each detection map the box's bottom-left
    and bottom-right corners to ground feet and record the width by class."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    widths = {k: [] for k in KNOWN_WIDTH_FT}
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret or fidx >= max_frames:
            break
        if fidx % sample_stride == 0:
            r = model.predict(frame, classes=list(VEHICLE.keys()),
                              conf=CONF, imgsz=IMGSZ, verbose=False)[0]
            if r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c in zip(xyxy, clss):
                    cls = VEHICLE.get(int(c))
                    if cls not in KNOWN_WIDTH_FT:
                        continue
                    # bottom-left and bottom-right ground-contact corners
                    gl = calib_pts.to_ground_ft(H, x1, y2)
                    gr = calib_pts.to_ground_ft(H, x2, y2)
                    if gl is None or gr is None:
                        continue
                    w_ft = abs(gr[0] - gl[0])       # lateral ground width
                    if 1 < w_ft < 20:               # sane range
                        widths[cls].append(w_ft)
        fidx += 1
    cap.release()
    return widths


def validate_camera(camera, model):
    out_dir = os.path.join(config.OUTPUT_DIR, camera)
    mat_path = os.path.join(out_dir, f"{camera}_velocity_matrix_cleaned.json")
    if not os.path.exists(mat_path):
        print(f"{camera}: no cleaned matrix yet, skipping")
        return
    with open(mat_path) as f:
        matrix = json.load(f)
    H, cal_meta = _load_calib(camera)
    if H is None:
        print(f"{camera}: not calibrated, cannot validate speeds")
        return

    vdir = os.path.join(out_dir, "validation")
    os.makedirs(vdir, exist_ok=True)
    result = {"camera": camera, "calibration": cal_meta}

    # --- 1. speed distribution ---
    speeds = [s["avg_speed_mph"] for s in matrix["slices"]
              if s.get("avg_speed_mph") is not None]
    if speeds:
        a = np.array(speeds)
        result["speed"] = {
            "n": len(a), "median_mph": round(float(np.median(a)), 1),
            "mean_mph": round(float(np.mean(a)), 1),
            "iqr_mph": [round(float(np.percentile(a, 25)), 1),
                        round(float(np.percentile(a, 75)), 1)],
            "min_mph": round(float(a.min()), 1), "max_mph": round(float(a.max()), 1),
        }
        plt.figure(figsize=(6, 4))
        plt.hist(a, bins=20, color="#3a7", edgecolor="white")
        plt.axvline(np.median(a), color="#333", ls="--",
                    label=f"median {np.median(a):.0f} mph")
        plt.xlabel("speed (mph)"); plt.ylabel("vehicles")
        plt.title(f"{camera} — speed distribution (n={len(a)})")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(vdir, "speed_distribution.png"), dpi=130)
        plt.close()

    # --- 2. vehicle-width validation (independent of calibration) ---
    video = _rep_video(camera)
    if video:
        widths = measure_vehicle_widths(video, model, H)
        wres = {}
        labels, preds, knowns, errs = [], [], [], []
        for cls, vals in widths.items():
            if len(vals) < 5:
                continue
            pred = float(np.median(vals))
            known = KNOWN_WIDTH_FT[cls]
            err = 100.0 * abs(pred - known) / known
            wres[cls] = {"n": len(vals), "predicted_width_ft": round(pred, 2),
                         "known_width_ft": known, "pct_error": round(err, 1)}
            labels.append(cls); preds.append(pred); knowns.append(known); errs.append(err)
        result["vehicle_width"] = wres

        if labels:
            x = np.arange(len(labels)); w = 0.38
            plt.figure(figsize=(6, 4))
            plt.bar(x - w/2, preds, w, label="predicted", color="#3a7")
            plt.bar(x + w/2, knowns, w, label="known", color="#888")
            for i, e in enumerate(errs):
                plt.text(x[i], max(preds[i], knowns[i]) + 0.2, f"{e:.0f}%",
                         ha="center", fontsize=9)
            plt.xticks(x, labels); plt.ylabel("width (ft)")
            plt.title(f"{camera} — vehicle-width validation (independent)")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(vdir, "vehicle_width_validation.png"), dpi=130)
            plt.close()

    with open(os.path.join(vdir, "validation.json"), "w") as f:
        json.dump(result, f, indent=2)

    # console summary
    sp = result.get("speed", {})
    print(f"{camera}: speed median={sp.get('median_mph')} mph IQR={sp.get('iqr_mph')}")
    for cls, w in result.get("vehicle_width", {}).items():
        print(f"    width {cls}: predicted {w['predicted_width_ft']} ft vs known "
              f"{w['known_width_ft']} ft -> {w['pct_error']}% error (n={w['n']})")


def main():
    cams = []
    if len(sys.argv) > 1:
        cams = [sys.argv[1]]
    else:
        for p in glob.glob(os.path.join(config.OUTPUT_DIR, "*",
                                        "*_velocity_matrix_cleaned.json")):
            cams.append(json.load(open(p))["camera"])
    if not cams:
        print("No cleaned matrices found. Run run_pipeline.py first.")
        return
    model = YOLO(config.YOLO_MODEL)
    for cam in cams:
        validate_camera(cam, model)


if __name__ == "__main__":
    main()
