"""
run_pipeline.py — THE pipeline. Run with NO arguments:

    python run_pipeline.py

Per camera: detect + track every vehicle (accurate low-stride tracking, capped
at config.MAX_SECONDS_PER_VIDEO), SPLIT trajectories ONLY at physically-
impossible steps (ID-switch teleports, judged by ground speed so away-moving
vehicles aren't fragmented), INTENSELY clean them, and export the velocity
matrix as a flat per-observation CSV.

CAMERAS (6): single_camera (10 pooled videos) + 5 DMTV cameras (separate).

VELOCITY MATRIX (matrix_<camera>.csv) — one row per observation of the CLEANED
vehicles:  id, x_px, y_px, timestamp_s, speed_mph, direction_deg

Cameras with a saved matrix are SKIPPED (delete matrix_<camera>.csv to re-run).
Calibration: calibrations/<camera>.json (make_picker.py).

Other per-camera outputs (outputs/<camera>/):
  <camera>_trajectories.png              cleaned trajectory overlay (Filter-2 ref)
  <camera>_velocity_matrix_cleaned.json  cleaned trajectory detail (internal)
  llm_inputs/representative.png          frame for the Filter-1 LLM call
"""

import os
import csv
import glob
import json
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

import config
import calibration_from_points as calib_pts
import clean_matrix

VEHICLE = config.VEHICLE_CLASSES
CONF = config.DETECT_CONF
IMGSZ = config.DETECT_IMGSZ
STRIDE = max(1, config.FRAME_STRIDE)
MAX_SEC = config.MAX_SECONDS_PER_VIDEO
MAX_STEP_MPH = config.MAX_STEP_SPEED_MPH
MAX_JUMP_PX = config.MAX_STEP_JUMP_PX
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER = os.path.join(SRC_DIR, "bytetrack_persist.yaml")
CALIB_DIR = os.path.join(config.PROJECT, "calibrations")

DOWN_COLOR = (0, 0, 255)
UP_COLOR = (255, 0, 0)


# ---------------------------------------------------------------- calibration
def load_camera_calibration(camera):
    path = os.path.join(CALIB_DIR, f"{camera}.json")
    if os.path.exists(path):
        return _build_H(path)
    if camera == "single_camera":
        legacy = os.path.join(SRC_DIR, "calibration.json")
        if os.path.exists(legacy):
            return _build_H(legacy)
    return None, None


def _build_H(path):
    with open(path) as f:
        cal = json.load(f)
    src = np.float32(cal["source_px"])
    W = float(cal["width_ft"]); L = float(cal["length_ft"])
    dst = np.float32([[0, 0], [W, 0], [W, L], [0, L]])
    H = cv2.getPerspectiveTransform(src, dst)
    meta = {"width_ft": W, "length_ft": L,
            "condition_number": float(np.linalg.cond(H)),
            "source_px": cal["source_px"], "source_file": os.path.basename(path)}
    return H, meta


# ------------------------------------------ ID-switch / teleport splitting
def split_on_jumps(pts, tt, H_calib):
    """Split a raw track ONLY at physically-impossible steps (ID-switch teleports).

    An away-moving vehicle makes large pixel steps when near the camera (close =
    big pixels) -- those are NOT teleports. So impossibility is judged by GROUND
    speed (perspective-correct): a big near-camera step maps to a normal ground
    speed and is kept. Only steps whose ground speed exceeds MAX_STEP_MPH are cut.
    Uncalibrated cameras fall back to a generous pixel cap.
    """
    segs = []
    cur_p, cur_t = [pts[0]], [tt[0]]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]; x1, y1 = pts[i]
        dt = tt[i] - tt[i - 1]
        impossible = False
        if H_calib is not None and dt > 0:
            g0 = calib_pts.to_ground_ft(H_calib, x0, y0)
            g1 = calib_pts.to_ground_ft(H_calib, x1, y1)
            if g0 and g1:
                mph = (np.hypot(g1[0]-g0[0], g1[1]-g0[1]) / dt) * calib_pts.FT_PER_S_TO_MPH
                impossible = mph > MAX_STEP_MPH        # perspective-correct
            else:
                impossible = True                      # mapped off-plane -> cut
        else:
            impossible = np.hypot(x1 - x0, y1 - y0) > MAX_JUMP_PX
        if impossible:
            segs.append((cur_p, cur_t))
            cur_p, cur_t = [pts[i]], [tt[i]]
        else:
            cur_p.append(pts[i]); cur_t.append(tt[i])
    segs.append((cur_p, cur_t))
    return [(p, t) for p, t in segs if len(p) >= 2]


# ---------------------------------------------------------------- tracking
def track_one_video(video_path, model, tid_offset=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    cannot open {video_path}")
        return {}, {}, {}, None, None, 0, 20.0, tid_offset
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    max_idx = int(MAX_SEC * fps) if MAX_SEC else total
    if total:
        max_idx = min(max_idx, total) if max_idx else total

    paths = defaultdict(list)
    times = defaultdict(list)
    classes = {}
    base_frame = None; rep_frame = None; rep_count = -1
    real_idx = 0; processed = 0; max_tid = tid_offset

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_idx and real_idx >= max_idx:
            break
        if real_idx % STRIDE != 0:
            real_idx += 1
            continue
        if processed % 10 == 0:
            print(f"\r      frame {real_idx}/{max_idx}  ({processed} processed, "
                  f"{len(paths)} vehicles)", end="", flush=True)
        processed += 1
        if base_frame is None:
            base_frame = frame.copy()

        r = model.track(frame, persist=True, classes=list(VEHICLE.keys()),
                        conf=CONF, imgsz=IMGSZ, tracker=TRACKER, verbose=False)[0]
        n_det = len(r.boxes.id) if (r.boxes is not None and r.boxes.id is not None) else 0
        if n_det > rep_count:
            rep_count = n_det; rep_frame = frame.copy()

        t = real_idx / fps
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xywh = r.boxes.xywh.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for tid, (cx, cy, w, h), c in zip(ids, xywh, clss):
                gid = tid_offset + int(tid)
                max_tid = max(max_tid, gid)
                paths[gid].append((float(cx), float(cy + h / 2)))
                times[gid].append(t)
                classes[gid] = VEHICLE.get(int(c), str(c))
        real_idx += 1

    cap.release()
    print(f"\r      done: {processed} frames, {len(paths)} raw tracks" + " " * 20)
    return dict(paths), dict(times), classes, base_frame, rep_frame, rep_count, fps, max_tid


# ------------------------------------------------ velocity matrix CSV (flat)
def write_velocity_csv(out_dir, camera, cleaned_slices, seg_lookup, H_calib):
    path = os.path.join(out_dir, f"matrix_{camera}.csv")
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_px", "y_px", "timestamp_s", "speed_mph", "direction_deg"])
        for sl in cleaned_slices:
            pts, tt = seg_lookup[sl["id"]]
            for i in range(len(pts)):
                x, y = pts[i]
                speed, ddeg = "", ""
                if i > 0:
                    x0, y0 = pts[i - 1]; dt = tt[i] - tt[i - 1]
                    ddeg = round(float(np.degrees(np.arctan2(y - y0, x - x0))) % 360.0, 1)
                    if H_calib is not None and dt > 0:
                        g0 = calib_pts.to_ground_ft(H_calib, x0, y0)
                        g1 = calib_pts.to_ground_ft(H_calib, x, y)
                        if g0 and g1:
                            mph = (np.hypot(g1[0]-g0[0], g1[1]-g0[1]) / dt) * calib_pts.FT_PER_S_TO_MPH
                            if mph <= MAX_STEP_MPH:
                                speed = round(mph, 1)
                w.writerow([sl["id"], round(x, 1), round(y, 1), round(tt[i], 3), speed, ddeg])
                n += 1
    return path, n


# ---------------------------------------------------------------- one camera
def process_camera(camera, video_list, model):
    out_dir = os.path.join(config.OUTPUT_DIR, camera)
    csv_path = os.path.join(out_dir, f"matrix_{camera}.csv")
    if os.path.exists(csv_path):
        print(f"\n=== {camera} === already done -> SKIP")
        H, _ = load_camera_calibration(camera)
        return camera, H is not None, None, None, 0, (0, 0)

    print(f"\n=== {camera} ===  ({len(video_list)} video(s)"
          + (" pooled)" if len(video_list) > 1 else ")")
          + f"  {config.YOLO_MODEL} imgsz={IMGSZ} stride={STRIDE} cap={MAX_SEC}s")
    os.makedirs(out_dir, exist_ok=True)
    H_calib, cal_meta = load_camera_calibration(camera)

    all_paths, all_times, all_classes = {}, {}, {}
    base_frame = None; best_rep, best_rep_count = None, -1
    cam_fps = 20.0; offset = 0
    for vi, v in enumerate(video_list, 1):
        print(f"    [{vi}/{len(video_list)}] {os.path.basename(v)}")
        paths, times, classes, bframe, rep, rc, fps, mtid = track_one_video(v, model, offset)
        cam_fps = fps
        if base_frame is None and bframe is not None:
            base_frame = bframe
        if rc > best_rep_count:
            best_rep_count, best_rep = rc, rep
        all_paths.update(paths); all_times.update(times); all_classes.update(classes)
        offset = mtid + 1

    if base_frame is None:
        print("    no frames; skipping")
        return camera, H_calib is not None, 0, None, cam_fps, (0, 0)

    llm_in = os.path.join(out_dir, "llm_inputs"); os.makedirs(llm_in, exist_ok=True)
    cv2.imwrite(os.path.join(llm_in, "representative.png"),
                best_rep if best_rep is not None else base_frame)

    # SPLIT each raw track at impossible jumps -> segments with new ids
    H_img, W_img = base_frame.shape[:2]
    slices = []
    seg_lookup = {}
    seg_id = 0
    n_splits = 0
    for tid, pts in all_paths.items():
        tt = all_times[tid]
        segs = split_on_jumps(pts, tt, H_calib)
        if len(segs) > 1:
            n_splits += len(segs) - 1
        for spts, stt in segs:
            dx = spts[-1][0]-spts[0][0]; dy = spts[-1][1]-spts[0][1]
            heading = round(float(np.degrees(np.arctan2(dy, dx))) % 360.0, 1)
            spd = calib_pts.avg_speed_mph(spts, stt, H_calib) if H_calib is not None else None
            slices.append({"id": seg_id, "class": all_classes.get(tid),
                           "heading_deg": heading,
                           "avg_speed_mph": round(spd, 1) if spd is not None else None,
                           "positions": [[round(x, 1), round(y, 1)] for x, y in spts],
                           "n_points": len(spts)})
            seg_lookup[seg_id] = (spts, stt)
            seg_id += 1

    matrix = {"camera": camera, "H": H_img, "W": W_img, "fps": round(cam_fps, 3),
              "n": len(slices), "cleaned": False,
              "calibrated": H_calib is not None, "calibration": cal_meta, "slices": slices}

    cleaned = clean_matrix.clean(matrix)
    with open(os.path.join(out_dir, f"{camera}_velocity_matrix_cleaned.json"), "w") as f:
        json.dump(cleaned, f, separators=(",", ":"))

    _, n_rows = write_velocity_csv(out_dir, camera, cleaned["slices"], seg_lookup, H_calib)

    vis = base_frame.copy(); drawn = 0
    for sl in cleaned["slices"]:
        pts = sl["positions"]
        if len(pts) < 2:
            continue
        color = DOWN_COLOR if np.sin(np.radians(sl["heading_deg"])) >= 0 else UP_COLOR
        arr = np.array(pts, dtype=np.int32)
        for j in range(len(arr)-1):
            cv2.line(vis, tuple(arr[j]), tuple(arr[j+1]), color, 1, cv2.LINE_AA)
        cv2.arrowedLine(vis, tuple(arr[-2]), tuple(arr[-1]), color, 1, cv2.LINE_AA, tipLength=0.4)
        drawn += 1
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(vis, f"{camera}: {drawn} cleaned trajectories", (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, f"{camera}_trajectories.png"), vis)

    with open(os.path.join(out_dir, f"{camera}_calibration_used.json"), "w") as f:
        json.dump(cal_meta, f, indent=2)

    c = cleaned["cleaning"]
    speeds = [s["avg_speed_mph"] for s in cleaned["slices"] if s["avg_speed_mph"] is not None]
    print(f"    frame {W_img}x{H_img} fps={cam_fps:.0f}  (split {n_splits} teleport jumps)")
    print(f"    raw {len(slices)} segs -> cleaned {cleaned['n']} vehicles "
          f"(short {c['dropped_short']}, frozen {c['dropped_frozen']}, "
          f"wiggly {c['dropped_wiggly']}, stray {c['dropped_stray']}, speed {c['dropped_speed_band']})")
    print(f"    velocity matrix: matrix_{camera}.csv ({n_rows} rows)")
    if H_calib is not None and speeds:
        a = np.array(speeds)
        print(f"    speeds: median={np.median(a):.0f} mph, range {a.min():.0f}-{a.max():.0f}")
    elif H_calib is None:
        print(f"    NO CALIBRATION -> speeds null. python make_picker.py {camera}")
    return camera, H_calib is not None, cleaned["n"], (np.median(speeds) if speeds else None), cam_fps, (W_img, H_img)


# ---------------------------------------------------------------- driver
def videos_in(d):
    v = []
    if os.path.isdir(d):
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            v.extend(glob.glob(os.path.join(d, ext)))
    return sorted(v)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(CALIB_DIR, exist_ok=True)
    model = YOLO(config.YOLO_MODEL)

    cameras = []
    single = videos_in(config.VIDEO_SINGLE_DIR)
    if single:
        cameras.append(("single_camera", single))
    for v in videos_in(config.VIDEO_MULTI_DIR):
        cameras.append((os.path.splitext(os.path.basename(v))[0], [v]))
    if not cameras:
        print("No videos found."); return

    print(f"Pipeline: {len(cameras)} cameras, model={config.YOLO_MODEL}, "
          f"imgsz={IMGSZ}, stride={STRIDE}, cap={MAX_SEC}s/video.")
    results = [process_camera(cam, vids, model) for cam, vids in cameras]

    print("\n================ SUMMARY ================")
    need = []
    for cam, calib, n, med, fps, size in results:
        if n is None:
            print(f"  {cam:26s} SKIPPED"); continue
        sp = f"median {med:.0f} mph" if med is not None else "speeds NULL"
        print(f"  {cam:26s} {size[0]}x{size[1]} fps={fps:.0f} clean_vehicles={n:4d}  "
              f"{'CALIBRATED' if calib else 'no calib'}  {sp}")
        if not calib:
            need.append(cam)
    if need:
        print("\nNeed calibration:")
        for cam in need:
            print(f"  python make_picker.py {cam}")
    print("Done.")


if __name__ == "__main__":
    main()
