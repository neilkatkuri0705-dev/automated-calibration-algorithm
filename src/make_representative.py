"""
make_representative.py — grab a plain representative frame (with vehicles
visible) for each camera, for the manual Filter-1 LLM call. No YOLO.

    python make_representative.py            # all 6 cameras
    python make_representative.py <camera>   # one camera

Picks the frame with the most motion vs. a median background (a proxy for "most
vehicles present") — a plain frame, no overlays. Writes
outputs/<camera>/llm_inputs/representative.png
"""

import os
import sys
import glob

import cv2
import numpy as np

import config


def representative_frame(video_path, n_bg=40, n_scan=60):
    """Frame with the most foreground motion vs a median background."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        # unknown length: just take a frame ~1/3 in
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, f = cap.read()
        cap.release()
        return f if ret else None

    # median background from evenly sampled frames
    bg_idx = np.linspace(0, total - 1, min(n_bg, total)).astype(int)
    bg_frames = []
    for i in bg_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            bg_frames.append(f)
    if not bg_frames:
        cap.release()
        return None
    bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8)
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    # scan frames; pick the one that differs most from background (most vehicles)
    scan_idx = np.linspace(0, total - 1, min(n_scan, total)).astype(int)
    best_frame, best_score = None, -1
    for i in scan_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if not ret:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g, bg_gray)
        score = float((diff > 25).sum())     # # of foreground (moving) pixels
        if score > best_score:
            best_score, best_frame = score, f.copy()
    cap.release()
    return best_frame


def videos_in(d):
    v = []
    if os.path.isdir(d):
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            v.extend(glob.glob(os.path.join(d, ext)))
    return sorted(v)


def camera_video_map():
    m = {}
    single = videos_in(config.VIDEO_SINGLE_DIR)
    if single:
        m["single_camera"] = single[0]
    for v in videos_in(config.VIDEO_MULTI_DIR):
        m[os.path.splitext(os.path.basename(v))[0]] = v
    return m


def main():
    cams = camera_video_map()
    if len(sys.argv) > 1:
        cams = {sys.argv[1]: cams[sys.argv[1]]} if sys.argv[1] in cams else {}
        if not cams:
            print(f"camera not found. available: {', '.join(camera_video_map())}")
            return
    for cam, video in cams.items():
        out_dir = os.path.join(config.OUTPUT_DIR, cam, "llm_inputs")
        os.makedirs(out_dir, exist_ok=True)
        frame = representative_frame(video)
        if frame is None:
            print(f"{cam}: could not read {video}")
            continue
        out = os.path.join(out_dir, "representative.png")
        cv2.imwrite(out, frame)
        print(f"{cam} -> {out}")


if __name__ == "__main__":
    main()
