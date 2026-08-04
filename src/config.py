"""
config.py — central settings for the whole pipeline.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)  # Immersion_Paper/

VIDEO_SINGLE_DIR = os.path.join(PROJECT, "videos", "single_camera_10x1min")
VIDEO_MULTI_DIR = os.path.join(PROJECT, "videos", "multi_camera_5x10min")
CAMERA_IMAGES_DIR = os.path.join(PROJECT, "camera_images")
OUTPUT_DIR = os.path.join(PROJECT, "outputs")
FIGURES_DIR = os.path.join(PROJECT, "figures")

# --- Detection / tracking ---
YOLO_MODEL = "yolov8m.pt"
DETECT_IMGSZ = 1280
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DETECT_CONF = 0.2
FRAME_STRIDE = 2               # low: keeps tracking continuous (no fragmented IDs)
MAX_SECONDS_PER_VIDEO = 180    # first 3 min of each video (0 = all)

# --- Physical plausibility (kills ID-switch "teleport" strays) ---
# Judged by GROUND speed (perspective-correct): a big near-camera pixel step for
# an away-moving vehicle maps to a NORMAL ground speed and is kept. Only steps
# exceeding this ground speed are cut. Pixel cap is a fallback for uncalibrated
# cameras only, set high so real fast near vehicles survive.
MAX_STEP_SPEED_MPH = 120.0
MAX_STEP_JUMP_PX = 250.0

# --- Trajectory quality filtering (INTENSE clean: no strays allowed) ---
TRAJ_PCTL_LOW = 25
TRAJ_PCTL_HIGH = 75
MIN_TRAJ_POINTS = 3            # low so short far-carriageway tracks survive
MIN_TRAVEL_PX = 30            # must actually move this far (drops jitter/frozen)
STRAY_TOL_DEG = 35            # heading within this of a dominant flow direction
MAX_HEADING_STD_DEG = 35      # a real vehicle goes ~straight; drop wandering paths

# --- Cell grid (Stage 4) ---
CELL_SIZE_PX = 30

# --- Known real-world vehicle dimensions (for validation) ---
VEHICLE_HEIGHTS_M = {"car": 1.5, "truck": 4.0, "bus": 3.2}
VEHICLE_WIDTHS_M = {"car": 1.8, "truck": 2.6, "bus": 2.55}
