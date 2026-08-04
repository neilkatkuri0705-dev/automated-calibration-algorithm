"""
make_figures.py — generate results figures for the paper from real pipeline
outputs. NO fabricated data.

Figures written to figures/:
  1. <camera>_overlay.png        cleaned trajectory (filter) overlay
  2. <camera>_speed_hist.png     speed distribution (per-vehicle median mph)
  3. all_cameras_speed.png       box plot comparing all cameras' speeds

    python make_figures.py
"""

import os
import glob
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FIG_DIR = config.FIGURES_DIR
os.makedirs(FIG_DIR, exist_ok=True)


def load_cameras():
    out = {}
    for p in glob.glob(os.path.join(config.OUTPUT_DIR, "*",
                                    "*_velocity_matrix_cleaned.json")):
        m = json.load(open(p))
        out[m["camera"]] = m
    return out


def fig_overlays(cameras):
    for cam in cameras:
        src = os.path.join(config.OUTPUT_DIR, cam, f"{cam}_trajectories.png")
        if os.path.exists(src):
            dst = os.path.join(FIG_DIR, f"{cam}_overlay.png")
            shutil.copy(src, dst)
            print(f"  overlay: {os.path.basename(dst)}")


def fig_speed_hist(cameras):
    for cam, m in cameras.items():
        speeds = [s["avg_speed_mph"] for s in m["slices"]
                  if s.get("avg_speed_mph") is not None]
        if len(speeds) < 5:
            print(f"  {cam}: <5 speeds, skipping hist")
            continue
        a = np.array(speeds)
        plt.figure(figsize=(6, 4))
        plt.hist(a, bins=18, color="#3a7", edgecolor="white")
        plt.axvline(np.median(a), color="#222", ls="--",
                    label=f"median {np.median(a):.0f} mph")
        plt.xlabel("vehicle speed (mph)")
        plt.ylabel("number of vehicles")
        plt.title(f"{cam} — speed distribution (n={len(a)})")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"{cam}_speed_hist.png")
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"  hist: {os.path.basename(out)}  (median {np.median(a):.0f}, "
              f"IQR {np.percentile(a,25):.0f}-{np.percentile(a,75):.0f})")


def fig_all_cameras(cameras):
    labels, data = [], []
    for cam, m in cameras.items():
        speeds = [s["avg_speed_mph"] for s in m["slices"]
                  if s.get("avg_speed_mph") is not None]
        if len(speeds) >= 5:
            short = cam.replace("_20260708", "").replace("_raw", "")
            labels.append(short)
            data.append(speeds)
    if not data:
        return
    plt.figure(figsize=(max(7, 1.4 * len(data)), 4.5))
    try:
        bp = plt.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    except TypeError:
        bp = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor="#9cd")
    plt.ylabel("vehicle speed (mph)")
    plt.title("Speed distribution by camera")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "all_cameras_speed.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  combined: {os.path.basename(out)}  ({len(data)} cameras)")


def main():
    cameras = load_cameras()
    if not cameras:
        print("No cleaned matrices found. Run run_pipeline.py first.")
        return
    print(f"{len(cameras)} cameras with data:")
    print("Overlays:")
    fig_overlays(cameras)
    print("Speed histograms:")
    fig_speed_hist(cameras)
    print("Combined:")
    fig_all_cameras(cameras)
    print(f"\nAll figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
