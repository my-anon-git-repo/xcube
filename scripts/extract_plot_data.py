#!/usr/bin/env python3
"""
extract_plot_data.py

Reverse-engineer (x, y) data from a matplotlib-style plot image with two colored lines
(blue + orange), like the provided MIMIC-IV figure. This script uses simple color
segmentation in HSV space and a linear calibration between pixel coordinates and axis
values (log-scale on X supported).

Dependencies:
  - opencv-python (cv2)
  - numpy
  - pandas
  - matplotlib (optional; only used when --debug is set)

Usage example (works out-of-the-box for the provided image):
  python extract_plot_data.py \
      --image D6BC0002-7E5D-462E-B1B6-19BFA3606A90.png \
      --out reverse_engineered_macroF1.csv \
      --crop 80:380,60:560 \
      --x-pixels 40 490 --x-log10 1 3.477 \
      --y-pixels 160 0 --y-values -0.2 0.8

Notes:
  - X mapping: pixel -> log10(x) is linear, then take 10** to return to the original scale.
  - Y mapping: pixel -> value is linear.
  - You can tweak HSV color ranges if your plot uses different line colors.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import cv2

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # debug plots disabled if matplotlib is unavailable


def parse_crop(s: str):
    """
    Parse a crop string "y1:y2,x1:x2" into integer bounds.
    """
    try:
        ypart, xpart = s.split(",")
        y1, y2 = [int(v) for v in ypart.split(":")]
        x1, x2 = [int(v) for v in xpart.split(":")]
        return y1, y2, x1, x2
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid --crop format '{s}'. Expected 'y1:y2,x1:x2'")


def build_mask(hsv, lower, upper):
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Clean small speckles
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def pixel_to_data(x_pix, y_pix,
                  x_pix_min, x_pix_max, x_log_min, x_log_max,
                  y_pix_min, y_pix_max, y_val_min, y_val_max):
    # Map x from pixel to log10 space, then exponentiate
    x_log = x_log_min + (x_pix - x_pix_min) / (x_pix_max - x_pix_min) * (x_log_max - x_log_min)
    x_val = 10 ** x_log
    # Map y from pixel to value (note y axis increases upward, but image y increases downward)
    y_val = y_val_min + (y_pix - y_pix_min) / (y_pix_max - y_pix_min) * (y_val_max - y_val_min)
    return x_val, y_val


def main():
    p = argparse.ArgumentParser(description="Reverse-engineer (x, y) data from a plot image.")
    p.add_argument("--image", required=True, help="Path to the plot image (PNG, JPG, etc.)")
    p.add_argument("--out", default="reverse_engineered_macroF1.csv", help="Output CSV path")

    # Crop bounds in source image (y1:y2, x1:x2)
    p.add_argument("--crop", type=parse_crop, default="80:380,60:560",
                   help="Crop region as 'y1:y2,x1:x2' to isolate the plotting area")

    # X-axis calibration: pixel range and matching log10(x) range
    p.add_argument("--x-pixels", type=int, nargs=2, default=[40, 490],
                   help="Pixel range within the CROPPED image that corresponds to the x-axis span")
    p.add_argument("--x-log10", type=float, nargs=2, default=[1.0, 3.477],  # log10(10) to log10(3000)
                   help="Axis range in log10 space (e.g., log10(min), log10(max))")

    # Y-axis calibration: pixel range and value range
    p.add_argument("--y-pixels", type=int, nargs=2, default=[160, 0],
                   help="Pixel range within the CROPPED image that corresponds to the y-axis span (min->max)")
    p.add_argument("--y-values", type=float, nargs=2, default=[-0.2, 0.8],
                   help="Y-axis values corresponding to the pixel range (min, max)")

    # HSV thresholds for blue and orange lines
    p.add_argument("--blue-hsv", type=int, nargs=6, default=[90, 50, 50, 130, 255, 255],
                   help="Blue HSV lower+upper bounds: Hmin Smin Vmin Hmax Smax Vmax")
    p.add_argument("--orange-hsv", type=int, nargs=6, default=[10, 100, 100, 25, 255, 255],
                   help="Orange HSV lower+upper bounds: Hmin Smin Vmin Hmax Smax Vmax")

    p.add_argument("--debug", action="store_true", help="Save debug images (masks and scatter) next to output")

    args = p.parse_args()

    # Load and crop
    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    y1, y2, x1, x2 = args.crop
    crop = img_rgb[y1:y2, x1:x2].copy()
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Build masks
    bl = args.blue_hsv[:3]
    bu = args.blue_hsv[3:]
    ol = args.orange_hsv[:3]
    ou = args.orange_hsv[3:]

    mask_blue = build_mask(crop_hsv, bl, bu)
    mask_orange = build_mask(crop_hsv, ol, ou)

    # Extract pixel coordinates
    blue_pts = np.column_stack(np.where(mask_blue > 0))  # (row, col)
    orange_pts = np.column_stack(np.where(mask_orange > 0))

    # Convert to (x, y) with origin at top-left of the cropped image
    blue_xy = np.fliplr(blue_pts)     # (col, row)
    orange_xy = np.fliplr(orange_pts)

    # Calibration
    x_pix_min, x_pix_max = args.x_pixels
    x_log_min, x_log_max = args.x_log10
    y_pix_min, y_pix_max = args.y_pixels
    y_val_min, y_val_max = args.y_values

    # Map pixels to data
    def map_points(points_xy):
        xs, ys = [], []
        for x_pix, y_pix in points_xy:
            xv, yv = pixel_to_data(
                x_pix, y_pix,
                x_pix_min, x_pix_max, x_log_min, x_log_max,
                y_pix_min, y_pix_max, y_val_min, y_val_max
            )
            xs.append(xv); ys.append(yv)
        return np.array(xs), np.array(ys)

    xb, yb = map_points(blue_xy)
    xo, yo = map_points(orange_xy)

    # Pack into DataFrame
    df_blue = pd.DataFrame({"code_frequency": xb, "macro_F1": yb, "dataset": "MIMIC-IV ICD-9"})
    df_orange = pd.DataFrame({"code_frequency": xo, "macro_F1": yo, "dataset": "MIMIC-IV ICD-10"})
    df = pd.concat([df_blue, df_orange], ignore_index=True)

    # Save CSV
    out_path = args.out
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to: {out_path}")

    # Optional debug visuals
    if args.debug and plt is not None:
        base = os.path.splitext(out_path)[0]
        import matplotlib.pyplot as plt

        # Save masks
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.imshow(mask_blue, cmap="gray"); plt.title("Blue Line Mask"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(mask_orange, cmap="gray"); plt.title("Orange Line Mask"); plt.axis("off")
        plt.tight_layout(); plt.savefig(base + "_masks.png", dpi=200); plt.close()

        # Scatter of recovered points
        plt.figure(figsize=(6, 4))
        plt.scatter(xb, yb, s=1, label="ICD-9 (blue)")
        plt.scatter(xo, yo, s=1, label="ICD-10 (orange)")
        plt.xscale("log"); plt.xlabel("Code frequency"); plt.ylabel("Macro F1"); plt.legend()
        plt.tight_layout(); plt.savefig(base + "_scatter.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
