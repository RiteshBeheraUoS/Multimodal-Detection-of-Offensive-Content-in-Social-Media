#!/usr/bin/env python3
"""
YOLO object detection over all images in a directory.
Outputs annotated images and a CSV of detections.
"""

import os
import argparse
import cv2
import torch
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch YOLO detection")
parser.add_argument("--input",  default=os.path.expanduser("~/img"),
                    help="Directory of input images (default: ~/img)")
# parser.add_argument("--output", default="output_detections",
#                     help="Directory for annotated images (default: output_detections)")
parser.add_argument("--model",  default="yolov8m-oiv7.pt",
                    help="YOLO model weights (default: yolov8m-oiv7.pt)")
parser.add_argument("--conf",   type=float, default=0.35,
                    help="Confidence threshold (default: 0.35)")
parser.add_argument("--csv",    default="detections.csv",
                    help="Output CSV filename (default: detections.csv)")
args = parser.parse_args()

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ── Setup ─────────────────────────────────────────────────────────────────────
# os.makedirs(args.output, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YOLO(args.model)
model.to(device)

image_files = sorted([
    f for f in os.listdir(args.input)
    if os.path.splitext(f)[1].lower() in VALID_EXT
])

if not image_files:
    print(f"No images found in {args.input}")
    exit(0)

print(f"Found {len(image_files)} image(s) in {args.input}")

object_records = []

# ── Main loop ─────────────────────────────────────────────────────────────────
loop = tqdm(enumerate(image_files, 1), total=len(image_files), desc="Processing images")
for i, fname in loop:
    filename   = os.path.splitext(fname)[0]
    image_path = os.path.join(args.input, fname)
    loop.set_postfix(file=filename)

    image = cv2.imread(image_path)
    if image is None:
        print(f"  [{i}/{len(image_files)}] WARNING: Could not read {fname}, skipping.")
        continue

    #print(f"  [{i}/{len(image_files)}] Processing: {fname}")

    img_h, img_w = image.shape[:2]
    results = model.predict(source=image, conf=args.conf, verbose=False)

    # ── Annotated figure ──────────────────────────────────────────────────────
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax   = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.axis("off")

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        all_boxes = r.boxes.xyxy
        all_cls   = r.boxes.cls.cpu().int()
        all_conf  = r.boxes.conf

        for obj_box, obj_cls, obj_conf in zip(all_boxes, all_cls, all_conf):
            ox1, oy1, ox2, oy2 = map(int, obj_box)
            object_label = model.names[int(obj_cls)]
            conf_val     = round(obj_conf.item(), 4)

            object_records.append({
                "filename":     filename,
                "object_label": object_label,
                "confidence":   conf_val,
                "bbox_x1":      ox1,
                "bbox_y1":      oy1,
                "bbox_x2":      ox2,
                "bbox_y2":      oy2,
                "bbox_width":   ox2 - ox1,
                "bbox_height":  oy2 - oy1,
            })

            # Bounding box
            rect = plt.Rectangle(
                (ox1, oy1), ox2 - ox1, oy2 - oy1,
                fill=False, edgecolor="red", linewidth=2
            )
            ax.add_patch(rect)

            # Label
            ax.text(
                ox1, oy1 - 5,
                f"{object_label} {conf_val:.2f}",
                color="white", fontsize=10,
                bbox=dict(facecolor="red", alpha=0.7)
            )

    # Save annotated image
    # out_path = os.path.join(args.output, f"{filename}_detected.jpg")
    plt.tight_layout(pad=0)
    # fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

# ── Save CSV ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(object_records)
df.to_csv(args.csv, index=False)

print(f"\nDone.")
# print(f"  Annotated images → {args.output}/")
print(f"  Detections CSV   → {args.csv}")
print(f"  Total detections : {len(object_records)}")