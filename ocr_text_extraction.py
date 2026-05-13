#!/usr/bin/env python3
"""
OCR pipeline for Facebook memes dataset.
Designed to run on the Iridis HPC cluster (University of Southampton).

Usage:
    python ocr_memes.py --images_dir ./data/img --output_file results.json
    python ocr_memes.py --images_dir ./data/img --output_file results.csv --limit 100 --psm 6 --oem 3
"""

import argparse
import os
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
except ModuleNotFoundError:
    print("pytesseract not found. Install with: pip install pytesseract")
    sys.exit(1)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    # Graceful fallback if tqdm unavailable on the cluster
    def tqdm(iterable, **kwargs):
        return iterable

# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess(image_path: str, min_dim: int = 800, invert_if_needed: bool = True) -> np.ndarray:
    """
    Preprocess an image for Tesseract OCR.

    Steps:
        1. Read image from disk.
        2. Convert to grayscale.
        3. Upscale small images (Tesseract accuracy degrades on tiny text).
        4. Median blur for light denoising.
        5. Histogram equalisation for contrast normalisation.
        6. Otsu binarisation; invert if background is dark.
        7. Morphological opening to remove specks.

    Args:
        image_path:       Path to the source image file.
        min_dim:          Minimum pixel dimension for upscaling.
        invert_if_needed: Automatically invert light-on-dark images.

    Returns:
        Binary (uint8) numpy array ready for pytesseract.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if max(h, w) < min_dim:
        scale = float(min_dim) / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    gray = cv2.medianBlur(gray, 3)
    gray = cv2.equalizeHist(gray)

    _, th_binary = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert_if_needed and np.mean(th_binary == 255) < 0.5:
        th_binary = cv2.bitwise_not(th_binary)

    kernel = np.ones((2, 2), np.uint8)
    th_binary = cv2.morphologyEx(th_binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return th_binary


# ── OCR ───────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def run_ocr(images_dir: str, limit: int, psm: int, oem: int,
            min_dim: int, invert: bool) -> list[dict]:
    """
    Run Tesseract OCR over images in `images_dir`.

    Args:
        images_dir: Directory containing image files.
        limit:      Maximum number of images to process (0 = all).
        psm:        Tesseract page segmentation mode.
        oem:        Tesseract OCR engine mode.
        min_dim:    Minimum dimension for upscaling during preprocessing.
        invert:     Whether to auto-invert dark-background images.

    Returns:
        List of dicts with keys ``fid`` and ``text``.
    """
    config = f"--psm {psm} --oem {oem}"
    all_files = [
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if limit > 0:
        all_files = all_files[:limit]

    print(f"Processing {len(all_files)} image(s) from {images_dir}")

    results = []
    errors = 0
    loop = tqdm(all_files, desc="OCR images")
    loop.set_description("Extracting text from OCR images")
    for filename in loop:
        loop.set_postfix(file= filename)
        fid = Path(filename).stem
        image_path = os.path.join(images_dir, filename)
        try:
            processed = preprocess(image_path, min_dim=min_dim,
                                   invert_if_needed=invert)
            text = pytesseract.image_to_string(processed, config=config).strip()
        except Exception as exc:
            print(f"Skipping {filename} — {exc}")
            text = ""
            errors += 1

        results.append({"fid": fid, "text": text})

    print(f"Finished. Errors: {errors} / {len(all_files)}")
    return results


# ── Output writers ────────────────────────────────────────────────────────────
def save_csv(results: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fid", "text"])
        writer.writeheader()
        writer.writerows(results)
    print("Saved CSV  → %s", path)


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ocr_memes",
        description="Batch OCR for the Facebook Memes dataset (Iridis-ready).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ──
    io = parser.add_argument_group("I/O")
    io.add_argument(
        "--images_dir", "-i",
        required=True,
        help="Directory containing meme images.",
    )
    io.add_argument(
        "--output_file", "-o",
        required=True,
        help="Output CSV file path (e.g. results.csv).",
    )

    # ── Processing ──
    proc = parser.add_argument_group("Processing")
    proc.add_argument(
        "--limit", "-n",
        type=int, default=0,
        help="Max images to process (0 = all).",
    )
    proc.add_argument(
        "--min_dim",
        type=int, default=800,
        help="Upscale images whose largest dimension is below this value.",
    )
    proc.add_argument(
        "--no_invert",
        action="store_true",
        help="Disable automatic inversion of dark-background images.",
    )

    # ── Tesseract ──
    tess = parser.add_argument_group("Tesseract")
    tess.add_argument(
        "--psm",
        type=int, default=6, choices=range(0, 14),
        metavar="{0-13}",
        help="Page segmentation mode (--psm). 6 = uniform block of text.",
    )
    tess.add_argument(
        "--oem",
        type=int, default=3, choices=[0, 1, 2, 3],
        metavar="{0-3}",
        help="OCR engine mode (--oem). 3 = default (LSTM + legacy).",
    )
    tess.add_argument(
        "--tesseract_cmd",
        default=None,
        help="Explicit path to the tesseract binary (e.g. on Iridis with a module load).",
    )

    return parser.parse_args(argv)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Override tesseract binary if specified
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
        print("Tesseract binary set to: %s", args.tesseract_cmd)

    # Validate inputs
    if not os.path.isdir(args.images_dir):
        print("images_dir does not exist: %s", args.images_dir)
        sys.exit(1)

    # Run OCR
    results = run_ocr(
        images_dir=args.images_dir,
        limit=args.limit,
        psm=args.psm,
        oem=args.oem,
        min_dim=args.min_dim,
        invert=not args.no_invert,
    )

    # Write output as CSV
    os.makedirs(Path(args.output_file).parent, exist_ok=True)
    save_csv(results, args.output_file)


if __name__ == "__main__":
    main()