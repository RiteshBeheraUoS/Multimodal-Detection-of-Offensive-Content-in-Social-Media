"""
remove_annotation.py
--------------------
Batch annotation-removal script replicating the logic from Remove_annotation.ipynb.

For every image in:
    Facebook_memes_dataset/data/img/

Three cleaned versions are produced and saved to:
    clean_img1/  – Method 1: Semi-transparent red overlay on detected text regions
    clean_img2/  – Method 2: OpenCV TELEA inpainting over detected text regions
    clean_img3/  – Method 3: DeepFillV2 (PyTorch) deep-inpainting over detected text regions

Output file names match the input file names exactly.

Requirements
------------
    pip install opencv-python-headless easyocr torch torchvision pillow numpy
    DeepFillV2 model:
        - Repo: https://github.com/nipponjo/deepfillv2-pytorch
        - Clone into ./deepfillv2-pytorch/
        - Pretrained weights at: deepfillv2-pytorch/pretrained/states_tf_places2.pth
"""

# Standard-library imports (always available)
import os
import sys
import argparse
from pathlib import Path

# Third-party imports — each wrapped so a missing package gives a clear message
try:
    import cv2
except ImportError:
    sys.exit(
        "[ERROR] 'opencv-python-headless' is not installed.\n"
        "        Install it with:  pip install opencv-python-headless"
    )

try:
    import numpy as np
except ImportError:
    sys.exit(
        "[ERROR] 'numpy' is not installed.\n"
        "        Install it with:  pip install numpy"
    )

try:
    from PIL import Image
except ImportError:
    sys.exit(
        "[ERROR] 'Pillow' is not installed.\n"
        "        Install it with:  pip install pillow"
    )

try:
    import torch
except ImportError:
    sys.exit(
        "[ERROR] 'torch' is not installed.\n"
        "        Install it with:  pip install torch  (see https://pytorch.org for CUDA builds)"
    )

try:
    import torchvision.transforms as T
except ImportError:
    sys.exit(
        "[ERROR] 'torchvision' is not installed.\n"
        "        Install it with:  pip install torchvision"
    )

try:
    import easyocr
except ImportError:
    sys.exit(
        "[ERROR] 'easyocr' is not installed.\n"
        "        Install it with:  pip install easyocr"
    )


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Helper: load DeepFillV2 generator
# ---------------------------------------------------------------------------
def load_deepfill_generator(device: str, deepfill_model_dir: Path):
    """Load the DeepFillV2 Generator from the local repo."""
    sys.path.insert(0, str(deepfill_model_dir / "model"))
    from networks_tf import Generator  # noqa: E402 (local import)

    weights = deepfill_model_dir / "pretrained" / "states_tf_places2.pth"
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    state_dict = torch.load(str(weights), map_location=device)
    generator.load_state_dict(state_dict["G"], strict=True)
    generator.eval()
    return generator


# ---------------------------------------------------------------------------
# Method 1 – semi-transparent red overlay
# ---------------------------------------------------------------------------
def method1_red_overlay(img_bgr: np.ndarray, ocr_results) -> np.ndarray:
    """
    Draw a near-transparent red filled polygon over every detected text bbox,
    plus a red outline (alpha=0.01 as in the notebook).
    Returns a BGR image.
    """
    img = img_bgr.copy()
    for bbox, _text, _conf in ocr_results:
        pts = np.array(bbox, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 0, 255))  # BGR red
        alpha = 0.01
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    return img


# ---------------------------------------------------------------------------
# Method 2 – OpenCV TELEA inpainting
# ---------------------------------------------------------------------------
def method2_cv_inpaint(img_bgr: np.ndarray, ocr_results) -> np.ndarray:
    """
    Build a binary mask from EasyOCR bboxes and inpaint with cv2.INPAINT_TELEA.
    Returns a BGR image.
    """
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for bbox, _text, _conf in ocr_results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    inpaint_radius = 3
    inpainted = cv2.inpaint(img_bgr, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
    return inpainted


# ---------------------------------------------------------------------------
# Method 3 – DeepFillV2 inpainting
# ---------------------------------------------------------------------------
def method3_deepfill(img_bgr: np.ndarray, ocr_results, generator, device: str) -> np.ndarray:
    """
    Build a dilated + blurred mask from EasyOCR bboxes and inpaint with
    the DeepFillV2 generator.  Returns an RGB uint8 numpy array.
    """
    # Build base mask
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for bbox, _text, _conf in ocr_results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Dilate + soft-blur mask
    kernel       = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    mask_soft    = cv2.GaussianBlur(mask_dilated, (5, 5), 0)

    # Convert BGR → RGB PIL
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)
    mask_pil  = Image.fromarray(mask_soft)

    # To tensors
    image_t = T.ToTensor()(image_pil)   # [3, H, W]
    mask_t  = T.ToTensor()(mask_pil)    # [1, H, W]

    # Crop to multiple of 8
    grid  = 8
    _, h, w = image_t.shape
    h_, w_  = h // grid * grid, w // grid * grid
    image_t = image_t[:3, :h_, :w_].unsqueeze(0)
    mask_t  = mask_t[0:1, :h_, :w_].unsqueeze(0)

    # Normalise
    image_t = (image_t * 2 - 1.0).to(device)
    mask_t  = (mask_t > 0.5).to(dtype=torch.float32, device=device)

    # 5-channel input
    image_masked = image_t * (1.0 - mask_t)
    ones_x       = torch.ones_like(image_masked)[:, 0:1]
    x            = torch.cat([image_masked, ones_x, ones_x * mask_t], dim=1)

    # Inference
    with torch.inference_mode():
        _, x_stage2 = generator(x, mask_t)

    # Composite & convert back
    image_inpainted = image_t * (1.0 - mask_t) + x_stage2 * mask_t
    result_img = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5)
    result_img = result_img.cpu().clamp(0, 255).to(torch.uint8).numpy()
    return result_img  # RGB


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def main(
    input_dir: Path,
    out_dir1: Path,
    out_dir2: Path,
    out_dir3: Path,
    deepfill_model_dir: Path,
    easyocr_model_dir: Path,
    skip_deepfill: bool = False,
):
    # Validate input directory
    if not input_dir.exists():
        sys.exit(f"[ERROR] Input directory not found: {input_dir}")

    # Collect image files
    image_files = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ])
    if not image_files:
        sys.exit(f"[ERROR] No supported image files found in {input_dir}")

    print(f"Found {len(image_files)} image(s) in {input_dir}")

    # Create output directories
    for d in (out_dir1, out_dir2, out_dir3):
        d.mkdir(parents=True, exist_ok=True)

    # Initialise EasyOCR reader (once for all images)
    # model_storage_directory: use pre-downloaded models (no network needed on compute nodes)
    # download_enabled=False: fail fast if models are missing rather than hanging on a network error
    print(f"Initialising EasyOCR reader (model dir: {easyocr_model_dir}) …")
    reader = easyocr.Reader(
        ["en"],
        model_storage_directory=str(easyocr_model_dir),
        download_enabled=False,
    )

    # Initialise DeepFillV2 (optional)
    generator = None
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    if not skip_deepfill:
        deepfill_weights = deepfill_model_dir / "pretrained" / "states_tf_places2.pth"
        if not deepfill_weights.exists():
            print(
                f"[WARNING] DeepFillV2 weights not found at {deepfill_weights}.\n"
                "          Skipping Method 3 (clean_img3). "
                "Pass --skip-deepfill to suppress this warning."
            )
            skip_deepfill = True
        else:
            print(f"Loading DeepFillV2 generator on {device} …")
            try:
                generator = load_deepfill_generator(device, deepfill_model_dir)
                print("DeepFillV2 loaded successfully.")
            except Exception as exc:
                print(f"[WARNING] Could not load DeepFillV2: {exc}\n"
                      "          Skipping Method 3 (clean_img3).")
                skip_deepfill = True

    # Process each image
    for idx, img_path in enumerate(image_files, start=1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")

        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] Could not read image: {img_path}")
            continue

        # Run OCR
        ocr_results = reader.readtext(str(img_path))
        # print(f"  OCR detected {len(ocr_results)} text region(s).")

        # ── Method 1: Red overlay ──────────────────────────────────────────
        try:
            m1_bgr  = method1_red_overlay(img_bgr, ocr_results)
            m1_rgb  = cv2.cvtColor(m1_bgr, cv2.COLOR_BGR2RGB)
            out1    = out_dir1 / img_path.name
            Image.fromarray(m1_rgb).save(str(out1))
            # print(f"  [OK] Method 1 → {out1}")
        except Exception as exc:
            print(f"  [FAIL] Method 1: {exc}")

        # ── Method 2: OpenCV inpainting ────────────────────────────────────
        try:
            m2_bgr = method2_cv_inpaint(img_bgr, ocr_results)
            m2_rgb = cv2.cvtColor(m2_bgr, cv2.COLOR_BGR2RGB)
            out2   = out_dir2 / img_path.name
            Image.fromarray(m2_rgb).save(str(out2))
            # print(f"  [OK] Method 2 → {out2}")
        except Exception as exc:
            print(f"  [FAIL] Method 2: {exc}")

        # ── Method 3: DeepFillV2 ──────────────────────────────────────────
        if not skip_deepfill and generator is not None:
            try:
                m3_rgb = method3_deepfill(img_bgr, ocr_results, generator, device)
                out3   = out_dir3 / img_path.name
                Image.fromarray(m3_rgb).save(str(out3))
                # print(f"  [OK] Method 3 → {out3}")
            except Exception as exc:
                print(f"  [FAIL] Method 3: {exc}")
        elif not skip_deepfill:
            print("  [SKIP] Method 3: generator not loaded.")

    print("\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove text annotations from meme images using three methods."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Facebook_memes_dataset/data/img"),
        help="Path to the directory containing input images. "
             "(default: Facebook_memes_dataset/data/img)",
    )
    parser.add_argument(
        "--out-dir1",
        type=Path,
        default=Path("clean_img1"),
        help="Output directory for Method 1 (red overlay). (default: clean_img1)",
    )
    parser.add_argument(
        "--out-dir2",
        type=Path,
        default=Path("clean_img2"),
        help="Output directory for Method 2 (OpenCV inpainting). (default: clean_img2)",
    )
    parser.add_argument(
        "--out-dir3",
        type=Path,
        default=Path("clean_img3"),
        help="Output directory for Method 3 (DeepFillV2). (default: clean_img3)",
    )
    parser.add_argument(
        "--easyocr-model-dir",
        type=Path,
        default=Path.home() / ".EasyOCR" / "model",
        help="Directory containing pre-downloaded EasyOCR model files "
             "(craft_mlt_25k.pth, english_g2.pth). "
             "Download on a login node first with: "
             "python -c \"import easyocr; easyocr.Reader(['en'])\". "
             "(default: ~/.EasyOCR/model)",
    )
    parser.add_argument(
        "--deepfill-model-dir",
        type=Path,
        default=Path("deepfillv2-pytorch"),
        help="Path to the cloned deepfillv2-pytorch repository. "
             "(default: deepfillv2-pytorch)",
    )
    parser.add_argument(
        "--skip-deepfill",
        action="store_true",
        help="Skip Method 3 (DeepFillV2) even if weights are present.",
    )
    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        out_dir1=args.out_dir1,
        out_dir2=args.out_dir2,
        out_dir3=args.out_dir3,
        deepfill_model_dir=args.deepfill_model_dir,
        easyocr_model_dir=args.easyocr_model_dir,
        skip_deepfill=args.skip_deepfill,
    )