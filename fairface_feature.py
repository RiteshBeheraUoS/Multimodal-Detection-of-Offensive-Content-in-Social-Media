import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import argparse
import pandas as pd
from tqdm import tqdm

# =========================
# Argument Parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Face detection and attribute classification pipeline using YOLOv8 + FairFace."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to input folder containing images (.png, .jpg, .jpeg)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="final_output",
        help="Path to output folder for processed images (default: final_output)"
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 model weights (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--fairface_model",
        type=str,
        default="res34_fair_align_multi_7_20190809.pt",
        help="Path to FairFace ResNet34 model weights"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=15,
        help="Maximum number of images to process (default: 15)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)"
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="results.csv",
        help="Filename for the output CSV (default: results.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting pipeline...")
    print(f"  Input folder  : {args.input_folder}")
    print(f"  Output folder : {args.output_folder}")
    print(f"  YOLO model    : {args.yolo_model}")
    print(f"  FairFace model: {args.fairface_model}")
    print(f"  Max images    : {args.max_images}")
    print(f"  Device        : {args.device}")
    print(f"  CSV output    : {args.csv_output}")

    os.makedirs(args.output_folder, exist_ok=True)

    # =========================
    # YOLO model
    # =========================
    model = YOLO(args.yolo_model)

    # =========================
    # FairFace Model
    # =========================
    fairface = models.resnet34(pretrained=False)
    fairface.fc = nn.Linear(fairface.fc.in_features, 18)

    state_dict = torch.load(args.fairface_model, map_location=args.device)
    fairface.load_state_dict(state_dict)
    fairface.to(args.device)
    fairface.eval()

    # =========================
    # Labels
    # =========================
    race_labels = [
        "White", "Black", "Latino_Hispanic",
        "East Asian", "Southeast Asian", "Indian", "Middle Eastern"
    ]
    gender_labels = ["Male", "Female"]
    age_labels = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    # =========================
    # Image Transform
    # =========================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # =========================
    # Process images
    # =========================
    count = 0
    records = []   # accumulates one dict per detected face
    loop = tqdm(os.listdir(args.input_folder), desc="Processing images")

    for filename in loop:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            if count >= args.max_images:
                print("Stopping pipeline due to image limit.")
                break

            loop.set_postfix(file=filename)

            image_path = os.path.join(args.input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"  WARNING: Could not read {filename}, skipping.")
                continue

            img_h, img_w = image.shape[:2]
            results = model(image, verbose=False)

            face_idx = 0  # per-image face counter

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                cls_ids = r.boxes.cls.cpu().int()
                person_mask = (cls_ids == 0)
                if not person_mask.any():
                    continue

                boxes      = r.boxes.xyxy[person_mask]
                confidence = r.boxes.conf[person_mask]

                for box, conf in zip(boxes, confidence):
                    x1, y1, x2, y2 = map(int, box)
                    face = image[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    face_pil    = Image.fromarray(face)
                    face_tensor = transform(face_pil).unsqueeze(0).to(args.device)

                    with torch.no_grad():
                        outputs = fairface(face_tensor)[0]

                        race_output   = outputs[0:7]
                        gender_output = outputs[7:9]
                        age_output    = outputs[9:18]

                        # Softmax probabilities for confidence scores
                        race_probs   = torch.softmax(race_output,   dim=0)
                        gender_probs = torch.softmax(gender_output, dim=0)
                        age_probs    = torch.softmax(age_output,    dim=0)

                        race_idx   = torch.argmax(race_probs).item()
                        gender_idx = torch.argmax(gender_probs).item()
                        age_idx    = torch.argmax(age_probs).item()

                        race   = race_labels[race_idx]
                        gender = gender_labels[gender_idx]
                        age    = age_labels[age_idx]

                        race_conf   = round(race_probs[race_idx].item(),    4)
                        gender_conf = round(gender_probs[gender_idx].item(), 4)
                        age_conf    = round(age_probs[age_idx].item(),      4)

                    # ── Build one record per face ──────────────────────────
                    records.append({
                        "filename":       filename,
                        # "image_width":    img_w,
                        # "image_height":   img_h,
                        "face_index":     face_idx,
                        "bbox_x1":        x1,
                        "bbox_y1":        y1,
                        "bbox_x2":        x2,
                        "bbox_y2":        y2,
                        "bbox_width":     x2 - x1,
                        "bbox_height":    y2 - y1,
                        "detection_conf": round(conf.item(), 4),
                        "race":           race,
                        "race_conf":      race_conf,
                        "gender":         gender,
                        "gender_conf":    gender_conf,
                        "age":            age,
                        "age_conf":       age_conf,
                    })

                    # label = f"{race} | {gender} | {age}"
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.putText(
                    #     image, label, (x1, y1 - 10),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    # )

                    face_idx += 1

            # output_path = os.path.join(args.output_folder, filename)
            # cv2.imwrite(output_path, image)
            count += 1

    # =========================
    # Save DataFrame to CSV
    # =========================
    csv_path = os.path.join(args.output_folder, args.csv_output)

    if records:
        df = pd.DataFrame(records, columns=[
            "filename",
            "image_width", "image_height",
            "face_index",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "bbox_width", "bbox_height",
            "detection_conf",
            "race",   "race_conf",
            "gender", "gender_conf",
            "age",    "age_conf",
        ])
        df.to_csv(csv_path, index=False)
        print(f"\nDataFrame saved to '{csv_path}' ({len(df)} face record(s)).")
        print(df.head().to_string(index=False))
    else:
        print("\nNo faces detected — CSV not written.")

    print(f"\nProcessing finished. {count} image(s) saved to '{args.output_folder}'.")


if __name__ == "__main__":
    main()