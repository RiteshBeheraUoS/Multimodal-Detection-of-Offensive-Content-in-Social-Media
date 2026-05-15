import os
import ast
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    BertModel,
    BertConfig,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    classification_report,
)

"""
UNITER_VILLA_ITM.py — Multimodal binary classifier using UNITER / VILLA ITM backbone.

Architecture overview
─────────────────────
UNITER and VILLA share the same backbone design: a single-stream BERT that
receives a *concatenated* sequence of text tokens followed by visual region
tokens (image-text matching / ITM objective).  We replicate that design here
using the HuggingFace BertModel with a custom visual-token projection layer so
the pipeline can be loaded from any BERT-compatible checkpoint
(e.g. "bert-base-uncased") or from the official UNITER weights if available
locally.

Inputs
──────
  - Raw image    → ResNetRegionExtractor → (B, R, 2048) region features
  - Caption text = caption + YOLO object tokens → single-stream BERT with image
  - Desc text    = per-image description from desc.csv → second BERT pass
                   (shared backbone weights, text-only, pooled [CLS])
  - Structured   = FairFace 8-dim feature vector (fused at the classifier head)
  - Label        = 0 (bad) or 1 (good)

The desc stream is encoded independently (text-only BERT forward pass) and its
pooled [CLS] is projected and concatenated with the caption+image CLS and the
FairFace vector before the binary classifier.  This keeps the UNITER ITM
single-stream design intact for the primary caption↔image fusion while giving
the model a separate, dedicated slot for the richer description signal.

Sequence layout
───────────────
  Stream A (caption + image — UNITER ITM):
      [CLS] caption_tokens [SEP] visual_region_tokens
       → pooler → (B, 768) → fusion_proj → (B, 256)

  Stream B (desc — text-only BERT):
      [CLS] desc_tokens [SEP]
       → pooler → (B, 768) → desc_proj  → (B, 128)

  FairFace:
      (B, 8) → fairface_encoder → (B, 16)

  Classifier:
      concat([256, 128, 16]) = (B, 400) → MLP → logit (B,)

Checkpoint options
──────────────────
  --pretrained_name  bert-base-uncased          (HuggingFace; always available)
  --pretrained_name  /path/to/uniter-base       (local UNITER weights)
  --pretrained_name  /path/to/villa-base        (local VILLA weights)
"""


# ── helpers ──────────────────────────────────────────────────────────────────
# (Unchanged from VL-BERT version)

def build_object_token_string(obj_rows: pd.DataFrame) -> str:
    """
    Convert YOLO rows for one image into a compact token string appended
    to the caption so the model can cross-attend text ↔ image regions.

    Example output:
        "[OBJ: person 0.95] [OBJ: car 0.87] [OBJ: traffic_light 0.72]"
    """
    if obj_rows.empty:
        return ""
    obj_rows = obj_rows.sort_values("confidence", ascending=False).head(10)
    parts = [
        f"[OBJ: {row.object_label} {row.confidence:.2f}]"
        for row in obj_rows.itertuples()
    ]
    return " ".join(parts)


def build_fairface_vector(face_rows: pd.DataFrame) -> torch.Tensor:
    """
    Build a fixed-size numeric feature vector from all detected faces.

    Strategy: average across faces (handles 0-N faces gracefully).
    Vector layout per face (8 dims):
        bbox_width, bbox_height, detection_conf,
        race_conf, gender_conf, age_conf,
        is_male (binary),
        age_bucket (0-6 ordinal)

    Returns a float32 tensor of shape (8,).
    """
    AGE_BUCKET = {
        "0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3,
        "30-39": 4, "40-49": 5, "50-59": 6, "60-69": 6, "70+": 6,
    }

    if face_rows.empty:
        return torch.zeros(8, dtype=torch.float32)

    vecs = []
    for row in face_rows.itertuples():
        age_bucket = AGE_BUCKET.get(str(row.age), 3)
        is_male    = 1.0 if str(row.gender).lower() == "male" else 0.0
        vecs.append([
            row.bbox_width,
            row.bbox_height,
            row.detection_conf,
            row.race_conf,
            row.gender_conf,
            row.age_conf,
            is_male,
            float(age_bucket),
        ])
    return torch.tensor(vecs, dtype=torch.float32).mean(dim=0)


# ── dataset ──────────────────────────────────────────────────────────────────
# Identical to VLBertBinaryDataset; renamed for clarity.

class UNITERBinaryDataset(Dataset):
    """
    Args:
        captions_df    : DataFrame with columns [filename, text, label, split]
        desc_df        : DataFrame with columns [filename, desc]  ← new
        objects_df     : DataFrame with YOLO records (one row per detected object)
        faces_df       : DataFrame with FairFace records (one row per face)
        image_dir      : Root directory containing the image files
        tokenizer      : HuggingFace tokenizer for the BERT variant
        image_transform: torchvision transform applied to the PIL image
        max_text_len   : Max token length for caption + YOLO object tokens
        max_desc_len   : Max token length for the description text  ← new
    """

    def __init__(
        self,
        captions_df: pd.DataFrame,
        desc_df: pd.DataFrame,
        objects_df: pd.DataFrame,
        faces_df: pd.DataFrame,
        image_dir: str,
        tokenizer,
        image_transform,
        max_text_len: int = 128,
        max_desc_len: int = 128,
    ):
        self.captions   = captions_df.reset_index(drop=True)
        self.objects_df = objects_df
        self.faces_df   = faces_df
        self.image_dir  = image_dir
        self.tokenizer  = tokenizer
        self.transform  = image_transform
        self.max_len    = max_text_len
        self.max_desc_len = max_desc_len

        # Build a filename → desc lookup for O(1) access
        self.desc_lookup = (
            desc_df.set_index("filename")["desc"].to_dict()
            if desc_df is not None and not desc_df.empty
            else {}
        )

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row      = self.captions.iloc[idx]
        filename = row["filename"]
        caption  = str(row["text"])
        label    = int(row["label"])  # 0 or 1

        # ── 1. Image ──────────────────────────────────────────────────────
        img_path     = os.path.join(self.image_dir, filename)
        image        = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)          # (C, H, W) float tensor

        # ── 2. Caption + YOLO object tokens (primary text stream) ─────────
        obj_key    = int(os.path.splitext(filename)[0])
        obj_rows   = self.objects_df[self.objects_df["filename"] == obj_key]
        obj_string = build_object_token_string(obj_rows)
        full_text  = caption.strip()
        if obj_string:
            full_text = full_text + " " + obj_string

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)       # (L,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (L,)

        # ── 3. Description text (secondary text stream) ───────────────────
        # Fallback to empty string if a description is missing for this image.
        desc_text = str(self.desc_lookup.get(filename, "")).strip()
        desc_encoding = self.tokenizer(
            desc_text,
            max_length=self.max_desc_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        desc_input_ids      = desc_encoding["input_ids"].squeeze(0)       # (D,)
        desc_attention_mask = desc_encoding["attention_mask"].squeeze(0)  # (D,)

        # ── 4. FairFace feature vector ────────────────────────────────────
        face_key  = os.path.splitext(filename)[0] + ".png"
        face_rows = self.faces_df[self.faces_df["filename"] == face_key]
        fairface_vec = build_fairface_vector(face_rows)         # (8,)

        return {
            "pixel_values":       pixel_values,
            "input_ids":          input_ids,
            "attention_mask":     attention_mask,
            "desc_input_ids":     desc_input_ids,
            "desc_attention_mask": desc_attention_mask,
            "fairface_vec":       fairface_vec,
            "label":              torch.tensor(label, dtype=torch.long),
            "filename":           filename,
        }


# ── UNITER / VILLA-ITM backbone ───────────────────────────────────────────────

class UNITERITMClassifier(nn.Module):
    """
    Binary image-quality classifier using UNITER / VILLA ITM architecture,
    extended with a separate description text stream.

    Architecture
    ────────────
    Stream A — caption + image (UNITER ITM single-stream):
        [CLS] caption_tokens [SEP] visual_region_tokens
         → BERT encoder → pooler → (B, 768) → fusion_proj → (B, 256)

    Stream B — description text (text-only BERT, shared backbone weights):
        [CLS] desc_tokens [SEP]
         → backbone.embeddings → backbone.encoder → pooler → (B, 768)
         → desc_proj → (B, 128)

    FairFace metadata:
        (B, 8) → fairface_encoder → (B, 16)

    Classifier head:
        concat([256, 128, 16]) = (B, 400) → MLP → logit (B,)

    Sharing backbone weights between the two streams:
      - Halves the parameter count vs. two independent BERTs
      - Both text inputs benefit from the same pretrained representations
      - The two streams are differentiated only by their input sequences
        (stream A includes visual tokens; stream B is text-only)

    Args:
        pretrained_name : HuggingFace model id or local path.
        visual_feat_dim : Dimensionality of region features (2048).
        fairface_dim    : Size of the FairFace feature vector (8).
        hidden_dim      : Bottleneck size for the caption+image fusion proj (256).
        desc_dim        : Bottleneck size for the description proj (128).
        dropout         : Dropout in the classifier head.
        freeze_layers   : Number of BERT encoder layers to freeze from the bottom.
        max_visual_len  : Maximum visual region tokens to concatenate (≤ R).
    """

    def __init__(
        self,
        pretrained_name:  str   = "bert-base-uncased",
        visual_feat_dim:  int   = 2048,
        fairface_dim:     int   = 8,
        hidden_dim:       int   = 256,
        desc_dim:         int   = 128,
        dropout:          float = 0.3,
        freeze_layers:    int   = 6,
        max_visual_len:   int   = 36,
    ):
        super().__init__()

        self.max_visual_len = max_visual_len

        # ── Shared BERT backbone (UNITER / VILLA single-stream) ────────────
        self.backbone  = BertModel.from_pretrained(pretrained_name)
        bert_hidden    = self.backbone.config.hidden_size  # typically 768

        self._freeze_layers(freeze_layers)

        # ── Visual region projection: (B, R, 2048) → (B, R, 768) ───────────
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_feat_dim, bert_hidden),
            nn.LayerNorm(bert_hidden),
            nn.GELU(),
        )

        # ── Stream A: caption+image fusion projection ───────────────────────
        self.fusion_proj = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Stream B: description projection ───────────────────────────────
        self.desc_proj = nn.Sequential(
            nn.Linear(bert_hidden, desc_dim),
            nn.LayerNorm(desc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── FairFace encoder ────────────────────────────────────────────────
        self.fairface_encoder = nn.Sequential(
            nn.Linear(fairface_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # ── Binary classifier head ──────────────────────────────────────────
        # 256 (caption+image) + 128 (desc) + 16 (fairface) = 400
        fused_dim = hidden_dim + desc_dim + 16
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),               # raw logit (BCEWithLogitsLoss)
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _freeze_layers(self, n: int):
        """Freeze the embedding layer and the first n encoder layers."""
        if n == 0:
            return
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        for layer in self.backbone.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:             torch.Tensor,  # (B, L)       text token ids
        attention_mask:        torch.Tensor,  # (B, L)       text padding mask
        visual_embeds:         torch.Tensor,  # (B, R, 2048) region features
        visual_attention_mask: torch.Tensor,  # (B, R)       visual padding mask
        fairface_vec:          torch.Tensor,  # (B, 8)       face metadata
        desc_input_ids:        torch.Tensor,
        desc_attention_mask:   torch.Tensor
    ) -> torch.Tensor:
        """
        UNITER / VILLA-ITM forward pass.

        The text and visual tokens are concatenated into a single flat sequence
        and processed together by BERT self-attention (single-stream fusion),
        matching the ITM pretraining objective of UNITER / VILLA.

        Returns raw logits of shape (B,).  Apply sigmoid for probabilities.
        """
        B, L = input_ids.shape
        R    = min(visual_embeds.size(1), self.max_visual_len)
        visual_embeds = visual_embeds[:, :R, :]              # (B, R, 2048)
        visual_attention_mask = visual_attention_mask[:, :R] # (B, R)

        # ── Step 1: Project visual regions into BERT's embedding space ──────
        # (B, R, 2048) → (B, R, 768)
        vis_tokens = self.visual_proj(visual_embeds)

        # ── Step 2: Get text embeddings from BERT's embedding layer ─────────
        # We pass input_ids through the embedding layer only, then concatenate
        # with the visual tokens before the encoder layers.
        # shape: (B, L, 768)
        txt_embeds = self.backbone.embeddings(input_ids=input_ids)

        # ── Step 3: Concatenate [text | visual] into one flat sequence ───────
        # UNITER / VILLA single-stream: [CLS] t₁…tₙ [SEP] v₁…vₖ
        # shape: (B, L+R, 768)
        combined_embeds = torch.cat([txt_embeds, vis_tokens], dim=1)

        # Build extended attention mask (text mask + visual mask)
        # shape: (B, L+R)
        combined_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        # BERT expects the attention mask in extended form for the encoder
        # (0 → attend, large negative → ignore); BertModel handles this
        # internally when we pass `attention_mask`.
        encoder_outputs = self.backbone.encoder(
            hidden_states=combined_embeds,
            attention_mask=self.backbone.get_extended_attention_mask(
                combined_mask, combined_mask.shape
            ),
        )
        sequence_output = encoder_outputs.last_hidden_state  # (B, L+R, 768)

        # ── Step 4: Pool [CLS] token (position 0 = text CLS) ────────────────
        # Identical to UNITER ITM head — use [CLS] pooled output.
        cls_output = self.backbone.pooler(sequence_output)   # (B, 768)

        # ── Step 5: Fusion projection ────────────────────────────────────────
        fused = self.fusion_proj(cls_output)                 # (B, 256)

        # ── Description text stream ───────────────────────────────
        desc_outputs = self.backbone(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask,
        )

        # pooled CLS representation
        desc_cls = desc_outputs.pooler_output  # (B, 768)

        # project to smaller embedding
        desc_feat = self.desc_proj(desc_cls)  # (B, 128)

        # ── Step 6: Encode FairFace metadata ─────────────────────────────────
        ff_encoded = self.fairface_encoder(fairface_vec)     # (B, 16)

        # ── Step 7: Concatenate and classify ─────────────────────────────────
        combined = torch.cat([fused, desc_feat, ff_encoded], dim=-1)    # (B, 272)
        logits   = self.classifier(combined).squeeze(-1)     # (B,)

        return logits


# ── visual feature extractor (pixel → region features) ───────────────────────
# Unchanged from VL-BERT version.

class ResNetRegionExtractor(nn.Module):
    """
    Lightweight adapter: converts a (B, C, H, W) pixel tensor into
    (B, R, 2048) region feature vectors using a pretrained ResNet-50.

    Args:
        num_regions       : number of spatial regions (default 36 = 6×6 grid)
        local_weights_path: path to a pre-saved ResNet-50 state dict
    """

    def __init__(
        self,
        num_regions: int = 36,
        local_weights_path: str = "/home/rkb1u25/resnet50_imagenet.pth",
    ):
        super().__init__()
        import torchvision.models as models

        resnet     = models.resnet50(weights=None)
        state_dict = torch.load(local_weights_path, map_location="cpu")
        resnet.load_state_dict(state_dict)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool              = nn.AdaptiveAvgPool2d((6, 6))  # 6×6 = 36 regions
        self.num_regions       = num_regions

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor):
        """
        Args:
            pixel_values: (B, 3, H, W)
        Returns:
            region_feats: (B, R, 2048)
            region_mask:  (B, R)  — all ones (all regions valid)
        """
        feats      = self.feature_extractor(pixel_values)  # (B, 2048, h, w)
        feats      = self.pool(feats)                       # (B, 2048, 6, 6)
        B, C, H, W = feats.shape
        feats      = feats.view(B, C, H * W)               # (B, 2048, 36)
        feats      = feats.permute(0, 2, 1)                # (B, 36, 2048)
        mask       = torch.ones(B, H * W, dtype=torch.long, device=feats.device)
        return feats, mask


# ── argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune UNITER / VILLA-ITM for binary image quality classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("Paths")
    paths.add_argument("--captions_csv", type=str, required=True,
                       help="CSV with columns: filename, text, label")
    paths.add_argument("--objects_csv",  type=str, required=True,
                       help="CSV of YOLO object records")
    paths.add_argument("--faces_csv",    type=str, required=True,
                       help="CSV of FairFace human-feature records")
    paths.add_argument("--image_dir",    type=str, required=True,
                       help="Root directory containing the image files")
    paths.add_argument("--output_dir",   type=str, default="./checkpoints",
                       help="Directory for checkpoints and training log")
    paths.add_argument("--desc_csv", type=str, required=True,
                       help="CSV with columns: filename, desc, label")

    data = parser.add_argument_group("Data")
    data.add_argument("--img_size",     type=int,   default=224)
    data.add_argument("--max_text_len", type=int,   default=128,
                      help="Max token length for caption + YOLO object tokens")
    data.add_argument("--val_frac",     type=float, default=0.1)
    data.add_argument("--num_workers",  type=int,   default=4)

    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--pretrained_name", type=str, default="bert-base-uncased",
        help=(
            "HuggingFace model id or local path for UNITER / VILLA-ITM backbone. "
            "Use 'bert-base-uncased' to start from a standard BERT checkpoint, "
            "or supply a path to locally saved UNITER / VILLA weights."
        ),
    )
    model_grp.add_argument("--resnet_weights", type=str,
                           default="/home/rkb1u25/resnet50_imagenet.pth",
                           help="Path to locally saved ResNet-50 state dict")
    model_grp.add_argument("--hidden_dim",     type=int,   default=256)
    model_grp.add_argument("--dropout",        type=float, default=0.3)
    model_grp.add_argument("--freeze_layers",  type=int,   default=6,
                           help="Number of bottom BERT encoder layers to freeze")
    model_grp.add_argument("--max_visual_len", type=int,   default=36,
                           help="Max number of visual region tokens concatenated "
                                "into the single-stream sequence")

    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--epochs",      type=int,   default=20)
    train_grp.add_argument("--batch_size",  type=int,   default=16)
    train_grp.add_argument("--backbone_lr", type=float, default=2e-5)
    train_grp.add_argument("--head_lr",     type=float, default=1e-4)
    train_grp.add_argument("--weight_decay",type=float, default=0.01)
    train_grp.add_argument("--grad_clip",   type=float, default=1.0)
    train_grp.add_argument("--warmup_ratio",type=float, default=0.1)
    train_grp.add_argument("--pos_weight",  type=float, default=1.0,
                           help="BCEWithLogitsLoss positive class weight")
    train_grp.add_argument("--early_stopping_patience", type=int, default=5)
    train_grp.add_argument("--log_every",   type=int,   default=50)
    train_grp.add_argument("--seed",        type=int,   default=42)

    return parser.parse_args()


# ── data helpers ──────────────────────────────────────────────────────────────

def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_dataloaders(args):
    captions_df = pd.read_csv(args.captions_csv)
    objects_df  = pd.read_csv(args.objects_csv)
    faces_df    = pd.read_csv(args.faces_csv)
    desc_df     = pd.read_csv(args.desc_csv)

    tokenizer        = AutoTokenizer.from_pretrained(args.pretrained_name)
    train_tf, val_tf = get_transforms(args.img_size)

    labelled_captions = captions_df[captions_df["split"] == "train"].reset_index(drop=True)
    test_captions     = captions_df[captions_df["split"] == "valid"].reset_index(drop=True)
    validation_size   = len(test_captions)

    train_captions, val_captions = train_test_split(
        labelled_captions, test_size=validation_size, random_state=42, shuffle=True
    )

    print(f"[INFO] Split sizes — train: {len(train_captions)}  "
          f"val: {len(val_captions)}  test: {len(test_captions)}")

    def make_ds(cap_df, transform):
        return UNITERBinaryDataset(
            captions_df=cap_df,
            desc_df= desc_df,
            objects_df=objects_df,
            faces_df=faces_df,
            image_dir=args.image_dir,
            tokenizer=tokenizer,
            image_transform=transform,
            max_text_len=args.max_text_len,
        )

    train_ds = make_ds(train_captions, train_tf)
    val_ds   = make_ds(val_captions,   val_tf)
    test_ds  = make_ds(test_captions,  val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── optimizer + scheduler ─────────────────────────────────────────────────────

def build_optimizer_and_scheduler(model, visual_encoder, args, num_train_steps):
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),         "lr": args.backbone_lr},
        {"params": model.visual_proj.parameters(),      "lr": args.head_lr},
        {"params": model.fusion_proj.parameters(),      "lr": args.head_lr},
        {"params": model.fairface_encoder.parameters(), "lr": args.head_lr},
        {"params": model.classifier.parameters(),       "lr": args.head_lr},
        {"params": visual_encoder.parameters(),         "lr": 0.0},   # ResNet frozen
    ], weight_decay=args.weight_decay)

    warmup_steps = int(num_train_steps * args.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_steps,
    )
    return optimizer, scheduler


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, visual_encoder, loader, criterion, device):
    model.eval()
    visual_encoder.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    loop = tqdm(enumerate(loader), total=len(loader))
    loop.set_description("Evaluating....")

    for step, batch in loop:
        pixel_values = batch["pixel_values"].to(device)
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        fairface_vec = batch["fairface_vec"].to(device)
        desc_input_ids = batch["desc_input_ids"].to(device)
        desc_attention_mask = batch["desc_attention_mask"].to(device)
        labels       = batch["label"].to(device).float()

        visual_embeds, visual_mask = visual_encoder(pixel_values)

        logits = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_mask,
            fairface_vec=fairface_vec,
            desc_input_ids=desc_input_ids,
            desc_attention_mask=desc_attention_mask
        )
        loss        = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    avg_loss  = total_loss / len(loader)
    acc       = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    auc       = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm        = confusion_matrix(all_labels, all_preds)
    return {
        "loss": avg_loss, "acc": acc, "precision": precision,
        "confusion_matrix": cm, "f1": f1, "auc": auc,
        "labels": all_labels, "preds": all_preds,
    }


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ──
    train_loader, val_loader, test_loader = build_dataloaders(args)
    # print(f"[INFO] Train={len(train_loader.dataset)} "
    #       f"Val={len(val_loader.dataset)} Test={len(test_loader.dataset)}")

    # ── Models ──
    visual_encoder = ResNetRegionExtractor(
        num_regions=36,
        local_weights_path=args.resnet_weights,
    ).to(device)

    model = UNITERITMClassifier(
        pretrained_name=args.pretrained_name,
        visual_feat_dim=2048,
        fairface_dim=8,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_layers=args.freeze_layers,
        max_visual_len=args.max_visual_len,
    ).to(device)

    # ── Loss ──
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )

    # ── Optimizer / scheduler ──
    n_steps              = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, visual_encoder, args, n_steps
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Logging setup ──
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_auc"
        ])
        writer.writeheader()

    best_val_f1    = -1.0
    patience_count = 0

    # ── Epoch loop ──
    for epoch in range(1, args.epochs + 1):
        model.train()
        visual_encoder.eval()   # ResNet weights frozen
        epoch_loss = 0.0
        loop = tqdm(enumerate(train_loader, 1), total=len(train_loader))
        loop.set_description_str(f"Training epoch {epoch}")

        for step, batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            input_ids    = batch["input_ids"].to(device)
            attn_mask    = batch["attention_mask"].to(device)
            fairface_vec = batch["fairface_vec"].to(device)
            labels       = batch["label"].to(device).float()
            desc_input_ids = batch["desc_input_ids"].to(device)
            desc_attention_mask = batch["desc_attention_mask"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == "cuda")):
                with torch.no_grad():
                    visual_embeds, visual_mask = visual_encoder(pixel_values)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    visual_embeds=visual_embeds,
                    visual_attention_mask=visual_mask,
                    fairface_vec=fairface_vec,
                    desc_input_ids=desc_input_ids,
                    desc_attention_mask=desc_attention_mask
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

            if step % args.log_every == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} "
                      f"| Loss {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validation ──
        val_metrics = evaluate(model, visual_encoder, val_loader, criterion, device)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f}"
        )
        print("Confusion Matrix:")
        print(val_metrics["confusion_matrix"])

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "val_loss", "val_acc",
                "val_precision", "val_f1", "val_auc", "tn", "fp", "fn", "tp"
            ])
            tn, fp, fn, tp = val_metrics["confusion_matrix"].ravel()
            writer.writerow({
                "epoch":        epoch,
                "train_loss":   round(avg_train_loss, 6),
                "val_loss":     round(val_metrics["loss"], 6),
                "val_acc":      round(val_metrics["acc"], 6),
                "val_precision":round(val_metrics["precision"], 6),
                "val_f1":       round(val_metrics["f1"], 6),
                "val_auc":      round(val_metrics["auc"], 6),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })

        # ── Checkpoint ──
        if val_metrics["f1"] > best_val_f1:
            best_val_f1    = val_metrics["f1"]
            patience_count = 0
            ckpt_path      = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1":          best_val_f1,
                "args":            vars(args),
            }, ckpt_path)
            print(f"  ✓ New best checkpoint saved (val_f1={best_val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.early_stopping_patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

    # ── Final test evaluation ──
    print("\n── Test Set Evaluation ──")
    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, visual_encoder, test_loader, criterion, device)
    print(classification_report(
        test_metrics["labels"], test_metrics["preds"],
        target_names=["bad (0)", "good (1)"]
    ))
    print(f"Test AUC: {test_metrics['auc']:.4f}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)