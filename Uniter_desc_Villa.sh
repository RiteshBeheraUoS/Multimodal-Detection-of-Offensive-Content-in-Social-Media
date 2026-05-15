#!/bin/bash -l
#SBATCH -p ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rkb1u25@soton.ac.uk
#SBATCH --time=06:00:00
#SBATCH --output=/home/rkb1u25/img_logs/slurm-%j.out
#SBATCH --error=/home/rkb1u25/img_logs/slurm-%j.err

echo "=== Job started at $(date) ==="

# Ensure logs dir exists
mkdir -p ~/img_logs/uniter_desc_checkpoints

# Load conda
source /iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh
conda activate rkb_pyEnv

# ---------------------------------------------------------------------------
# Paths — edit these to change input/output locations
# ---------------------------------------------------------------------------

echo "=== Checking training script ==="
ls ~/Uniter_desc_Villa.py

echo "=== Checking input data ==="
ls ~/img_logs/clean3 | head -5
ls ~/img_logs/merged_dataset.csv | head -5
ls ~/img_logs/object_detections.csv | head -5
ls ~/img_logs/human_results.csv | head -5

echo "Starting UNITER / VILLA-ITM fine-tuning..."

python ~/Uniter_desc_Villa.py \
    --captions_csv  ~/img_logs/merged_dataset.csv           \
    --desc_csv      ~/img_logs/desc.csv                     \
    --objects_csv   ~/img_logs/object_detections.csv        \
    --faces_csv     ~/img_logs/human_results.csv            \
    --image_dir     ~/img_logs/clean3                       \
    --output_dir    ~/img_logs/uniter_desc_checkpoints     \
    --pretrained_name bert-base-uncased                     \
    --resnet_weights ~/resnet50_imagenet.pth                \
    --img_size      224                                     \
    --max_text_len  128                                     \
    --max_visual_len 36                                     \
    --val_frac      0.10                                    \
    --num_workers   4                                       \
    --hidden_dim    256                                     \
    --dropout       0.3                                     \
    --freeze_layers 6                                       \
    --epochs        20                                      \
    --batch_size    16                                      \
    --backbone_lr   2e-5                                    \
    --head_lr       1e-4                                    \
    --weight_decay  0.01                                    \
    --grad_clip     1.0                                     \
    --warmup_ratio  0.1                                     \
    --pos_weight    1.0                                     \
    --early_stopping_patience 10                            \
    --log_every     50                                      \
    --seed          42
#   --pretrained_name ~/uniter_base          # Uncomment to use local UNITER weights
#   --pretrained_name ~/villa_base           # Uncomment to use local VILLA weights
#   --max_visual_len 64                      # Uncomment to use more region tokens (check L+R < 512)
#   --pos_weight    2.5                      # Uncomment if dataset is imbalanced (num_bad / num_good)
#   --freeze_layers 0                        # Uncomment to fine-tune all backbone layers
#   --epochs        30                       # Uncomment to extend training

echo "=== Job finished at $(date) ==="