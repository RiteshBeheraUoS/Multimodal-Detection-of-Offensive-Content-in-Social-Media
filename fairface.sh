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
mkdir -p ~/img_logs

# Load conda
source /iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh
conda activate rkb_pyEnv

# ---------------------------------------------------------------------------
# Paths — edit these to change input/output locations
# ---------------------------------------------------------------------------

echo "=== Checking script ==="
ls ~/fairface_feature.py

echo "=== Checking input dataset ==="
ls ~/img_logs/clean3 | head -5

cd ~
python fairface_feature.py \
    --input_folder  ~/img_logs/clean3    \
    --output_folder  ~/img_logs   \
    --yolo_model     ~/models/yolov8n.pt      \
    --fairface_model ~/models/res34_fair_align_multi_7_20190809.pt  \
    --max_images     1000000

echo "=== Job finished at $(date) ==="