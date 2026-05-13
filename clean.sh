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
mkdir -p ~/img_logs/clean1
mkdir -p ~/img_logs/clean2
mkdir -p ~/img_logs/clean3

# Load conda
source /iridisfs/ixsoftware/conda/miniconda-py3/etc/profile.d/conda.sh
conda activate rkb_pyEnv

# ---------------------------------------------------------------------------
# Paths — edit these to change input/output locations
# ---------------------------------------------------------------------------

echo "=== Checking script ==="
ls ~/remove_annotation.py

echo "=== Checking input dataset ==="
ls ~/meme_img | head -5

cd ~
python remove_annotation.py \
    --input-dir ~/meme_img \
    --out-dir1  ~/img_logs/clean1  \
    --out-dir2  ~/img_logs/clean2  \
    --out-dir3  ~/img_logs/clean3  \
    --deepfill-model-dir ~/deepfillv2-pytorch  \
    --easyocr-model-dir ~/.EasyOCR/model

echo "=== Job finished at $(date) ==="