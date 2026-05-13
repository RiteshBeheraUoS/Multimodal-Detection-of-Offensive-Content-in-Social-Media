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
ls ~/ocr_text.py

echo "=== Checking input dataset ==="
ls ~/meme_img | head -5

echo "Starting OCR pipeline..."

python ~/ocr_text.py \
    --images_dir  ~/meme_img  \
    --output_file ~/img_logs/textimage.csv \
    --psm 6                      \
    --oem 3                      \
    --min_dim 800
#    --limit 5               # Uncomment to cap the number of images
    # --no_invert               # Uncomment to disable auto-inversion
    # --tesseract_cmd $(which tesseract)   # Usually not needed after module load


echo "=== Job finished at $(date) ==="