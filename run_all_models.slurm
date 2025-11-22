#!/bin/bash
#SBATCH --job-name=utd_all_models
#SBATCH --partition=gpu
#SBATCH --nodelist=gpuk[001-006]
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/all_models_%j.out

# Email notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rd935@rutgers.edu

# ------------------ Environment Setup ------------------

# Use your personal conda initialization instead of module load
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate multimodal_posture

# Go to your project directory
cd ~/multimodal_posture   # adjust if different

# Make logs directory if needed
mkdir -p logs

run() {
    echo
    echo "============================================================"
    echo ">>> $(date) : $*"
    echo "============================================================"
    # With a single task in this job, plain python is fine; srun is optional
    python "$@"
}

echo "---------------------------------------"
echo "Running on host: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "---------------------------------------"

nvidia-smi

# ------------------ Run Training ------------------

# RGB-only model
run src/train_baseline_rgb.py config/rgb_baseline.yaml

# Depth-only model
run src/train_baseline_depth.py config/depth_baseline.yaml

# Early fusion model
run src/train_fusion_early.py config/fusion_early.yaml

# Plain attention fusion (your original attention model)
run src/train_fusion_attention.py config/fusion_attention.yaml

# Core fusion model: attention + contrastive + uncertainty
run src/train_fusion_core.py config/fusion_core.yaml

echo
echo "[$(date)] All models finished."
