#!/bin/bash
#SBATCH --job-name=fusion_attention_contrastive_rgbd
#SBATCH --partition=gpu
#SBATCH --nodelist=gpuk[001-006]
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fusion_attn_%j.out

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

echo "---------------------------------------"
echo "Running on host: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "---------------------------------------"

# Quick sanity test for CUDA
python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# ------------------ Run Training ------------------

python src/train_fusion_attention.py config/fusion_attention_contrastive.yaml

