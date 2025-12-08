# Vision-Depth Fusion for Posture Stability Detection

This repository implements multiple RGB-D multimodal deep learning models for human posture stability classification on the UTD-MHAD dataset.

It includes:

- RGB-only and Depth-only baselines  
- Early Fusion (feature concatenation)  
- Attention Fusion (learned modality weighting)  
- **Core Fusion (Attention + Contrastive Learning + Uncertainty Estimation)**  
- Missing-modality robustness evaluation  
- Calibration analysis (ECE + reliability diagrams)

The project compares how different fusion strategies behave under:
1. Full RGB-D input  
2. Missing RGB  
3. Missing Depth  

The primary objective is demonstrating that **Core Fusion exceeds Attention Fusion in full-modality accuracy and is significantly more robust under missing modalities.**

---

## Project Goals

1. Build strong unimodal and multimodal baseline models.  
2. Implement a unified **Core Fusion** model combining:
   - Attention-based modality weighting  
   - InfoNCE contrastive embedding alignment  
   - Per-modality uncertainty estimation  
3. Achieve **higher or equal accuracy vs. Attention Fusion** on full RGB-D input.  
4. Demonstrate **substantially improved missing-modality robustness**.  
5. Evaluate model calibration using **Expected Calibration Error (ECE)**.  

---

## Repository Structure

```
multimodal_posture/
│
├── config/
|   ├── depth_baseline.yaml
│   ├── fusion_core.yaml
│   ├── fusion_attention.yaml
│   ├── fusion_early.yaml
|   └── rgb_baseline.yaml
│
├── checkpoints/
│   ├── depth_baseline/
│   │   └──  depth_baseline_best.pt
│   ├── fusion_core/
│   │   └──  fusion_core_best.pt
│   ├── fusion_attention/
│   │   └── fusion_attention_best.pt
│   ├── fusion_early/
│   |   └── fusion_early_best.pt
│   └── rgb_baseline/
│       └──  rgb_baseline_best.pt
│
├── data /
│   └── utd_mhad/ (train/val/test CSV splits)
|
├── datasets/
│   ├── dataloaders.py
|   └── utd_mhad_rgbd.py
│
├── logs/
│   └── (all slurm script results)
│
├── models/
│   ├── rgb_depth_baselines.py (RGB, Depth Baselines)
│   └── multimodal_rgbd.py
│       (Early Fusion,
│        Attention Fusion, Core Fusion)
│
├── src/
│   ├── train_rgb.py
│   ├── train_depth.py
│   ├── train_fusion_early.py
│   ├── train_fusion_attention.py
│   ├── train_fusion_core.py
│   ├── compute_ece.py
│   └── eval_missing_modalities.py
|
├── scripts/
│   ├── check_dataset.py
│   ├── make_index.py
│   ├── make_splits.py
│   ├── run_all_fusions.ps1
│   ├── test_fusion_models.py
│   └── test_labels.py
│
├── results/
|   ├── calibration/
|   ├── missing_modalities/
|   ├── fusion_attention/
|   ├── fusion_early/
|   ├── fusion_core/
|   ├── rgb_baseline/
|   └── depth_baseline/
|
└── (slurm scripts for all the models)
    
```

---

## Dataset: UTD-MHAD RGB-D

Dataset includes:
- RGB video frames  
- Depth video frames  
- IMU data (not used here)

We follow subject-independent splits:
- Certain subjects → train  
- Others → validation  
- Unseen subjects → test  

CSV files in `data/utd_mhad/splits/` define these partitions.

---

## Model Overview

### 1. RGB Baseline  
- ResNet-18 backbone  
- Single-stream classifier  

### 2. Depth Baseline  
- Same architecture as RGB  
- 1-channel depth input  

### 3. Early Fusion  
- Concatenate RGB + Depth embeddings  
- MLP classifier  

### 4. Attention Fusion  
Learns modality weights:

```
α_rgb, α_depth = softmax(MLP([z_rgb ; z_depth]))
z_fused = α_rgb·z_rgb + α_depth·z_depth
```

### 5. Core Fusion (Proposed Model)

Extends Attention Fusion with:

✔ **Contrastive Alignment (InfoNCE)**  
✔ **Uncertainty Estimation**  
✔ **Unified loss:** CE + Contrastive + Attention entropy + Uncertainty regularizer  

---

## Installation

```
conda create -n multimodal_posture python=3.10
conda activate multimodal_posture
pip install torch torchvision torchaudio matplotlib scikit-learn pyyaml
```

---

## Training Models

### RGB-only  
```
python src/train_baseline_rgb.py config/rgb_baseline.yaml
```

### Depth-only  
```
python src/train_baseline_depth.py config/depth_baseline.yaml
```

### Early Fusion  
```
python src/train_fusion_early.py config/fusion_early.yaml
```

### Attention Fusion  
```
python src/train_fusion_attention.py config/fusion_attention.yaml
```

### Core Fusion  
```
python src/train_fusion_core.py config/fusion_core.yaml
```

Best checkpoint saved to:  
```
checkpoints/fusion_{}/fusion_{}_best.pt
```

**Note:** All the models have their own individual slurm scripts to train remotely.

---

## Evaluation

### Full-Modality Metrics

Saved to:

```
results/fusion_{}/fusion_{}_results.json
```

Includes:
- Accuracy  
- Macro F1  
- Confusion matrix  
- Attention statistics  

---

## Missing Modality Robustness

Run:

```
python src/eval_missing_modalities.py
```

Outputs to:

```
results/missing_modalities/
```

---

## Model Calibration (ECE)

Run:

```
python src/compute_ece.py
```

Outputs to:

```
results/calibration/
```

---

## Interpretation of Expected Behaviors

- **Core Fusion should match or exceed Attention Fusion in full-modality accuracy**  
- **Core Fusion should strongly outperform when Depth is missing**  
- **Attention Fusion may perform better when RGB is missing (dataset-dependent)**  

---

## Citations

[1] UTD-MHAD RGB-D Human Action Dataset  
[2] Attention Fusion: RGB-D Human Action Recognition (TIP 2020)  
[3] Kendall & Gal — Bayesian Deep Learning Uncertainty (NIPS 2017)  
[4] SimCLR/InfoNCE Contrastive Loss (Chen et al., 2020)

---
