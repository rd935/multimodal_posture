# src/compute_ece_core.py

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloaders import make_utd_mhad_loaders
from models.multimodal_rgbd import MultimodalRGBDCoreFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def expected_calibration_error(probs, labels, n_bins=15):
    """
    Standard ECE implementation.
    probs: N x C softmax probabilities
    labels: N ground truth labels
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        start = bin_boundaries[i]
        end = bin_boundaries[i + 1]

        mask = (confidences >= start) & (confidences < end)
        if mask.sum() == 0:
            continue

        bin_acc = (predictions[mask] == labels[mask]).mean()
        bin_conf = confidences[mask].mean()

        ece += np.abs(bin_acc - bin_conf) * (mask.sum() / len(labels))

    return ece


def load_config(path="config/fusion_core.yaml"):
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / path
    print(f"[INFO] Using config: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_model(cfg):
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    model = MultimodalRGBDCoreFusion(
        num_classes=3,
        embed_dim=model_cfg.get("embed_dim", 256),
        fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 512),
        attn_hidden_dim=model_cfg.get("attn_hidden_dim", 256),
        pretrained=model_cfg.get("pretrained", True),
        normalize_embeddings=model_cfg.get("normalize_embeddings", False),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        contrastive_temperature=loss_cfg.get("contrastive_temperature", 0.1),
    ).to(DEVICE)

    ckpt_dir = PROJECT_ROOT / cfg["logging"]["ckpt_dir"]
    ckpt_ft = ckpt_dir / "fusion_core_best_ft.pt"
    ckpt_main = ckpt_dir / "fusion_core_best.pt"
    ckpt = ckpt_ft if ckpt_ft.exists() else ckpt_main

    print(f"[INFO] Loading checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    return model


@torch.no_grad()
def gather_probs(model, loader):
    all_probs = []
    all_labels = []

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].cpu().numpy()

        logits = model(rgb, depth)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    cfg = load_config()
    _, _, test_loader = make_utd_mhad_loaders(
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        cfg["data"]["test_csv"],
        batch_size=cfg["train"].get("batch_size", 8),
        num_workers=cfg["loader"].get("num_workers", 4),
        rgb_frames=cfg["data"].get("rgb_frames", 16),
        resize=tuple(cfg["data"].get("resize", [224, 224])),
    )

    model = load_model(cfg)

    probs, labels = gather_probs(model, test_loader)

    ece = expected_calibration_error(probs, labels)
    print(f"\n========================")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(f"========================\n")


if __name__ == "__main__":
    main()
