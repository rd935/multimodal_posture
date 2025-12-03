# src/eval_missing_modalities_core.py

import sys
import yaml
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

# ----------------- project root / imports -----------------
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloaders import make_utd_mhad_loaders
from models.multimodal_rgbd import MultimodalRGBDCoreFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(default_path: str = "config/fusion_core.yaml"):
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / default_path
    print(f"[INFO] Using config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg, config_path


def build_loaders(cfg):
    data_cfg = cfg["data"]
    loader_cfg = cfg.get("loader", {})

    train_csv = PROJECT_ROOT / data_cfg["train_csv"]
    val_csv = PROJECT_ROOT / data_cfg["val_csv"]
    test_csv = PROJECT_ROOT / data_cfg["test_csv"]

    batch_size = int(cfg["train"].get("batch_size", 8))
    rgb_frames = int(data_cfg.get("rgb_frames", 16))
    resize = tuple(data_cfg.get("resize", [224, 224]))
    num_workers = int(loader_cfg.get("num_workers", 4))

    train_loader, val_loader, test_loader = make_utd_mhad_loaders(
        str(train_csv),
        str(val_csv),
        str(test_csv),
        batch_size=batch_size,
        num_workers=num_workers,
        rgb_frames=rgb_frames,
        resize=resize,
    )

    print(
        f"[INFO] Dataset sizes: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )
    return train_loader, val_loader, test_loader


def build_core_model(cfg):
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    num_classes = 3
    class_names = ["stable", "unstable", "falling"]

    embed_dim = int(model_cfg.get("embed_dim", 256))
    fusion_hidden_dim = int(model_cfg.get("fusion_hidden_dim", 512))
    attn_hidden_dim = int(model_cfg.get("attn_hidden_dim", 256))
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
    normalize_embeddings = bool(model_cfg.get("normalize_embeddings", False))
    contrastive_temperature = float(loss_cfg.get("contrastive_temperature", 0.1))

    model = MultimodalRGBDCoreFusion(
        num_classes=num_classes,
        embed_dim=embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        attn_hidden_dim=attn_hidden_dim,
        pretrained=pretrained,
        normalize_embeddings=normalize_embeddings,
        freeze_backbone=freeze_backbone,
        contrastive_temperature=contrastive_temperature,
    ).to(DEVICE)

    return model, num_classes, class_names


def load_core_checkpoint(model, cfg):
    log_cfg = cfg["logging"]
    ckpt_dir = PROJECT_ROOT / log_cfg.get("ckpt_dir", "checkpoints/fusion_core")
    ckpt_ft = ckpt_dir / "fusion_core_best_ft.pt"
    ckpt_main = ckpt_dir / "fusion_core_best.pt"

    if ckpt_ft.exists():
        ckpt_path = ckpt_ft
        print(f"[INFO] Loading fine-tuned checkpoint: {ckpt_path}")
    elif ckpt_main.exists():
        ckpt_path = ckpt_main
        print(f"[INFO] Loading main checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(
            f"No fusion_core_best_ft.pt or fusion_core_best.pt found in {ckpt_dir}"
        )

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def eval_scenario(model, data_loader, num_classes: int, scenario: str):
    """
    scenario in {"full", "rgb_missing", "depth_missing"}
    """
    model.eval()
    all_preds = []
    all_labels = []

    for batch in data_loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        if scenario == "rgb_missing":
            rgb = torch.zeros_like(rgb)
        elif scenario == "depth_missing":
            depth = torch.zeros_like(depth)
        # "full" → leave both as-is

        # No extras requested → model returns ONLY logits
        logits = model(
            rgb,
            depth,
            return_embeddings=False,
            return_attention=False,
            return_uncertainty=False,
        )

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    acc = (all_preds == all_labels).mean()

    return acc, cm, all_preds, all_labels



def main():
    cfg, config_path = load_config()
    _ = int(cfg["train"].get("seed", 42))  # we don't need to force a new seed here

    _, _, test_loader = build_loaders(cfg)
    model, num_classes, class_names = build_core_model(cfg)
    model = load_core_checkpoint(model, cfg)

    scenarios = ["full", "rgb_missing", "depth_missing"]
    results = {}

    for scenario in scenarios:
        print(f"\n[INFO] Evaluating scenario: {scenario}")
        acc, cm, preds, labels = eval_scenario(model, test_loader, num_classes, scenario)
        print(f"[{scenario}] test_acc = {acc:.4f}")
        print(f"[{scenario}] confusion matrix:\n{cm}")
        results[scenario] = {
            "acc": float(acc),
            "confusion_matrix": cm.tolist(),
        }

    # optionally save a JSON summary
    log_cfg = cfg["logging"]
    results_dir = PROJECT_ROOT / log_cfg.get("results_dir", "results/fusion_core")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "fusion_core_missing_modality_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "config_path": str(config_path),
                "results": results,
                "class_names": class_names,
            },
            f,
            indent=2,
        )
    print(f"\n[INFO] Saved missing-modality results to {out_json}")


if __name__ == "__main__":
    main()
