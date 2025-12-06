import sys
import yaml
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt

# ----------------- project root / imports -----------------
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloaders import make_utd_mhad_loaders
from models.multimodal_rgbd import (
    MultimodalRGBDCoreFusion,
    MultimodalRGBDAttentionFusion,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
#   Checkpoint loading (same policy as compute_ece.py)
# =========================================================
def load_checkpoint(model, model_name: str):
    ckpt_root = PROJECT_ROOT / "checkpoints"

    if model_name == "core":
        ckpt_dir = ckpt_root / "fusion_core"
        ckpt_main = ckpt_dir / "fusion_core_best.pt"
        ckpt_ft = ckpt_dir / "fusion_core_best_ft.pt"

        if ckpt_main.exists():
            ckpt_path = ckpt_main
            print(f"[INFO] Loading CORE main checkpoint: {ckpt_path}")
        elif ckpt_ft.exists():
            ckpt_path = ckpt_ft
            print(f"[INFO] Loading CORE fine-tuned checkpoint (legacy): {ckpt_path}")
        else:
            raise FileNotFoundError(
                "No core fusion checkpoint found in checkpoints/fusion_core/ "
                "(expected fusion_core_best.pt or fusion_core_best_ft.pt)"
            )

    elif model_name == "attention":
        ckpt_dir = ckpt_root / "fusion_attention"
        ckpt_path = ckpt_dir / "fusion_attention_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                "Expected attention checkpoint "
                "checkpoints/fusion_attention/fusion_attention_best.pt not found."
            )
        print(f"[INFO] Loading ATTENTION checkpoint: {ckpt_path}")

    else:
        raise ValueError(f"Invalid model_name: {model_name}")

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# =========================================================
#   Build dataloaders & models
# =========================================================
def build_loaders(cfg):
    data_cfg = cfg["data"]
    loader_cfg = cfg.get("loader", {})
    train_cfg = cfg.get("train", {})

    train_csv = PROJECT_ROOT / data_cfg["train_csv"]
    val_csv = PROJECT_ROOT / data_cfg["val_csv"]
    test_csv = PROJECT_ROOT / data_cfg["test_csv"]

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(loader_cfg.get("num_workers", 4))
    rgb_frames = int(data_cfg.get("rgb_frames", 16))
    resize = tuple(data_cfg.get("resize", [224, 224]))

    train_loader, val_loader, test_loader = make_utd_mhad_loaders(
        str(train_csv),
        str(val_csv),
        str(test_csv),
        batch_size=batch_size,
        num_workers=num_workers,
        rgb_frames=rgb_frames,
        resize=resize,
    )
    return train_loader, val_loader, test_loader


def build_model(cfg, model_name: str, num_classes: int = 3):
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    embed_dim = int(model_cfg.get("embed_dim", 256))
    fusion_hidden_dim = int(model_cfg.get("fusion_hidden_dim", 512))
    attn_hidden_dim = int(model_cfg.get("attn_hidden_dim", 256))
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))
    normalize_embeddings = bool(model_cfg.get("normalize_embeddings", True))
    contrastive_temperature = float(loss_cfg.get("contrastive_temperature", 0.1))

    if model_name == "core":
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
    elif model_name == "attention":
        model = MultimodalRGBDAttentionFusion(
            num_classes=num_classes,
            embed_dim=embed_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            attn_hidden_dim=attn_hidden_dim,
            pretrained=pretrained,
            normalize_embeddings=normalize_embeddings,
            freeze_backbone=freeze_backbone,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model = load_checkpoint(model, model_name)
    return model


# =========================================================
#   Scenario evaluation
# =========================================================
@torch.no_grad()
def eval_scenario(model, data_loader, num_classes: int, scenario: str):
    """
    scenario in {"full", "rgb_missing", "depth_missing"}.
    - "rgb_missing": zero out RGB
    - "depth_missing": zero out depth
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
        elif scenario == "full":
            pass
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # For core fusion, we can safely request uncertainty outputs.
        # This does NOT change logits; it just populates extras.
        if isinstance(model, MultimodalRGBDCoreFusion):
            out = model(
                rgb,
                depth,
                return_embeddings=False,
                return_attention=False,
                return_uncertainty=True,
            )
        else:
            # Attention baseline: no uncertainty flag
            out = model(
                rgb,
                depth,
                return_embeddings=False,
                return_attention=False,
            )

        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    acc = float(accuracy_score(all_labels, all_preds))
    f1_macro = float(f1_score(all_labels, all_preds, average="macro"))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(len(all_labels)),
    }


def plot_missing_modality_bar(results, out_path: Path):
    """
    results: dict like:
        {
          "attention": {"full": {...}, "rgb_missing": {...}, "depth_missing": {...}},
          "core":      {"full": {...}, "rgb_missing": {...}, "depth_missing": {...}}
        }
    """
    models = ["attention", "core"]
    scenarios = ["full", "rgb_missing", "depth_missing"]
    x_labels = ["Full", "RGB missing", "Depth missing"]

    x = np.arange(len(scenarios))
    width = 0.35

    acc_att = [results["attention"][s]["acc"] for s in scenarios]
    acc_core = [results["core"][s]["acc"] for s in scenarios]

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, acc_att, width, label="Attention")
    plt.bar(x + width / 2, acc_core, width, label="Core")

    plt.xticks(x, x_labels)
    plt.ylabel("Accuracy")
    plt.title("Missing Modality Robustness")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# =========================================================
#   Main
# =========================================================
def main():
    # Use core config for dataset paths etc.
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = PROJECT_ROOT / "config" / "fusion_core.yaml"

    print(f"[INFO] Using config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _, _, test_loader = build_loaders(cfg)
    num_classes = 3
    class_names = ["stable", "unstable", "falling"]

    scenarios = ["full", "rgb_missing", "depth_missing"]

    all_results = {}
    for model_name in ["attention", "core"]:
        print(f"\n=================================")
        print(f"[INFO] Evaluating model: {model_name}")
        print("=================================")

        model = build_model(cfg, model_name, num_classes=num_classes)
        model_results = {}

        for scenario in scenarios:
            print(f"\n[SCENARIO] {model_name} - {scenario}")
            metrics = eval_scenario(model, test_loader, num_classes, scenario)
            print(
                f"  acc={metrics['acc']:.4f}, "
                f"macro_F1={metrics['f1_macro']:.4f}"
            )
            model_results[scenario] = metrics

        all_results[model_name] = model_results

    # Save JSON
    out_dir = PROJECT_ROOT / "results" / "missing_modalities"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "missing_modality_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "config_path": str(config_path),
                "class_names": class_names,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n[INFO] Saved missing-modality results to {out_json}")

    # Plot bar chart comparison
    plot_path = out_dir / "missing_modality_accuracy.png"
    plot_missing_modality_bar(all_results, plot_path)
    print(f"[INFO] Saved missing-modality bar plot to {plot_path}")


if __name__ == "__main__":
    main()
