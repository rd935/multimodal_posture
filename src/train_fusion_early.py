# src/train_fusion_early.py

import sys
import json
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Path setup so we can import datasets/ and models/
# ---------------------------------------------------------
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloaders import make_utd_mhad_loaders
from models.multimodal_rgbd import MultimodalRGBDEarlyFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE)         # (B, T, 3, H, W)
        depth = batch["depth"].to(DEVICE)     # (B, T, 1, H, W)
        labels = batch["label"].to(DEVICE)    # (B,)

        optimizer.zero_grad()
        logits = model(rgb, depth)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits = model(rgb, depth)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, avg_acc, cm, all_preds, all_labels


def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # -----------------------------------------------------
    # Load YAML config
    # -----------------------------------------------------
    # Default config path: config/fusion_early.yaml
    config_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "config" / "fusion_early.yaml"
    )
    print(f"[INFO] Using config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # -------------------- Paths --------------------------
    data_cfg = cfg["data"]
    train_csv = PROJECT_ROOT / data_cfg["train_csv"]
    val_csv = PROJECT_ROOT / data_cfg["val_csv"]
    test_csv = PROJECT_ROOT / data_cfg["test_csv"]

    train_cfg = cfg["train"]
    loader_cfg = cfg.get("loader", {})
    log_cfg = cfg["logging"]
    model_cfg = cfg.get("model", {})

    epochs = int(train_cfg.get("epochs", 10))
    patience = int(train_cfg.get("patience", 3))
    lr = float(train_cfg.get("learning_rate", 1e-4))

    batch_size = int(train_cfg.get("batch_size", 8))
    rgb_frames = int(data_cfg.get("rgb_frames", 16))
    resize = tuple(data_cfg.get("resize", [224, 224]))
    num_workers = int(loader_cfg.get("num_workers", 4))

    ckpt_dir = PROJECT_ROOT / log_cfg.get("ckpt_dir", "checkpoints/fusion_early")
    results_dir = PROJECT_ROOT / log_cfg.get("results_dir", "results/fusion_early")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- Data loaders --------------------
    train_loader, val_loader, test_loader = make_utd_mhad_loaders(
        str(train_csv),
        str(val_csv),
        str(test_csv),
        batch_size=batch_size,
        num_workers=num_workers,
        rgb_frames=rgb_frames,
        resize=resize,
    )

    # infer num_classes from train dataset
    train_ds = train_loader.dataset
    labels = [int(row["label"]) for row in train_ds.items]
    num_classes = len(set(labels))
    class_names = list(range(num_classes))

    # -------------------- Model & Optimizer ---------------
    embed_dim = int(model_cfg.get("embed_dim", 256))
    fusion_hidden_dim = int(model_cfg.get("fusion_hidden_dim", 512))
    pretrained = bool(model_cfg.get("pretrained", True))

    model = MultimodalRGBDEarlyFusion(
        num_classes=num_classes,
        embed_dim=embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        pretrained=pretrained,
        normalize_embeddings=True,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # -------------------- Training loop -------------------
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # checkpoint + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "fusion_early_best.pt")
            print(f"  [*] New best val_acc={val_acc:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # -------------------- Final test evaluation -----------    
    best_ckpt = ckpt_dir / "fusion_early_best.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    test_loss, test_acc, test_cm, test_preds, test_labels = evaluate(model, test_loader, criterion)

    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")
    print("[TEST] Confusion matrix:\n", test_cm)

    # -------------------- Save JSON results ---------------
    results = {
        "modality": "rgb+depth",
        "fusion_type": "early_concat",
        "num_classes": num_classes,
        "class_names": class_names,
        "config_path": str(config_path),
        "train_history": history,
        "best_val": {
            "epoch": best_epoch,
            "val_acc": best_val_acc,
        },
        "test": {
            "loss": float(test_loss),
            "acc": float(test_acc),
            "confusion_matrix": test_cm.tolist(),
        },
    }

    json_path = results_dir / "fusion_early_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved JSON results to {json_path}")

    # -------------------- Save confusion matrix heatmap ---
    cm_path = results_dir / "fusion_early_confusion_matrix.png"
    plot_confusion_matrix(
        test_cm,
        class_names,
        cm_path,
        title="Early Fusion (RGB+Depth) Confusion Matrix",
    )
    print(f"[INFO] Saved confusion matrix heatmap to {cm_path}")


if __name__ == "__main__":
    main()
