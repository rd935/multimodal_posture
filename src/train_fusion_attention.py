# src/train_fusion_attention.py

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
from models.multimodal_rgbd import MultimodalRGBDAttentionFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(rgb, depth)  # no need for attention outputs during training
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_with_attention(model, loader, criterion, num_classes):
    """
    Evaluate model, returning:
      - loss, acc, confusion matrix, preds, labels
      - mean modality attention overall and per class

    We assume model forward signature:
        logits, extras = model(..., return_attention=True)
        extras["modality_attention"]: (B, 2) [rgb_weight, depth_weight]
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    # For attention statistics
    modality_att_sum_overall = np.zeros(2, dtype=np.float64)
    overall_count = 0

    modality_att_sum_per_class = np.zeros((num_classes, 2), dtype=np.float64)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits, extras = model(
            rgb,
            depth,
            return_embeddings=False,
            return_attention=True,
        )

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        modality_attention = extras["modality_attention"].detach().cpu().numpy()  # (B, 2)
        labels_np = labels.cpu().numpy()

        # overall aggregate
        modality_att_sum_overall += modality_attention.sum(axis=0)
        overall_count += modality_attention.shape[0]

        # per-class aggregate
        for att_vec, lbl in zip(modality_attention, labels_np):
            modality_att_sum_per_class[lbl] += att_vec
            class_counts[lbl] += 1

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    # overall mean attention
    if overall_count > 0:
        mean_modality_attention = modality_att_sum_overall / overall_count
    else:
        mean_modality_attention = np.zeros(2, dtype=np.float64)

    # per-class mean attention (handle zero counts)
    mean_att_per_class = np.zeros_like(modality_att_sum_per_class)
    for c in range(num_classes):
        if class_counts[c] > 0:
            mean_att_per_class[c] = modality_att_sum_per_class[c] / class_counts[c]
        else:
            mean_att_per_class[c] = np.array([np.nan, np.nan])

    return (
        avg_loss,
        avg_acc,
        cm,
        all_preds,
        all_labels,
        mean_modality_attention,
        mean_att_per_class,
    )


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


def plot_attention_heatmap(
    mean_att_per_class,
    class_names,
    out_path,
    title="Mean Modality Attention per Class",
    modality_labels=("RGB", "Depth"),
):
    """
    mean_att_per_class: (num_classes, 2) array with [rgb_weight, depth_weight].
    """
    num_classes = len(class_names)

    plt.figure(figsize=(8, 6))
    plt.imshow(mean_att_per_class, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(2), modality_labels)
    plt.yticks(np.arange(num_classes), class_names)
    plt.xlabel("Modality")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # -----------------------------------------------------
    # Load YAML config
    # -----------------------------------------------------
    # Default config path: config/fusion_attention.yaml
    config_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "config" / "fusion_attention.yaml"
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

    ckpt_dir = PROJECT_ROOT / log_cfg.get("ckpt_dir", "checkpoints/fusion_attention")
    results_dir = PROJECT_ROOT / log_cfg.get("results_dir", "results/fusion_attention")
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
    attn_hidden_dim = int(model_cfg.get("attn_hidden_dim", 256))
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))

    model = MultimodalRGBDAttentionFusion(
        num_classes=num_classes,
        embed_dim=embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        attn_hidden_dim=attn_hidden_dim,
        pretrained=pretrained,
        normalize_embeddings=True,
        freeze_backbone=freeze_backbone,
    ).to(DEVICE)


    criterion = nn.CrossEntropyLoss()
    
    # Separate backbone vs "head" (projections + attention + classifier)
    backbone_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "rgb_backbone" in name or "depth_backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = Adam(
        [
            {"params": backbone_params, "lr": 1e-5, "weight_decay": 1e-5},   # very gentle FT
            {"params": head_params, "lr": 1e-4, "weight_decay": 5e-5},       # as before, slightly softer WD
        ]
    )


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
        val_loss, val_acc, _, _, _, _, _ = evaluate_with_attention(
            model, val_loader, criterion, num_classes
        )

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
            torch.save(model.state_dict(), ckpt_dir / "fusion_attention_best.pt")
            print(f"  [*] New best val_acc={val_acc:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # -------------------- Final test evaluation -----------    
    best_ckpt = ckpt_dir / "fusion_attention_best.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    (
        test_loss,
        test_acc,
        test_cm,
        test_preds,
        test_labels,
        mean_modality_attention,
        mean_att_per_class,
    ) = evaluate_with_attention(model, test_loader, criterion, num_classes)

    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")
    print("[TEST] Confusion matrix:\n", test_cm)
    print("[TEST] Mean modality attention (overall) [RGB, Depth]:", mean_modality_attention)

    # -------------------- Save JSON results ---------------
    results = {
        "modality": "rgb+depth",
        "fusion_type": "attention_baseline",  # strong baseline
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
            "mean_modality_attention": mean_modality_attention.tolist(),
            "mean_modality_attention_per_class": mean_att_per_class.tolist(),
        },
    }

    json_path = results_dir / "fusion_attention_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved JSON results to {json_path}")

    # -------------------- Save confusion matrix heatmap ---
    cm_path = results_dir / "fusion_attention_confusion_matrix.png"
    plot_confusion_matrix(
        test_cm,
        class_names,
        cm_path,
        title="Attention Fusion (RGB+Depth) Confusion Matrix",
    )
    print(f"[INFO] Saved confusion matrix heatmap to {cm_path}")

    # -------------------- Save attention heatmap ----------
    attn_heatmap_path = results_dir / "fusion_attention_heatmap.png"
    plot_attention_heatmap(
        mean_att_per_class,
        class_names,
        attn_heatmap_path,
        title="Attention Fusion: Mean Modality Attention per Class",
        modality_labels=("RGB", "Depth"),
    )
    print(f"[INFO] Saved attention heatmap to {attn_heatmap_path}")


if __name__ == "__main__":
    main()
