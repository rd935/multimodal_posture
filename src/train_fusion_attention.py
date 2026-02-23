# src/train_fusion_attention.py

import sys
import json
import yaml
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, f1_score
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def contrastive_loss_rgb_depth(z_rgb: torch.Tensor, z_depth: torch.Tensor, temperature: float = 0.1):
    """
    InfoNCE-style contrastive loss between RGB and depth embeddings in a minibatch.
    """
    z_rgb = F.normalize(z_rgb, p=2, dim=-1)
    z_depth = F.normalize(z_depth, p=2, dim=-1)

    logits = torch.matmul(z_rgb, z_depth.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_rgb_to_depth = F.cross_entropy(logits, labels)
    loss_depth_to_rgb = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_rgb_to_depth + loss_depth_to_rgb)


def train_one_epoch(model, loader, optimizer, criterion, w_contrastive=0.0, temperature=0.1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        # We need embeddings if contrastive is on
        return_emb = bool(w_contrastive > 0.0)

        logits, extras = model(
            rgb,
            depth,
            return_embeddings=return_emb,
            return_attention=True,
        )
        loss_ce = criterion(logits, labels)

        # Entropy of modality attention: -(p log p) averaged
        attn_probs = extras["modality_attention"]  # (B, 2)
        attn_entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()

        # Small weight to push attention away from uniform 0.5/0.5
        loss = loss_ce + 0.001 * attn_entropy

        # Optional: Contrastive alignment (attention + contrastive ablation)
        if w_contrastive > 0.0:
            z_rgb = extras["z_rgb"]
            z_depth = extras["z_depth"]
            c_loss = contrastive_loss_rgb_depth(z_rgb, z_depth, temperature=temperature)
            loss = loss + w_contrastive * c_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_with_attention(model, loader, criterion, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

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

        modality_attention = extras["modality_attention"].detach().cpu().numpy()
        labels_np = labels.cpu().numpy()

        modality_att_sum_overall += modality_attention.sum(axis=0)
        overall_count += modality_attention.shape[0]

        for att_vec, lbl in zip(modality_attention, labels_np):
            modality_att_sum_per_class[lbl] += att_vec
            class_counts[lbl] += 1

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    mean_modality_attention = modality_att_sum_overall / overall_count if overall_count > 0 else np.zeros(2)
    mean_att_per_class = np.zeros_like(modality_att_sum_per_class)
    for c in range(num_classes):
        mean_att_per_class[c] = (
            modality_att_sum_per_class[c] / class_counts[c] if class_counts[c] > 0 else np.array([np.nan, np.nan])
        )

    return avg_loss, avg_acc, cm, all_preds, all_labels, mean_modality_attention, mean_att_per_class


def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_attention_heatmap(attention_per_class, class_names, out_path,
                           title="Mean Modality Attention per Class",
                           modality_labels=("RGB", "Depth")):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attention_per_class, cmap=plt.cm.Oranges, aspect="auto")

    ax.set_xticks(np.arange(len(modality_labels)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(modality_labels)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Modality")
    ax.set_ylabel("Class")
    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(attention_per_class.shape[0]):
        for j in range(attention_per_class.shape[1]):
            val = attention_per_class[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main():
    # -----------------------------------------------------
    # Load YAML config
    # -----------------------------------------------------
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "config" / "fusion_attention.yaml"
    print(f"[INFO] Using config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    loader_cfg = cfg.get("loader", {})
    log_cfg = cfg["logging"]
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    print(f"[INFO] Using seed: {seed}")

    epochs = int(train_cfg.get("epochs", 10))
    patience = int(train_cfg.get("patience", 3))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    batch_size = int(train_cfg.get("batch_size", 8))

    rgb_frames = int(data_cfg.get("rgb_frames", 16))
    resize = tuple(data_cfg.get("resize", [224, 224]))
    num_workers = int(loader_cfg.get("num_workers", 4))

    # Contrastive knobs (OFF by default)
    w_contrastive = float(loss_cfg.get("contrastive_weight", 0.0))
    contrastive_temperature = float(loss_cfg.get("contrastive_temperature", 0.1))

    ckpt_dir = PROJECT_ROOT / log_cfg.get("ckpt_dir", "checkpoints/fusion_attention_contrastive")
    results_dir = PROJECT_ROOT / log_cfg.get("results_dir", "results/fusion_attention_contrastive")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- Data loaders --------------------
    train_loader, val_loader, test_loader = make_utd_mhad_loaders(
        str(PROJECT_ROOT / data_cfg["train_csv"]),
        str(PROJECT_ROOT / data_cfg["val_csv"]),
        str(PROJECT_ROOT / data_cfg["test_csv"]),
        batch_size=batch_size,
        num_workers=num_workers,
        rgb_frames=rgb_frames,
        resize=resize,
        label_mode="stability3",
    )

    print(
        f"[DEBUG] Dataset sizes: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    # -------------------- Classes + class weights --------
    from collections import Counter

    num_classes = 3
    class_names = ["stable", "unstable", "falling"]

    train_ds = train_loader.dataset
    label_counts = Counter(int(train_ds[i]["label"]) for i in range(len(train_ds)))
    print("[INFO] Train label counts:", label_counts)

    counts = torch.tensor([label_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float, device=DEVICE)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    print("[INFO] Class weights:", weights)

    # -------------------- Model & Optimizer ---------------
    embed_dim = int(model_cfg.get("embed_dim", 256))
    fusion_hidden_dim = int(model_cfg.get("fusion_hidden_dim", 512))
    attn_hidden_dim = int(model_cfg.get("attn_hidden_dim", 256))
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
    normalize_embeddings = bool(model_cfg.get("normalize_embeddings", False))

    model = MultimodalRGBDAttentionFusion(
        num_classes=num_classes,
        embed_dim=embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        attn_hidden_dim=attn_hidden_dim,
        pretrained=pretrained,
        normalize_embeddings=normalize_embeddings,
        freeze_backbone=freeze_backbone,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # -------------------- Training loop -------------------
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            w_contrastive=w_contrastive,
            temperature=contrastive_temperature,
        )
        val_loss, val_acc, _, _, _, _, _ = evaluate_with_attention(model, val_loader, criterion, num_classes)

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / f"fusion_attention_contrastive_best_seed{seed}.pt")
            print(f"  [*] New best val_acc={val_acc:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # -------------------- Final test evaluation -----------
    best_ckpt = ckpt_dir / f"fusion_attention_contrastive_best_seed{seed}.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    test_loss, test_acc, test_cm, test_preds, test_labels, mean_att, mean_att_per_class = evaluate_with_attention(
        model, test_loader, criterion, num_classes
    )

    test_f1_macro = f1_score(test_labels, test_preds, average="macro")

    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}, macro_F1={test_f1_macro:.4f}")

    results = {
        "modality": "rgb+depth",
        "fusion_type": "attention_contrastive_fusion",
        "seed": seed,
        "num_classes": num_classes,
        "class_names": class_names,
        "config_path": str(config_path),
        "train_history": history,
        "best_val": {"epoch": best_epoch, "val_acc": best_val_acc},
        "test": {
            "loss": float(test_loss),
            "acc": float(test_acc),
            "f1_macro": float(test_f1_macro),
            "confusion_matrix": test_cm.tolist(),
            "mean_modality_attention": mean_att.tolist(),
            "mean_modality_attention_per_class": mean_att_per_class.tolist(),
        },
        "loss_hyperparams": {
            "contrastive_weight": w_contrastive,
            "contrastive_temperature": contrastive_temperature,
        },
    }

    json_path = results_dir / f"fusion_attention_contrastive_results_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved JSON results to {json_path}")

    cm_path = results_dir / f"fusion_attention_contrastive_confusion_matrix_seed{seed}.png"
    plot_confusion_matrix(test_cm, class_names, cm_path, title="Attention Fusion (RGB+Depth) Confusion Matrix")
    print(f"[INFO] Saved confusion matrix heatmap to {cm_path}")

    attn_heatmap_path = results_dir / f"fusion_attention_contrastive_heatmap_seed{seed}.png"
    plot_attention_heatmap(
        mean_att_per_class, class_names, attn_heatmap_path,
        title="Attention Fusion: Mean Modality Attention per Class",
        modality_labels=("RGB", "Depth"),
    )
    print(f"[INFO] Saved attention heatmap to {attn_heatmap_path}")


if __name__ == "__main__":
    main()