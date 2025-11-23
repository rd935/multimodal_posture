import sys
import json
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# Path setup so we can import datasets/ and models/
# ---------------------------------------------------------
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloaders import make_utd_mhad_loaders
from models.multimodal_rgbd import MultimodalRGBDAttnContrastiveUncertainty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def contrastive_loss_rgb_depth(
    proj_rgb: torch.Tensor,
    proj_depth: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Symmetric InfoNCE-style contrastive loss between RGB and Depth projections.

    proj_rgb:   (B, D)
    proj_depth: (B, D)
    """
    # proj_rgb/proj_depth should already be normalized by the model,
    # but we normalize again here for safety.
    proj_rgb = F.normalize(proj_rgb, dim=-1)
    proj_depth = F.normalize(proj_depth, dim=-1)

    logits = proj_rgb @ proj_depth.T / temperature  # (B, B)
    labels = torch.arange(proj_rgb.size(0), device=proj_rgb.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_i + loss_j)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    lambda_contrastive: float,
    lambda_var_reg: float,
    lambda_attn_entropy: float,
    epoch: int,
    contrastive_warmup: int = 5,
):
    """
    Train for one epoch.

    - Main objective: classification cross-entropy (with label smoothing via `criterion`)
    - Auxiliary: RGB-Depth contrastive alignment (InfoNCE)
    - Auxiliary: small regularization on uncertainty log-variances
    - Auxiliary: attention entropy regularizer (encourages sharper attention)
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_contrastive_loss = 0.0
    total_correct = 0
    total_samples = 0

    # only apply aux losses after a few epochs
    use_aux = epoch > contrastive_warmup

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        logits, extras = model(
            rgb,
            depth,
            return_embeddings=True,
            return_attention=True,
            return_uncertainty=True,
            return_projections=True,
        )

        # ----- (1) classification is the main objective -----
        cls_loss = criterion(logits, labels)

        # ----- (2) contrastive loss (auxiliary) -----
        proj_rgb = extras["proj_rgb"]
        proj_depth = extras["proj_depth"]
        contr_loss = contrastive_loss_rgb_depth(proj_rgb, proj_depth)

        # ----- (3) small regularizer on logvars (to keep them bounded) -----
        logvar_rgb = extras["logvar_rgb"].squeeze(-1)  # (B,)
        logvar_depth = extras["logvar_depth"].squeeze(-1)
        # push logvars gently toward 0 so they don’t explode
        var_reg = (logvar_rgb**2 + logvar_depth**2).mean()

        # ----- (4) attention entropy regularizer -----
        attn_probs = extras["modality_attention"]  # (B, 2)
        attn_entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()

        aux_scale = 1.0 if use_aux else 0.0

        # ----- total loss -----
        # NOTE: positive lambda_attn_entropy here; this matches the attention fusion baseline
        # behaviour (minimizing loss encourages lower entropy = sharper attention).
        loss = (
            cls_loss
            + aux_scale
            * (
                lambda_contrastive * contr_loss
                + lambda_var_reg * var_reg
                + lambda_attn_entropy * attn_entropy
            )
        )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_contrastive_loss += contr_loss.item() * batch_size

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "cls_loss": total_cls_loss / total_samples,
        "contrastive_loss": total_contrastive_loss / total_samples,
        "acc": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate_with_extras(model, loader, num_classes):
    """
    Evaluate model for classification metrics, attention, and uncertainty stats.

    Uses plain CE (no uncertainty weighting) for reporting loss.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    # attention stats
    modality_att_sum_overall = np.zeros(2, dtype=np.float64)
    overall_count = 0

    modality_att_sum_per_class = np.zeros((num_classes, 2), dtype=np.float64)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    # uncertainty stats
    logvar_rgb_list = []
    logvar_depth_list = []

    criterion = nn.CrossEntropyLoss(reduction="sum")

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        logits, extras = model(
            rgb,
            depth,
            return_embeddings=True,
            return_attention=True,
            return_uncertainty=True,
            return_projections=False,
        )

        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item()
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        modality_attention = extras["modality_attention"].detach().cpu().numpy()  # (B, 2)
        labels_np = labels.cpu().numpy()

        modality_att_sum_overall += modality_attention.sum(axis=0)
        overall_count += modality_attention.shape[0]

        for c in range(num_classes):
            mask_c = labels_np == c
            if mask_c.any():
                modality_att_sum_per_class[c] += modality_attention[mask_c].sum(axis=0)
                class_counts[c] += mask_c.sum()

        logvar_rgb = extras["logvar_rgb"].detach().cpu().numpy().reshape(-1)
        logvar_depth = extras["logvar_depth"].detach().cpu().numpy().reshape(-1)
        logvar_rgb_list.append(logvar_rgb)
        logvar_depth_list.append(logvar_depth)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    # attention aggregates
    mean_modality_attention_overall = modality_att_sum_overall / max(overall_count, 1)
    mean_modality_attention_per_class = modality_att_sum_per_class / np.maximum(
        class_counts[:, None], 1
    )

    # uncertainty aggregates
    logvar_rgb_all = np.concatenate(logvar_rgb_list, axis=0)
    logvar_depth_all = np.concatenate(logvar_depth_list, axis=0)
    mean_logvar_rgb = float(logvar_rgb_all.mean())
    mean_logvar_depth = float(logvar_depth_all.mean())

    return (
        avg_loss,
        avg_acc,
        cm,
        all_preds,
        all_labels,
        mean_modality_attention_overall,
        mean_modality_attention_per_class,
        mean_logvar_rgb,
        mean_logvar_depth,
    )


def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
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
    fig.savefig(out_path)
    plt.close(fig)


def plot_attention_heatmap(mean_att_per_class, class_names, out_path, title, modality_labels=("RGB", "Depth")):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mean_att_per_class, aspect="auto", cmap=plt.cm.Oranges)

    ax.figure.colorbar(im, ax=ax)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xticks(np.arange(len(modality_labels)))
    ax.set_xticklabels(modality_labels)
    ax.set_xlabel("Modality")
    ax.set_ylabel("Class")
    ax.set_title(title)

    for i in range(mean_att_per_class.shape[0]):
        for j in range(mean_att_per_class.shape[1]):
            ax.text(
                j,
                i,
                f"{mean_att_per_class[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    set_seed(42)
    # -----------------------------------------------------
    # Load YAML config
    # -----------------------------------------------------
    # Default config path: config/fusion_core.yaml
    config_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "config" / "fusion_core.yaml"
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

    # ↓↓↓ ONLY CHANGE HERE: gentler default weights for aux losses ↓↓↓
    lambda_contrastive = float(train_cfg.get("lambda_contrastive", 0.0005))
    lambda_var_reg = float(train_cfg.get("lambda_uncertainty_reg", 0.0001))
    lambda_attn_entropy = float(train_cfg.get("lambda_attn_entropy", 0.01))
    # ↑↑↑ everything else unchanged ↑↑↑

    batch_size = int(train_cfg.get("batch_size", 8))
    rgb_frames = int(data_cfg.get("rgb_frames", 16))
    resize = tuple(data_cfg.get("resize", [224, 224]))
    num_workers = int(loader_cfg.get("num_workers", 4))

    ckpt_dir = PROJECT_ROOT / log_cfg.get("ckpt_dir", "checkpoints/fusion_core")
    results_dir = PROJECT_ROOT / log_cfg.get("results_dir", "results/fusion_core")
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
    proj_dim = int(model_cfg.get("proj_dim", 128))
    pretrained = bool(model_cfg.get("pretrained", True))
    normalize_embeddings = bool(model_cfg.get("normalize_embeddings", False))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))

    model = MultimodalRGBDAttnContrastiveUncertainty(
        num_classes=num_classes,
        embed_dim=embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        attn_hidden_dim=attn_hidden_dim,
        proj_dim=proj_dim,
        pretrained=pretrained,
        normalize_embeddings=normalize_embeddings,
        freeze_backbone=freeze_backbone,
    ).to(DEVICE)

    # same as attention fusion: label-smoothed CE
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- SIMPLE OPTIMIZER LIKE ATTENTION BASELINE ---
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_cls_loss": [],
        "train_contrastive_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # -------------------- Training loop -------------------
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion=criterion,
            lambda_contrastive=lambda_contrastive,
            lambda_var_reg=lambda_var_reg,
            lambda_attn_entropy=lambda_attn_entropy,
            epoch=epoch,
        )

        (
            val_loss,
            val_acc,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = evaluate_with_extras(model, val_loader, num_classes)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["train_cls_loss"].append(train_stats["cls_loss"])
        history["train_contrastive_loss"].append(train_stats["contrastive_loss"])
        history["train_acc"].append(train_stats["acc"])
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_stats['loss']:.4f}, "
            f"train_cls={train_stats['cls_loss']:.4f}, "
            f"train_contrast={train_stats['contrastive_loss']:.4f}, "
            f"train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # checkpoint + early stopping on **val_acc**
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "fusion_core_best.pt")
            print(f"  [*] New best val_acc={val_acc:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # -------------------- Final test evaluation -----------
    best_ckpt = ckpt_dir / "fusion_core_best.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    (
        test_loss,
        test_acc,
        test_cm,
        test_preds,
        test_labels,
        mean_modality_attention,
        mean_att_per_class,
        mean_logvar_rgb,
        mean_logvar_depth,
    ) = evaluate_with_extras(model, test_loader, num_classes)

    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")
    print("[TEST] Confusion matrix:\n", test_cm)
    print("[TEST] Mean modality attention (overall) [RGB, Depth]:", mean_modality_attention)
    print("[TEST] Mean log-variance RGB:", mean_logvar_rgb)
    print("[TEST] Mean log-variance Depth:", mean_logvar_depth)

    # -------------------- Save JSON results ---------------
    results = {
        "modality": "rgb+depth",
        "fusion_type": "core_attention_contrastive_uncertainty",
        "num_classes": num_classes,
        "class_names": class_names,
        "config_path": str(config_path),
        "train_history": history,
        "best_val": {
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
        },
        "test": {
            "loss": float(test_loss),
            "acc": float(test_acc),
            "confusion_matrix": test_cm.tolist(),
            "mean_modality_attention": mean_modality_attention.tolist(),
            "mean_modality_attention_per_class": mean_att_per_class.tolist(),
            "mean_logvar_rgb": mean_logvar_rgb,
            "mean_logvar_depth": mean_logvar_depth,
        },
    }

    json_path = results_dir / "fusion_core_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved JSON results to {json_path}")

    # -------------------- Save confusion matrix heatmap ---
    cm_path = results_dir / "fusion_core_confusion_matrix.png"
    plot_confusion_matrix(
        test_cm,
        class_names,
        cm_path,
        title="Core Model (RGB+Depth) Confusion Matrix",
    )
    print(f"[INFO] Saved confusion matrix heatmap to {cm_path}")

    # -------------------- Save attention heatmap ----------
    attn_heatmap_path = results_dir / "fusion_core_attention_heatmap.png"
    plot_attention_heatmap(
        mean_att_per_class,
        class_names,
        attn_heatmap_path,
        title="Core Model: Mean Modality Attention per Class",
        modality_labels=("RGB", "Depth"),
    )
    print(f"[INFO] Saved attention heatmap to {attn_heatmap_path}")


if __name__ == "__main__":
    main()
