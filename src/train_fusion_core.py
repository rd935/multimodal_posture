import sys
import json
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import confusion_matrix, f1_score
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
from models.multimodal_rgbd import MultimodalRGBDCoreFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Loss components
# ---------------------------------------------------------

def contrastive_loss_rgb_depth(
    z_rgb: torch.Tensor,
    z_depth: torch.Tensor,
    temperature: float = 0.1,
):
    """
    Simple InfoNCE-style contrastive loss between RGB and depth embeddings.
    """
    z_rgb = F.normalize(z_rgb, p=2, dim=-1)
    z_depth = F.normalize(z_depth, p=2, dim=-1)

    logits = torch.matmul(z_rgb, z_depth.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_rgb_to_depth = F.cross_entropy(logits, labels)
    loss_depth_to_rgb = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_rgb_to_depth + loss_depth_to_rgb)


def attention_entropy_loss(modality_attention: torch.Tensor):
    """
    Entropy of the modality attention distribution.
    modality_attention: (B, 2) with probs for [RGB, Depth]
    """
    entropy = -(modality_attention * torch.log(modality_attention + 1e-8)).sum(dim=-1).mean()
    return entropy


# ---------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------

def train_one_epoch_core(
    model,
    train_loader,
    optimizer,
    base_criterion,
    temperature: float,
    w_contrastive: float,
    w_attn_entropy: float,
):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_contrast = 0.0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        # We need embeddings + attention, no uncertainty yet
        logits, extras = model(
            rgb,
            depth,
            return_embeddings=True,
            return_attention=True,
            return_uncertainty=False,
        )

        ce_per_sample = base_criterion(logits, labels)  # (B,)
        cls_loss = ce_per_sample.mean()

        loss = cls_loss
        c_loss = torch.tensor(0.0, device=DEVICE)
        ent = torch.tensor(0.0, device=DEVICE)

        # contrastive term
        if w_contrastive > 0 and "z_rgb" in extras and "z_depth" in extras:
            z_rgb = extras["z_rgb"]
            z_depth = extras["z_depth"]
            c_loss = contrastive_loss_rgb_depth(z_rgb, z_depth, temperature=temperature)
            loss = loss + w_contrastive * c_loss

        # attention entropy term
        if w_attn_entropy > 0 and "modality_attention" in extras:
            modality_attention = extras["modality_attention"]  # (B, 2)
            ent = attention_entropy_loss(modality_attention)
            loss = loss + w_attn_entropy * ent

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_cls += cls_loss.item() * bs
        total_contrast += c_loss.item() * bs
        total_correct += (preds == labels).sum().item()
        total_samples += bs

    epoch_loss = total_loss / total_samples
    epoch_cls = total_cls / total_samples
    epoch_contrast = total_contrast / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_cls, epoch_contrast, epoch_acc


def evaluate_with_attention_core(
    model,
    data_loader,
    criterion,
    num_classes: int,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_mod_att = []
    all_labels_for_att = []

    with torch.no_grad():
        for batch in data_loader:
            rgb = batch["rgb"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits, extras = model(
                rgb,
                depth,
                return_embeddings=False,
                return_attention=True,
                return_uncertainty=False,
            )

            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            bs = labels.size(0)

            total_loss += loss.item() * bs
            total_correct += (preds == labels).sum().item()
            total_samples += bs

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if "modality_attention" in extras:
                modality_attention = extras["modality_attention"].detach().cpu().numpy()
                all_mod_att.append(modality_attention)
                all_labels_for_att.append(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    if len(all_mod_att) > 0:
        all_mod_att = np.concatenate(all_mod_att, axis=0)           # (N, 2)
        all_labels_for_att = np.concatenate(all_labels_for_att, 0)  # (N,)
        mean_modality_attention = all_mod_att.mean(axis=0)

        mean_att_per_class = np.zeros((num_classes, 2), dtype=np.float32)
        for c in range(num_classes):
            idxs = (all_labels_for_att == c)
            if idxs.sum() > 0:
                mean_att_per_class[c] = all_mod_att[idxs].mean(axis=0)
            else:
                mean_att_per_class[c] = np.array([np.nan, np.nan])
    else:
        mean_modality_attention = np.array([np.nan, np.nan], dtype=np.float32)
        mean_att_per_class = np.full((num_classes, 2), np.nan, dtype=np.float32)

    return avg_loss, avg_acc, cm, all_preds, all_labels, mean_modality_attention, mean_att_per_class


def finetune_uncertainty(
    model,
    train_loader,
    val_loader,
    base_criterion,
    eval_criterion,
    contrastive_temperature: float,
    w_uncertainty_reg: float,
    num_classes: int,
    finetune_epochs: int = 15,
    finetune_lr: float = 5e-5,
    patience: int = 5,
):
    """
    Stage-2: fine-tune classifier + uncertainty heads with an uncertainty-weighted CE.
    """
    print("[INFO] Starting uncertainty fine-tuning stage...")

    # freeze everything
    for _, param in model.named_parameters():
        param.requires_grad = False

    # unfreeze classifier + uncertainty heads
    for name, param in model.named_parameters():
        if "core_classifier" in name or "rgb_var_head" in name or "depth_var_head" in name:
            param.requires_grad = True

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_lr,
        weight_decay=1e-4,
    )

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_unc = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            rgb = batch["rgb"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            logits, extras = model(
                rgb,
                depth,
                return_embeddings=True,
                return_attention=False,
                return_uncertainty=True,
            )

            ce_per_sample = base_criterion(logits, labels)  # (B,)

            log_var_rgb = extras["log_var_rgb"]
            log_var_depth = extras["log_var_depth"]

            var_rgb = log_var_rgb.exp()
            var_depth = log_var_depth.exp()

            inv_var_rgb = torch.exp(-log_var_rgb)
            inv_var_depth = torch.exp(-log_var_depth)

            rel_eff = (inv_var_rgb + inv_var_depth) / 2.0
            rel_eff = rel_eff / (rel_eff.mean().detach() + 1e-6)

            cls_loss = (rel_eff * ce_per_sample).mean()

            unc_reg = 0.5 * (var_rgb + var_depth).mean()

            loss = cls_loss + w_uncertainty_reg * unc_reg

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_cls += cls_loss.item() * bs
            total_unc += unc_reg.item() * bs
            total_correct += (preds == labels).sum().item()
            total_samples += bs

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        val_loss, val_acc, *_ = evaluate_with_attention_core(
            model,
            val_loader,
            eval_criterion,
            num_classes,
        )

        print(
            f"[FT] Epoch {epoch:02d} | train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt_ft = PROJECT_ROOT / "checkpoints" / "fusion_core" / "fusion_core_best_ft.pt"
            ckpt_ft.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_ft)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[FT] Early stopping fine-tune at epoch {epoch} (best epoch {best_epoch})")
                break

    ckpt_ft = PROJECT_ROOT / "checkpoints" / "fusion_core" / "fusion_core_best_ft.pt"
    if ckpt_ft.exists():
        model.load_state_dict(torch.load(ckpt_ft, map_location=DEVICE))
    else:
        print("[FT] Warning: fine-tune best checkpoint not found; using last epoch weights.")

    return best_val_acc


# ---------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------

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


def plot_attention_heatmap(
    attention_per_class,
    class_names,
    out_path,
    title="Mean Modality Attention per Class",
    modality_labels=("RGB", "Depth"),
):
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    # Load YAML config
    config_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "config" / "fusion_core.yaml"
    )
    print(f"[INFO] Using config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_csv = PROJECT_ROOT / data_cfg["train_csv"]
    val_csv = PROJECT_ROOT / data_cfg["val_csv"]
    test_csv = PROJECT_ROOT / data_cfg["test_csv"]

    train_cfg = cfg["train"]
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

    ckpt_dir = PROJECT_ROOT / log_cfg.get("ckpt_dir", "checkpoints/fusion_core")
    results_dir = PROJECT_ROOT / log_cfg.get("results_dir", "results/fusion_core")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    contrastive_temperature = float(loss_cfg.get("contrastive_temperature", 0.1))
    w_contrastive = float(loss_cfg.get("contrastive_weight", 0.1))
    w_uncertainty_reg = float(loss_cfg.get("uncertainty_reg_weight", 0.01))
    w_attn_entropy = float(loss_cfg.get("attn_entropy_weight", 0.01))

    # Data loaders
    train_loader, val_loader, test_loader = make_utd_mhad_loaders(
        str(train_csv),
        str(val_csv),
        str(test_csv),
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

    from collections import Counter

    num_classes = 3
    class_names = ["stable", "unstable", "falling"]

    train_ds = train_loader.dataset
    label_counts = Counter(int(train_ds[i]["label"]) for i in range(len(train_ds)))
    print("[INFO] Core fusion train label counts:", label_counts)

    counts = torch.tensor(
        [label_counts.get(i, 1) for i in range(num_classes)],
        dtype=torch.float,
        device=DEVICE,
    )
    weights = 1.0 / counts
    weights = weights / weights.sum()
    print("[INFO] Core fusion class weights:", weights)

    embed_dim = int(model_cfg.get("embed_dim", 256))
    fusion_hidden_dim = int(model_cfg.get("fusion_hidden_dim", 512))
    attn_hidden_dim = int(model_cfg.get("attn_hidden_dim", 256))
    pretrained = bool(model_cfg.get("pretrained", True))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
    normalize_embeddings = bool(model_cfg.get("normalize_embeddings", False))

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

    label_smoothing = float(train_cfg.get("label_smoothing", 0))

    base_criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    eval_criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=label_smoothing,
    )

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "train_cls": [],
        "train_contrast": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_cls_loss, train_contrast_loss, train_acc = train_one_epoch_core(
            model,
            train_loader,
            optimizer,
            base_criterion,
            temperature=contrastive_temperature,
            w_contrastive=w_contrastive,
            w_attn_entropy=w_attn_entropy,
        )

        val_loss, val_acc, _, _, _, _, _ = evaluate_with_attention_core(
            model, val_loader, eval_criterion, num_classes
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_cls"].append(train_cls_loss)
        history["train_contrast"].append(train_contrast_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_cls={train_cls_loss:.4f}, "
            f"train_contrast={train_contrast_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "fusion_core_best.pt")
            print(f"  [*] New best val_acc={val_acc:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    best_ckpt = ckpt_dir / "fusion_core_best.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))

    print("[INFO] Starting Stage-2 uncertainty fine-tune...")
    ft_best_val = finetune_uncertainty(
        model,
        train_loader,
        val_loader,
        base_criterion,
        eval_criterion,
        contrastive_temperature,
        w_uncertainty_reg=w_uncertainty_reg,
        num_classes=num_classes,
        finetune_epochs=8,
        finetune_lr=5e-5,
        patience=3,
    )

    print(f"[INFO] Fine-tune best val_acc={ft_best_val:.4f}")

    (
        test_loss,
        test_acc,
        test_cm,
        test_preds,
        test_labels,
        mean_modality_attention,
        mean_att_per_class,
    ) = evaluate_with_attention_core(model, test_loader, eval_criterion, num_classes)

    # ---- F1 score (macro) ----
    test_f1_macro = f1_score(test_labels, test_preds, average="macro")

    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}, macro_F1={test_f1_macro:.4f}")
    print("[TEST] Confusion matrix:\n", test_cm)
    print("[TEST] Mean modality attention (overall) [RGB, Depth]:", mean_modality_attention)

    results = {
        "modality": "rgb+depth",
        "fusion_type": "core_fusion",
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
            "f1_macro": float(test_f1_macro),
            "confusion_matrix": test_cm.tolist(),
            "mean_modality_attention": mean_modality_attention.tolist(),
            "mean_modality_attention_per_class": mean_att_per_class.tolist(),
        },
        "loss_hyperparams": {
            "contrastive_temperature": contrastive_temperature,
            "contrastive_weight": w_contrastive,
            "uncertainty_reg_weight": w_uncertainty_reg,
            "attn_entropy_weight": w_attn_entropy,
        },
    }

    json_path = results_dir / "fusion_core_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved JSON results to {json_path}")

    cm_path = results_dir / "fusion_core_confusion_matrix.png"
    plot_confusion_matrix(
        test_cm,
        class_names,
        cm_path,
        title="Core Fusion (RGB+Depth) Confusion Matrix",
    )
    print(f"[INFO] Saved confusion matrix heatmap to {cm_path}")

    attn_heatmap_path = results_dir / "fusion_core_heatmap.png"
    plot_attention_heatmap(
        mean_att_per_class,
        class_names,
        attn_heatmap_path,
        title="Core Fusion: Mean Modality Attention per Class",
        modality_labels=("RGB", "Depth"),
    )
    print(f"[INFO] Saved attention heatmap to {attn_heatmap_path}")


if __name__ == "__main__":
    main()
