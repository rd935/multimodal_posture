import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import json

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
#   Checkpoint loading (fixed to prefer new unified core)
# =========================================================
def load_checkpoint(model, model_name: str):
    """
    model_name in {"core", "attention"}.

    Expected layout (under PROJECT_ROOT/checkpoints):

        fusion_core/
            fusion_core_best.pt        # <-- preferred (new unified core)
            fusion_core_best_ft.pt     # legacy fine-tuned core (fallback)

        fusion_attention/
            fusion_attention_best.pt
    """
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
#   Build dataloaders & models from config
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
#   ECE + reliability diagram
# =========================================================
def expected_calibration_error(probs, labels, n_bins: int = 15):
    """
    probs: (N, C) numpy
    labels: (N,) numpy, ints
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    labels = labels.astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= start) & (confidences <= end)
        else:
            in_bin = (confidences >= start) & (confidences < end)

        frac_in_bin = in_bin.mean()
        if frac_in_bin > 0:
            acc_in_bin = (preds[in_bin] == labels[in_bin]).mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += frac_in_bin * abs(acc_in_bin - conf_in_bin)

    return float(ece)


def reliability_diagram(probs, labels, n_bins: int = 15):
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    labels = labels.astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    accs = []
    confs = []
    counts = []

    for i in range(n_bins):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= start) & (confidences <= end)
        else:
            in_bin = (confidences >= start) & (confidences < end)

        count = in_bin.sum()
        counts.append(count)
        if count > 0:
            accs.append((preds[in_bin] == labels[in_bin]).mean())
            confs.append(confidences[in_bin].mean())
        else:
            accs.append(0.0)
            confs.append(0.0)

    return bin_centers, np.array(accs), np.array(confs), np.array(counts)


def plot_reliability(bin_centers, accs, confs, model_name: str, out_path: Path):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1])  # perfect calibration line
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram - {model_name}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# =========================================================
#   Inference utils
# =========================================================
@torch.no_grad()
def gather_probs_and_labels(model, data_loader, num_classes: int):
    all_probs = []
    all_labels = []

    model.eval()
    for batch in data_loader:
        rgb = batch["rgb"].to(DEVICE)
        depth = batch["depth"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        # For core fusion, we can safely request uncertainty outputs.
        if isinstance(model, MultimodalRGBDCoreFusion):
            out = model(
                rgb,
                depth,
                return_embeddings=False,
                return_attention=False,
                return_uncertainty=True,
            )
        else:
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

        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_probs, all_labels


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

    results = {}
    out_dir = PROJECT_ROOT / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in ["attention", "core"]:
        print(f"\n============================")
        print(f"[INFO] Evaluating model: {model_name}")
        print("============================")

        model = build_model(cfg, model_name=model_name, num_classes=num_classes)
        probs, labels = gather_probs_and_labels(model, test_loader, num_classes)

        preds = probs.argmax(axis=1)
        acc = float(accuracy_score(labels, preds))
        ece = expected_calibration_error(probs, labels, n_bins=15)

        bin_centers, accs, confs, counts = reliability_diagram(probs, labels, n_bins=15)
        plot_path = out_dir / f"reliability_{model_name}.png"
        plot_reliability(bin_centers, accs, confs, model_name, plot_path)

        print(f"[{model_name.upper()}] test_acc = {acc:.4f}, ECE = {ece:.4f}")
        print(f"[{model_name.upper()}] reliability diagram saved to: {plot_path}")

        results[model_name] = {
            "acc": acc,
            "ece": ece,
            "num_samples": int(len(labels)),
        }

    # Save JSON comparison
    out_json = out_dir / "ece_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "config_path": str(config_path),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n[INFO] Saved ECE comparison to {out_json}")


if __name__ == "__main__":
    main()
