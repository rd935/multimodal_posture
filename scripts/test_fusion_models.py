#!/usr/bin/env python

"""
Quick sanity check for multimodal RGB+Depth models.

Run with:
    python scripts/test_fusion_models.py
"""

import os
import sys
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../multimodal_posture/scripts
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                     # .../multimodal_posture

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.multimodal_rgbd import (
    MultimodalRGBDEarlyFusion,
    MultimodalRGBDAttentionFusion,
)


def make_dummy_batch(
    batch_size: int = 2,
    num_frames: int = 8,
    rgb_size=(3, 224, 224),
    depth_size=(1, 224, 224),
    device: torch.device = torch.device("cpu"),
):
    C_rgb, H_rgb, W_rgb = rgb_size
    C_d, H_d, W_d = depth_size

    rgb = torch.randn(batch_size, num_frames, C_rgb, H_rgb, W_rgb, device=device)
    depth = torch.randn(batch_size, num_frames, C_d, H_d, W_d, device=device)

    num_classes = 10
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    return rgb, depth, labels, num_classes


def test_early_fusion(device: torch.device):
    print("\n=== Testing MultimodalRGBDEarlyFusion ===")

    rgb, depth, labels, num_classes = make_dummy_batch(device=device)

    model = MultimodalRGBDEarlyFusion(
        num_classes=num_classes,
        embed_dim=256,
        fusion_hidden_dim=512,
        pretrained=False,
        normalize_embeddings=True,
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits, emb = model(rgb, depth, return_embeddings=True)

    print(f"Logits shape (early fusion): {logits.shape}")
    print(f"z_rgb shape: {emb['z_rgb'].shape}")
    print(f"z_depth shape: {emb['z_depth'].shape}")

    loss = F.cross_entropy(logits, labels)
    print(f"Dummy CE loss (early fusion): {loss.item():.4f}")


def test_attention_fusion(device: torch.device):
    print("\n=== Testing MultimodalRGBDAttentionFusion ===")

    rgb, depth, labels, num_classes = make_dummy_batch(device=device)

    model = MultimodalRGBDAttentionFusion(
        num_classes=num_classes,
        embed_dim=256,
        fusion_hidden_dim=512,
        num_heads=4,
        attn_dropout=0.1,
        pretrained=False,
        normalize_embeddings=True,
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits, extras = model(
            rgb,
            depth,
            return_embeddings=True,
            return_attention=True,
        )

    print(f"Logits shape (attention fusion): {logits.shape}")
    print(f"z_rgb shape: {extras['z_rgb'].shape}")
    print(f"z_depth shape: {extras['z_depth'].shape}")

    attn_matrix = extras["attn_matrix"]          # (B, 2, 2)
    modality_attention = extras["modality_attention"]  # (B, 2)

    print(f"Attention matrix shape: {attn_matrix.shape}")
    print(f"Modality attention shape: {modality_attention.shape}")
    print(f"Modality attention (first sample): {modality_attention[0].tolist()}")

    loss = F.cross_entropy(logits, labels)
    print(f"Dummy CE loss (attention fusion): {loss.item():.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    test_early_fusion(device)
    test_attention_fusion(device)


if __name__ == "__main__":
    main()
