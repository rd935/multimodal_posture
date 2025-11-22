# models/multimodal_rgbd.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .rgb_depth_baselines import RGBBaselineResNet18, DepthBaselineResNet18


class MultimodalRGBDEarlyFusion(nn.Module):
    """
    Baseline 2: Simple fusion via concatenation (early fusion).

    - Uses ResNet-18 backbones (same as RGBBaselineResNet18 / DepthBaselineResNet18)
    - Averages frame features over time for each modality
    - Projects RGB and Depth features into a shared embedding space
    - Early fusion via concatenation of embeddings + MLP classifier
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        fusion_hidden_dim: int = 512,
        pretrained: bool = True,
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings

        # ------------------------------------------------------------------
        # Modality-specific backbones (reuse baseline definitions)
        # ------------------------------------------------------------------
        rgb_baseline = RGBBaselineResNet18(num_classes=num_classes, pretrained=pretrained)
        depth_baseline = DepthBaselineResNet18(num_classes=num_classes, pretrained=pretrained)

        self.rgb_backbone = rgb_baseline.backbone           # (B*T, C, H, W) -> (B*T, feat_dim, 1, 1)
        self.depth_backbone = depth_baseline.backbone

        self.rgb_feature_dim = rgb_baseline.feature_dim
        self.depth_feature_dim = depth_baseline.feature_dim

        # ------------------------------------------------------------------
        # Shared embedding projections
        # ------------------------------------------------------------------
        self.rgb_proj = nn.Linear(self.rgb_feature_dim, embed_dim)
        self.depth_proj = nn.Linear(self.depth_feature_dim, embed_dim)

        # ------------------------------------------------------------------
        # Early fusion classifier
        # ------------------------------------------------------------------
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    # -----------------------------
    # Modality encoders
    # -----------------------------
    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb: (B, T, 3, H, W)
        Returns:
            z_rgb: (B, embed_dim)
        """
        B, T, C, H, W = rgb.shape
        x = rgb.view(B * T, C, H, W)                # (B*T, 3, H, W)
        feats = self.rgb_backbone(x)                # (B*T, feat_dim, 1, 1)
        feats = feats.view(B, T, self.rgb_feature_dim)
        feats = feats.mean(dim=1)                   # (B, feat_dim)

        z_rgb = self.rgb_proj(feats)                # (B, embed_dim)
        if self.normalize_embeddings:
            z_rgb = F.normalize(z_rgb, dim=-1)
        return z_rgb

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        depth: (B, T, 1, H, W)
        Returns:
            z_depth: (B, embed_dim)
        """
        B, T, C, H, W = depth.shape
        x = depth.view(B * T, C, H, W)              # (B*T, 1, H, W)
        feats = self.depth_backbone(x)              # (B*T, feat_dim, 1, 1)
        feats = feats.view(B, T, self.depth_feature_dim)
        feats = feats.mean(dim=1)                   # (B, feat_dim)

        z_depth = self.depth_proj(feats)            # (B, embed_dim)
        if self.normalize_embeddings:
            z_depth = F.normalize(z_depth, dim=-1)
        return z_depth

    # -----------------------------
    # Fusion + forward
    # -----------------------------
    def fuse_early(self, z_rgb: torch.Tensor, z_depth: torch.Tensor) -> torch.Tensor:
        """
        Early fusion by concatenation.

        z_rgb:   (B, embed_dim)
        z_depth: (B, embed_dim)
        Returns:
            logits: (B, num_classes)
        """
        z_fused = torch.cat([z_rgb, z_depth], dim=-1)   # (B, 2*embed_dim)
        logits = self.fusion_mlp(z_fused)               # (B, num_classes)
        return logits

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        return_embeddings: bool = False,
    ):
        """
        rgb:   (B, T, 3, H, W)
        depth: (B, T, 1, H, W)

        Returns:
            logits: (B, num_classes)
            optionally (if return_embeddings=True) a dict:
                {
                    "z_rgb": (B, embed_dim),
                    "z_depth": (B, embed_dim)
                }
        """
        z_rgb = self.encode_rgb(rgb)
        z_depth = self.encode_depth(depth)

        logits = self.fuse_early(z_rgb, z_depth)

        if return_embeddings:
            return logits, {"z_rgb": z_rgb, "z_depth": z_depth}
        else:
            return logits


class MultimodalRGBDAttentionFusion(nn.Module):
    """
    Baseline 3: Attention-based fusion (strong baseline).

    - Encode RGB and Depth into shared embeddings z_rgb, z_depth
    - Predict modality attention weights [alpha_rgb, alpha_depth]
      from their concatenated embeddings
    - Apply weights to each modality and concatenate:
          z_rgb_w   = alpha_rgb * z_rgb
          z_depth_w = alpha_depth * z_depth
          z_fused   = [z_rgb_w; z_depth_w]
    - Classifier on z_fused

    This keeps the expressive power of early fusion (2*embed_dim input)
    while adding adaptive, interpretable gating.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        fusion_hidden_dim: int = 512,
        attn_hidden_dim: int = 256,
        pretrained: bool = True,
        normalize_embeddings: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings

        # --------- backbones ----------
        rgb_baseline = RGBBaselineResNet18(num_classes=num_classes, pretrained=pretrained)
        depth_baseline = DepthBaselineResNet18(num_classes=num_classes, pretrained=pretrained)

        self.rgb_backbone = rgb_baseline.backbone
        self.depth_backbone = depth_baseline.backbone

        self.rgb_feature_dim = rgb_baseline.feature_dim
        self.depth_feature_dim = depth_baseline.feature_dim

        # Freeze all but the last ResNet block
        if freeze_backbone:
            # Freeze most of RGB backbone
            for name, param in self.rgb_backbone.named_parameters():
                if "layer4" not in name:      # keep last block trainable
                    param.requires_grad = False

            # Freeze most of Depth backbone
            for name, param in self.depth_backbone.named_parameters():
                if "layer4" not in name:
                    param.requires_grad = False

        # --------- shared projections ----------

        self.rgb_proj = nn.Linear(self.rgb_feature_dim, embed_dim)
        self.depth_proj = nn.Linear(self.depth_feature_dim, embed_dim)

        # --------- attention MLP over modalities ----------
        # input: concat [z_rgb, z_depth] -> (B, 2*D)
        # output: logits for [RGB, Depth] -> (B, 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, attn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden_dim, 2),
        )

        last_attn_linear = self.attn_mlp[-1]
        nn.init.zeros_(last_attn_linear.weight)
        nn.init.zeros_(last_attn_linear.bias)

        # --------- classifier on fused representation ----------
        # input: concat [z_rgb_w, z_depth_w] -> (B, 2*D)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    # ---------- encoders ----------
    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = rgb.shape
        x = rgb.view(B * T, C, H, W)
        feats = self.rgb_backbone(x)                      # (B*T, feat_dim, 1, 1)
        feats = feats.view(B, T, self.rgb_feature_dim)    # (B, T, feat_dim)
        feats = feats.mean(dim=1)                         # (B, feat_dim)
        z_rgb = self.rgb_proj(feats)                      # (B, D)
        if self.normalize_embeddings:
            z_rgb = F.normalize(z_rgb, dim=-1)
        return z_rgb

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = depth.shape
        x = depth.view(B * T, C, H, W)
        feats = self.depth_backbone(x)                    # (B*T, feat_dim, 1, 1)
        feats = feats.view(B, T, self.depth_feature_dim)  # (B, T, feat_dim)
        feats = feats.mean(dim=1)                         # (B, feat_dim)
        z_depth = self.depth_proj(feats)                  # (B, D)
        if self.normalize_embeddings:
            z_depth = F.normalize(z_depth, dim=-1)
        return z_depth

    # ---------- attention fusion ----------
    def fuse_attention(
        self,
        z_rgb: torch.Tensor,
        z_depth: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        z_rgb, z_depth: (B, D)

        Returns:
            logits: (B, num_classes)
            attn_info (optional): {"modality_attention": (B, 2)}
        """

        # concat embeddings
        h = torch.cat([z_rgb, z_depth], dim=-1)  # (B, 2*D)

        # attention logits and weights
        attn_logits = self.attn_mlp(h)           # (B, 2)
        tau = 1.0
        modality_attention = F.softmax(attn_logits / tau, dim=-1)  # (B, 2)

        alpha_rgb = modality_attention[:, 0:1]   # (B, 1)
        alpha_depth = modality_attention[:, 1:2] # (B, 1)

        # weighted embeddings
        z_rgb_w = alpha_rgb * z_rgb             # (B, D)
        z_depth_w = alpha_depth * z_depth       # (B, D)

        # fused representation keeps 2*D like early fusion
        z_fused = torch.cat([z_rgb_w, z_depth_w], dim=-1)  # (B, 2*D)

        logits = self.fusion_mlp(z_fused)

        if not return_attention:
            return logits, None

        attn_info = {
            "modality_attention": modality_attention,  # (B, 2)
        }
        return logits, attn_info

    # ---------- forward ----------
    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        return_embeddings: bool = False,
        return_attention: bool = False,
    ):
        z_rgb = self.encode_rgb(rgb)
        z_depth = self.encode_depth(depth)

        logits, attn_info = self.fuse_attention(
            z_rgb, z_depth, return_attention=return_attention
        )

        if not (return_embeddings or return_attention):
            return logits

        extras: Dict[str, Any] = {}
        if return_embeddings:
            extras["z_rgb"] = z_rgb
            extras["z_depth"] = z_depth
        if return_attention and attn_info is not None:
            extras.update(attn_info)

        return logits, extras


class MultimodalRGBDAttnContrastiveUncertainty(nn.Module):
    """
    Main model: Attention fusion + contrastive alignment + uncertainty heads.

    - Encodes RGB and Depth into shared embeddings z_rgb, z_depth
    - Uses attention gating (as in MultimodalRGBDAttentionFusion)
    - Projects embeddings into a contrastive space (proj_rgb, proj_depth)
      for InfoNCE-style RGB–Depth alignment
    - Predicts per-modality log-variance (logvar_rgb, logvar_depth) to
      support uncertainty analysis (regularized during training)

    Forward interface is designed for training scripts to:
      - use logits for classification
      - use proj_* for contrastive loss
      - use logvar_* for uncertainty regularization / analysis
      - use modality_attention for analysis
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        fusion_hidden_dim: int = 512,
        attn_hidden_dim: int = 256,
        proj_dim: int = 128,
        pretrained: bool = True,
        normalize_embeddings: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.proj_dim = proj_dim

        # --------- backbones ----------
        rgb_baseline = RGBBaselineResNet18(num_classes=num_classes, pretrained=pretrained)
        depth_baseline = DepthBaselineResNet18(num_classes=num_classes, pretrained=pretrained)

        self.rgb_backbone = rgb_baseline.backbone
        self.depth_backbone = depth_baseline.backbone

        self.rgb_feature_dim = rgb_baseline.feature_dim
        self.depth_feature_dim = depth_baseline.feature_dim

        if freeze_backbone:
            for name, p in self.rgb_backbone.named_parameters():
                if "layer4" not in name:   # keep the last block trainable
                    p.requires_grad = False
            for name, p in self.depth_backbone.named_parameters():
                if "layer4" not in name:
                    p.requires_grad = False

        # --------- shared projections ----------
        self.rgb_proj = nn.Linear(self.rgb_feature_dim, embed_dim)
        self.depth_proj = nn.Linear(self.depth_feature_dim, embed_dim)

        # --------- attention MLP over modalities ----------
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, attn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden_dim, 2),
        )

        # NEW: start attention at equal RGB/Depth weights (0.5, 0.5)
        last_attn_linear = self.attn_mlp[-1]
        nn.init.zeros_(last_attn_linear.weight)
        nn.init.zeros_(last_attn_linear.bias)

        # --------- classifier on fused representation ----------
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

        # --------- contrastive projection heads ----------
        # Map embeddings to a lower-dim space for contrastive learning
        self.rgb_contrastive_head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.depth_contrastive_head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

        # --------- uncertainty (variance) heads ----------
        # Predict per-sample log-variance for each modality (scalar)
        var_hidden_dim = attn_hidden_dim // 2 if attn_hidden_dim >= 2 else 16

        self.rgb_var_head = nn.Sequential(
            nn.Linear(embed_dim, var_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(var_hidden_dim, 1),  # raw variance score
        )
        self.depth_var_head = nn.Sequential(
            nn.Linear(embed_dim, var_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(var_hidden_dim, 1),  # raw variance score
        )

    # ---------- encoders ----------
    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb: (B, T, 3, H, W)
        Returns: z_rgb (B, D)
        """
        B, T, C, H, W = rgb.shape
        x = rgb.view(B * T, C, H, W)
        feats = self.rgb_backbone(x)                      # (B*T, feat_dim, 1, 1)
        feats = feats.view(B, T, self.rgb_feature_dim)    # (B, T, feat_dim)
        feats = feats.mean(dim=1)                         # (B, feat_dim)
        z_rgb = self.rgb_proj(feats)                      # (B, D)
        if self.normalize_embeddings:
            z_rgb = F.normalize(z_rgb, dim=-1)
        return z_rgb

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        depth: (B, T, 1, H, W)
        Returns: z_depth (B, D)
        """
        B, T, C, H, W = depth.shape
        x = depth.view(B * T, C, H, W)
        feats = self.depth_backbone(x)                    # (B*T, feat_dim, 1, 1)
        feats = feats.view(B, T, self.depth_feature_dim)  # (B, T, feat_dim)
        feats = feats.mean(dim=1)                         # (B, feat_dim)
        z_depth = self.depth_proj(feats)                  # (B, D)
        if self.normalize_embeddings:
            z_depth = F.normalize(z_depth, dim=-1)
        return z_depth

    # ---------- attention fusion ----------
    def _attention_fuse(
        self,
        z_rgb: torch.Tensor,
        z_depth: torch.Tensor,
    ):
        """
        Internal helper to compute:
          - modality attention [alpha_rgb, alpha_depth]
          - fused representation z_fused
        """
        h = torch.cat([z_rgb, z_depth], dim=-1)  # (B, 2*D)

        attn_logits = self.attn_mlp(h)          # (B, 2)
        modality_attention = F.softmax(attn_logits, dim=-1)  # (B, 2)

        alpha_rgb = modality_attention[:, 0:1]    # (B, 1)
        alpha_depth = modality_attention[:, 1:2]  # (B, 1)

        z_rgb_w = alpha_rgb * z_rgb               # (B, D)
        z_depth_w = alpha_depth * z_depth         # (B, D)

        z_fused = torch.cat([z_rgb_w, z_depth_w], dim=-1)  # (B, 2*D)
        return z_fused, modality_attention

    # ---------- uncertainty heads ----------
    def _predict_logvars(
        self,
        z_rgb: torch.Tensor,
        z_depth: torch.Tensor,
        eps: float = 1e-6,
    ):
        """
        Predict log-variance for each modality (scalar per sample).

        Returns:
            logvar_rgb:   (B, 1)
            logvar_depth: (B, 1)
        """
        raw_rgb = self.rgb_var_head(z_rgb)        # (B, 1)
        raw_depth = self.depth_var_head(z_depth)  # (B, 1)

        # ensure positive variance via softplus, then take log
        var_rgb = F.softplus(raw_rgb) + eps
        var_depth = F.softplus(raw_depth) + eps

        logvar_rgb = torch.log(var_rgb)           # (B, 1)
        logvar_depth = torch.log(var_depth)       # (B, 1)

        return logvar_rgb, logvar_depth

    # ---------- contrastive projections ----------
    def _project_for_contrastive(
        self,
        z_rgb: torch.Tensor,
        z_depth: torch.Tensor,
        normalize: bool = True,
    ):
        """
        Project embeddings to contrastive space.

        Returns:
            proj_rgb:   (B, proj_dim)
            proj_depth: (B, proj_dim)
        """
        proj_rgb = self.rgb_contrastive_head(z_rgb)
        proj_depth = self.depth_contrastive_head(z_depth)

        if normalize:
            proj_rgb = F.normalize(proj_rgb, dim=-1)
            proj_depth = F.normalize(proj_depth, dim=-1)

        return proj_rgb, proj_depth

    # ---------- forward ----------
    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        return_embeddings: bool = True,
        return_attention: bool = True,
        return_uncertainty: bool = True,
        return_projections: bool = True,
    ):
        """
        rgb:   (B, T, 3, H, W)
        depth: (B, T, 1, H, W)

        Returns:
            If no return_* flags:
                logits
            Otherwise:
                logits, extras

            extras may contain:
                - "z_rgb": (B, embed_dim)
                - "z_depth": (B, embed_dim)
                - "proj_rgb": (B, proj_dim)
                - "proj_depth": (B, proj_dim)
                - "modality_attention": (B, 2)
                - "logvar_rgb": (B, 1)
                - "logvar_depth": (B, 1)
        """
        z_rgb = self.encode_rgb(rgb)
        z_depth = self.encode_depth(depth)

        # attention fusion
        z_fused, modality_attention = self._attention_fuse(z_rgb, z_depth)
        logits = self.fusion_mlp(z_fused)

        # If nothing else requested, just return logits
        if not (return_embeddings or return_attention or return_uncertainty or return_projections):
            return logits

        extras: Dict[str, Any] = {}

        if return_embeddings:
            extras["z_rgb"] = z_rgb
            extras["z_depth"] = z_depth

        if return_projections:
            proj_rgb, proj_depth = self._project_for_contrastive(z_rgb, z_depth)
            extras["proj_rgb"] = proj_rgb
            extras["proj_depth"] = proj_depth

        if return_attention:
            extras["modality_attention"] = modality_attention

        if return_uncertainty:
            logvar_rgb, logvar_depth = self._predict_logvars(z_rgb, z_depth)
            extras["logvar_rgb"] = logvar_rgb
            extras["logvar_depth"] = logvar_depth

        return logits, extras