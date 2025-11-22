# models/rgb_depth_baselines.py

import torch
import torch.nn as nn
import torchvision.models as models

class RGBBaselineResNet18(nn.Module):
    """
    Simple RGB-only video classifier:
    - Apply pretrained ResNet-18 per frame
    - Average frame features over time
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.feature_dim = resnet.fc.in_features
        # backbone: everything except final FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, feat_dim, 1, 1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, rgb):
        """
        rgb: (B, T, 3, H, W)
        """
        B, T, C, H, W = rgb.shape
        x = rgb.view(B * T, C, H, W)           # (B*T, 3, H, W)
        feats = self.backbone(x)               # (B*T, feat_dim, 1, 1)
        feats = feats.view(B, T, self.feature_dim)  # (B, T, feat_dim)
        feats = feats.mean(dim=1)              # (B, feat_dim)
        logits = self.classifier(feats)        # (B, num_classes)
        return logits


class DepthBaselineResNet18(nn.Module):
    """
    Depth-only baseline:
    - Adapt ResNet-18 to 1-channel input
    - Apply per frame, average over time
    """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Modify first conv to accept 1 channel instead of 3
        old_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )
        if pretrained:
            # Average weights across RGB channels to init the 1-channel conv
            with torch.no_grad():
                resnet.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)

        self.feature_dim = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, depth):
        """
        depth: (B, T, 1, H, W)
        """
        B, T, C, H, W = depth.shape
        x = depth.view(B * T, C, H, W)          # (B*T, 1, H, W)
        feats = self.backbone(x)                # (B*T, feat_dim, 1, 1)
        feats = feats.view(B, T, self.feature_dim)
        feats = feats.mean(dim=1)               # (B, feat_dim)
        logits = self.classifier(feats)
        return logits
