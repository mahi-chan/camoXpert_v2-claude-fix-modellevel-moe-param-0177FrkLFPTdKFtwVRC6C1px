"""
Boundary Prior Network (BPN)

Predicts object boundaries BEFORE segmentation using discontinuity cues.
Key insight: Predicting boundaries first prevents segmentation leakage.

Inputs: Multi-scale features + texture discontinuity + gradient anomaly maps
Output: Boundary map that guides expert segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryPriorNetwork(nn.Module):
    """
    Fuses discontinuity signals to predict boundaries before segmentation.

    Args:
        feature_dims: List of feature dimensions [64, 128, 320, 512]
        hidden_dim: Hidden dimension for processing (default: 64)
    """
    def __init__(self, feature_dims=[64, 128, 320, 512], hidden_dim=64):
        super().__init__()

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim

        # Fuse discontinuity signals (texture + gradient)
        self.signal_fusion = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Per-scale boundary predictors
        self.scale_predictors = nn.ModuleList()
        for dim in feature_dims:
            self.scale_predictors.append(nn.Sequential(
                nn.Conv2d(dim + hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, 1)
            ))

        # Multi-scale fusion for final boundary
        self.boundary_fusion = nn.Sequential(
            nn.Conv2d(len(feature_dims), 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features, texture_disc, gradient_anomaly):
        """
        Args:
            features: List of multi-scale features [f1, f2, f3, f4]
            texture_disc: Texture discontinuity map [B, 1, H, W]
            gradient_anomaly: Gradient anomaly map [B, 1, H, W]

        Returns:
            boundary: Final boundary prediction [B, 1, H, W]
            boundary_scales: List of per-scale boundary predictions
        """
        # Fuse discontinuity signals
        signals = torch.cat([texture_disc, gradient_anomaly], dim=1)
        fused_signal = self.signal_fusion(signals)  # [B, hidden_dim, H, W]

        # Target size (use f1 resolution)
        target_size = features[0].shape[2:]

        # Predict boundaries at each scale
        boundary_preds = []

        for feat, predictor in zip(features, self.scale_predictors):
            # Resize signal to feature scale
            signal_resized = F.interpolate(
                fused_signal,
                size=feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            # Concatenate features with signal
            combined = torch.cat([feat, signal_resized], dim=1)

            # Predict boundary at this scale
            boundary = predictor(combined)

            # Resize to target size
            boundary = F.interpolate(
                boundary,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            boundary_preds.append(boundary)

        # Fuse multi-scale predictions
        multi_scale = torch.cat(boundary_preds, dim=1)
        final_boundary = torch.sigmoid(self.boundary_fusion(multi_scale))

        # Also return individual scale predictions (for deep supervision)
        boundary_scales = [torch.sigmoid(b) for b in boundary_preds]

        return final_boundary, boundary_scales


# Test
if __name__ == '__main__':
    print("Testing BoundaryPriorNetwork...")

    bpn = BoundaryPriorNetwork(feature_dims=[64, 128, 320, 512], hidden_dim=64)

    # Count parameters
    params = sum(p.numel() for p in bpn.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")

    # Create dummy inputs
    features = [
        torch.randn(2, 64, 104, 104),   # f1: H/4
        torch.randn(2, 128, 52, 52),    # f2: H/8
        torch.randn(2, 320, 26, 26),    # f3: H/16
        torch.randn(2, 512, 13, 13),    # f4: H/32
    ]
    texture_disc = torch.randn(2, 1, 104, 104)
    gradient_anomaly = torch.randn(2, 1, 104, 104)

    # Test forward
    boundary, boundary_scales = bpn(features, texture_disc, gradient_anomaly)

    print(f"Boundary shape: {boundary.shape}")
    print(f"Boundary range: [{boundary.min():.3f}, {boundary.max():.3f}]")
    print(f"Number of scale predictions: {len(boundary_scales)}")

    # Verify gradient flow
    loss = boundary.mean()
    loss.backward()
    print("✓ Gradient flow OK")

    print("✓ BPN test passed!")
