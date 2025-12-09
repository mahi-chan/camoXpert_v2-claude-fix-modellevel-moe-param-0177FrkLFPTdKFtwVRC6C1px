"""
Gradient Anomaly Detector (GAD)

Detects gradient anomalies that indicate camouflage boundaries.
Key insight: Camouflaged boundaries often have artificially smooth or sharp gradients.

Output: Anomaly map where high values indicate unnatural gradient patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GradientAnomalyDetector(nn.Module):
    """
    Multi-directional gradient analysis to detect camouflage boundaries.

    Args:
        in_channels: Input feature channels
        num_directions: Number of gradient directions (default: 8)
    """
    def __init__(self, in_channels, num_directions=8):
        super().__init__()

        self.in_channels = in_channels
        self.num_directions = num_directions

        # Learnable multi-directional gradient filters (depthwise)
        self.gradient_conv = nn.Conv2d(
            in_channels,
            in_channels * num_directions,
            kernel_size=3,
            padding=1,
            groups=in_channels,  # Depthwise convolution
            bias=False
        )

        # Initialize with directional gradient filters
        self._init_directional_weights()

        # Gradient statistics encoder
        self.stats_encoder = nn.Sequential(
            nn.Conv2d(in_channels * num_directions, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def _init_directional_weights(self):
        """Initialize with 8 directional Sobel-like gradient filters"""
        with torch.no_grad():
            weight = self.gradient_conv.weight  # [C*8, 1, 3, 3]

            for i in range(self.num_directions):
                angle = i * math.pi / 4  # 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

                # Create directional kernel based on angle
                kernel = torch.zeros(3, 3)

                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                # Sobel-like directional filter
                if abs(cos_a) > 0.5:  # Horizontal component
                    kernel[0, 0] += -cos_a * 0.5
                    kernel[0, 2] += cos_a * 0.5
                    kernel[1, 0] += -cos_a
                    kernel[1, 2] += cos_a
                    kernel[2, 0] += -cos_a * 0.5
                    kernel[2, 2] += cos_a * 0.5

                if abs(sin_a) > 0.5:  # Vertical component
                    kernel[0, 0] += -sin_a * 0.5
                    kernel[2, 0] += sin_a * 0.5
                    kernel[0, 1] += -sin_a
                    kernel[2, 1] += sin_a
                    kernel[0, 2] += -sin_a * 0.5
                    kernel[2, 2] += sin_a * 0.5

                # Normalize kernel
                if kernel.abs().sum() > 0:
                    kernel = kernel / (kernel.abs().sum() + 1e-6)

                # Apply to all input channels at this direction
                for c in range(self.in_channels):
                    weight[c * self.num_directions + i, 0] = kernel

    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W]

        Returns:
            anomaly_map: [B, 1, H, W] - high where gradients are anomalous
            gradient_features: [B, 64, H, W] - gradient feature representation
        """
        # Extract multi-directional gradients
        gradients = self.gradient_conv(x)  # [B, C*8, H, W]

        # Take absolute value (magnitude)
        gradients = torch.abs(gradients)

        # Encode gradient statistics
        gradient_features = self.stats_encoder(gradients)

        # Detect anomalies
        anomaly_map = torch.sigmoid(self.anomaly_head(gradient_features))

        return anomaly_map, gradient_features


# Test
if __name__ == '__main__':
    print("Testing GradientAnomalyDetector...")

    gad = GradientAnomalyDetector(in_channels=64, num_directions=8)

    # Count parameters
    params = sum(p.numel() for p in gad.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")

    # Test forward
    x = torch.randn(2, 64, 104, 104)
    anomaly_map, grad_features = gad(x)

    print(f"Input shape: {x.shape}")
    print(f"Anomaly map shape: {anomaly_map.shape}")
    print(f"Gradient features shape: {grad_features.shape}")
    print(f"Anomaly range: [{anomaly_map.min():.3f}, {anomaly_map.max():.3f}]")

    # Verify gradient flow
    loss = anomaly_map.mean()
    loss.backward()
    print("✓ Gradient flow OK")

    print("✓ GAD test passed!")
