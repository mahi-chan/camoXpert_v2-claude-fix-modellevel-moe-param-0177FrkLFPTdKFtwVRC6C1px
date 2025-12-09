"""
Texture Discontinuity Detector (TDD)

Detects where local texture doesn't match surrounding texture.
Key insight: Camouflage matches texture but matching is never perfect.

Output: Discontinuity map where high values indicate texture breaks (potential boundaries)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureDiscontinuityDetector(nn.Module):
    """
    Learns texture descriptors and compares them across local regions.

    Args:
        in_channels: Input feature channels (e.g., 64 for f1 from PVT)
        descriptor_dim: Dimension of learned texture descriptors (default: 64)
        scales: Neighborhood sizes for comparison (default: [3, 5, 7, 11])
    """
    def __init__(self, in_channels, descriptor_dim=64, scales=[3, 5, 7, 11]):
        super().__init__()

        self.descriptor_dim = descriptor_dim
        self.scales = scales

        # Learn texture descriptors (neural texture encoder)
        self.texture_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, descriptor_dim, 1, bias=False),
            nn.BatchNorm2d(descriptor_dim)
        )

        # Discontinuity prediction from multi-scale comparisons
        self.discontinuity_head = nn.Sequential(
            nn.Conv2d(len(scales), 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W] (typically f1 from backbone)

        Returns:
            discontinuity_map: [B, 1, H, W] - high where texture breaks
            descriptors: [B, D, H, W] - per-pixel texture descriptors (for loss)
        """
        B, C, H, W = x.shape

        # Get texture descriptors for each pixel
        descriptors = self.texture_encoder(x)  # [B, D, H, W]
        descriptors = F.normalize(descriptors, dim=1, eps=1e-6)  # L2 normalize

        discontinuity_maps = []

        for kernel_size in self.scales:
            padding = kernel_size // 2

            # Get neighborhood average descriptor
            neighbor_desc = F.avg_pool2d(
                descriptors,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )

            # Handle edge cases where pooling changes size
            if neighbor_desc.shape[2:] != descriptors.shape[2:]:
                neighbor_desc = F.interpolate(
                    neighbor_desc,
                    size=descriptors.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            neighbor_desc = F.normalize(neighbor_desc, dim=1, eps=1e-6)

            # Compute cosine similarity
            similarity = (descriptors * neighbor_desc).sum(dim=1, keepdim=True)

            # Discontinuity = 1 - similarity (high where texture differs)
            discontinuity = 1 - similarity
            discontinuity_maps.append(discontinuity)

        # Combine multi-scale discontinuities
        multi_scale = torch.cat(discontinuity_maps, dim=1)  # [B, num_scales, H, W]

        # Predict final discontinuity map
        discontinuity_map = torch.sigmoid(self.discontinuity_head(multi_scale))

        return discontinuity_map, descriptors


# Test
if __name__ == '__main__':
    print("Testing TextureDiscontinuityDetector...")

    tdd = TextureDiscontinuityDetector(in_channels=64, descriptor_dim=64)

    # Count parameters
    params = sum(p.numel() for p in tdd.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")

    # Test forward
    x = torch.randn(2, 64, 104, 104)  # Simulating f1 features at 416/4 = 104
    disc_map, descriptors = tdd(x)

    print(f"Input shape: {x.shape}")
    print(f"Discontinuity map shape: {disc_map.shape}")
    print(f"Descriptors shape: {descriptors.shape}")
    print(f"Discontinuity range: [{disc_map.min():.3f}, {disc_map.max():.3f}]")

    # Verify gradient flow
    loss = disc_map.mean()
    loss.backward()
    print("✓ Gradient flow OK")

    print("✓ TDD test passed!")
