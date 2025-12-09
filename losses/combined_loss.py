"""
Combined Loss Function for Camouflaged Object Detection

Addresses under-segmentation by:
1. FocalLoss - handles class imbalance with pos_weight
2. TverskyLoss - asymmetric IoU penalizing false negatives (beta=0.7)
3. BoundaryLoss - penalizes boundary errors
4. SSIMLoss - structural similarity
5. DiceLoss - standard segmentation metric

Key: TverskyLoss with beta=0.7 forces model to predict MORE positive pixels,
fixing under-segmentation issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Args:
        alpha: Weight for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        pos_weight: Additional weight for positive pixels (default: 3.0)
    """

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] - model output (NOT sigmoid)
            targets: [B, 1, H, W] - ground truth in [0, 1]
        """
        # Flatten for easier computation
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)

        # Binary cross entropy with logits (autocast-safe)
        bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction='none')

        # Get probabilities for focal weighting (detached to save memory)
        with torch.no_grad():
            probs = torch.sigmoid(logits_flat)
            # Focal term: (1 - p_t)^gamma
            p_t = probs * targets_flat + (1 - probs) * (1 - targets_flat)
            focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_weight = self.alpha * targets_flat + (1 - self.alpha) * (1 - targets_flat)

        # Positive pixel weighting (boost foreground importance)
        pos_weight_tensor = self.pos_weight * targets_flat + 1.0 * (1 - targets_flat)

        # Combined focal loss
        focal_loss = alpha_weight * focal_weight * pos_weight_tensor * bce

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Asymmetric IoU that penalizes false negatives more.

    Critical for fixing under-segmentation!

    Args:
        alpha: Weight for false positives (default: 0.3)
        beta: Weight for false negatives (default: 0.7)
        smooth: Smoothing constant (default: 1.0)

    With beta=0.7 > alpha=0.3, we penalize missed detections (FN) more than
    false alarms (FP), forcing the model to predict MORE positive pixels.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        assert alpha + beta == 1.0, "alpha + beta should equal 1.0"

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] - model output (NOT sigmoid)
            targets: [B, 1, H, W] - ground truth in [0, 1]
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives, False Negatives
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()

        # Tversky Index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Return loss (1 - Tversky)
        return 1 - tversky


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - Penalizes errors near object boundaries.

    Uses Laplacian edge detection to find boundaries and weights
    errors near boundaries more heavily.
    """

    def __init__(self):
        super().__init__()

        # Laplacian kernel for edge detection
        laplacian_kernel = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] - model output (NOT sigmoid)
            targets: [B, 1, H, W] - ground truth in [0, 1]
        """
        # Force float32 for stable edge detection (bypass autocast)
        with torch.cuda.amp.autocast(enabled=False):
            # Cast inputs to float32 for edge detection
            targets_fp32 = targets.float()
            logits_fp32 = logits.float()

            # Ensure kernel is on same device as input
            kernel = self.laplacian_kernel.to(device=targets_fp32.device, dtype=targets_fp32.dtype)

            # Detect boundaries in GT using Laplacian
            boundaries = torch.abs(F.conv2d(
                targets_fp32,
                kernel,
                padding=1
            ))

            # Normalize boundaries to [0, 1]
            boundaries = torch.clamp(boundaries, 0, 1)

            # Create boundary weight map (1.0 + 2.0 * boundary_strength)
            boundary_weights = 1.0 + 2.0 * boundaries

            # Weighted BCE with logits (autocast-safe)
            bce = F.binary_cross_entropy_with_logits(logits_fp32, targets_fp32, reduction='none')
            weighted_bce = bce * boundary_weights

        return weighted_bce.mean()


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.

    Captures structural information and perceptual quality.
    """

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 1

        # Create Gaussian window
        self.window = self._create_window(window_size, sigma, self.channel)

    def _gaussian(self, window_size, sigma):
        """Create 1D Gaussian kernel"""
        gauss = torch.tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, sigma, channel):
        """Create 2D Gaussian window"""
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] - model output (NOT sigmoid)
            targets: [B, 1, H, W] - ground truth in [0, 1]
        """
        # Apply sigmoid
        probs = torch.sigmoid(logits)

        # Move window to same device
        if self.window.device != probs.device:
            self.window = self.window.to(probs.device)

        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Mean
        mu1 = F.conv2d(probs, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(targets, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Variance and covariance
        sigma1_sq = F.conv2d(probs * probs, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(targets * targets, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(probs * targets, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # Return 1 - SSIM as loss
        return 1 - ssim_map.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss - Standard segmentation loss.

    Args:
        smooth: Smoothing constant (default: 1.0)
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1, H, W] - model output (NOT sigmoid)
            targets: [B, 1, H, W] - ground truth in [0, 1]
        """
        # Apply sigmoid
        probs = torch.sigmoid(logits)

        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Dice coefficient
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        # Return dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Loss for fixing under-segmentation in COD.

    Combines multiple loss functions with carefully tuned weights:
    - FocalLoss (1.0): Handles class imbalance with pos_weight=3
    - TverskyLoss (2.0): HIGH WEIGHT - penalizes false negatives (beta=0.7)
    - BoundaryLoss (1.0): Improves boundary accuracy
    - SSIMLoss (0.5): Structural similarity
    - DiceLoss (1.0): Standard segmentation metric

    Total weight: 5.5

    Args:
        focal_weight: Weight for focal loss (default: 1.0)
        tversky_weight: Weight for tversky loss (default: 2.0) - HIGH for IoU
        boundary_weight: Weight for boundary loss (default: 1.0)
        ssim_weight: Weight for SSIM loss (default: 0.5)
        dice_weight: Weight for dice loss (default: 1.0)
        pos_weight: Weight for positive pixels in focal loss (default: 3.0)
    """

    def __init__(self,
                 focal_weight=1.0,
                 tversky_weight=2.0,
                 boundary_weight=1.0,
                 ssim_weight=0.5,
                 dice_weight=1.0,
                 pos_weight=3.0):
        super().__init__()

        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.ssim_weight = ssim_weight
        self.dice_weight = dice_weight

        # Initialize loss components
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=1.0)
        self.boundary_loss = BoundaryLoss()
        self.ssim_loss = SSIMLoss(window_size=11, sigma=1.5)
        self.dice_loss = DiceLoss(smooth=1.0)

        print("\n" + "="*70)
        print("COMBINED LOSS FOR FIXING UNDER-SEGMENTATION")
        print("="*70)
        print(f"  Focal Loss weight:    {focal_weight:.1f} (pos_weight={pos_weight:.1f})")
        print(f"  Tversky Loss weight:  {tversky_weight:.1f} ⭐ (beta=0.7 penalizes FN)")
        print(f"  Boundary Loss weight: {boundary_weight:.1f}")
        print(f"  SSIM Loss weight:     {ssim_weight:.1f}")
        print(f"  Dice Loss weight:     {dice_weight:.1f}")
        print(f"  Total weight:         {focal_weight + tversky_weight + boundary_weight + ssim_weight + dice_weight:.1f}")
        print("="*70 + "\n")

    def forward(self, logits, targets):
        """
        Compute combined loss.

        Args:
            logits: [B, 1, H, W] - model output (NOT sigmoid)
            targets: [B, 1, H, W] - ground truth in [0, 1]

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components for logging
        """
        # Compute individual losses
        focal = self.focal_loss(logits, targets)
        tversky = self.tversky_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        ssim = self.ssim_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        # Weighted combination
        total_loss = (
            self.focal_weight * focal +
            self.tversky_weight * tversky +
            self.boundary_weight * boundary +
            self.ssim_weight * ssim +
            self.dice_weight * dice
        )

        # Create loss dict for logging
        loss_dict = {
            'total': total_loss.item(),
            'focal': focal.item(),
            'tversky': tversky.item(),
            'boundary': boundary.item(),
            'ssim': ssim.item(),
            'dice': dice.item()
        }

        return total_loss, loss_dict


# Test the loss functions
if __name__ == '__main__':
    print("Testing Combined Loss...")

    # Create dummy data
    batch_size = 2
    height, width = 352, 352

    # Logits (model output before sigmoid)
    logits = torch.randn(batch_size, 1, height, width)

    # Targets (ground truth)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    # Test individual losses
    print("\nTesting individual losses:")
    focal = FocalLoss()
    print(f"  Focal Loss: {focal(logits, targets):.4f}")

    tversky = TverskyLoss()
    print(f"  Tversky Loss: {tversky(logits, targets):.4f}")

    boundary = BoundaryLoss()
    print(f"  Boundary Loss: {boundary(logits, targets):.4f}")

    ssim = SSIMLoss()
    print(f"  SSIM Loss: {ssim(logits, targets):.4f}")

    dice = DiceLoss()
    print(f"  Dice Loss: {dice(logits, targets):.4f}")

    # Test combined loss
    print("\nTesting combined loss:")
    combined = CombinedLoss()
    total_loss, loss_dict = combined(logits, targets)

    print(f"  Total Loss: {total_loss:.4f}")
    print("\n  Individual components:")
    for name, value in loss_dict.items():
        if name != 'total':
            print(f"    {name}: {value:.4f}")

    print("\n✓ All loss functions working correctly!")
