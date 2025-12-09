"""
Tversky Loss - Asymmetric IoU loss that penalizes false negatives more

Key for fixing under-segmentation:
- alpha (FP weight): 0.3 - tolerate some false positives
- beta (FN weight): 0.7 - strongly penalize false negatives (missed detections)

With beta > alpha, the model is forced to predict MORE positive pixels,
directly addressing under-segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice coefficient

    Args:
        alpha: Weight for false positives (default: 0.3)
        beta: Weight for false negatives (default: 0.7)
        smooth: Smoothing constant (default: 1e-6)

    The key insight:
    - beta > alpha means FN are penalized more than FP
    - This forces the model to predict more positive pixels
    - Fixes under-segmentation at the cost of some over-segmentation
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        assert abs(alpha + beta - 1.0) < 1e-6, "alpha + beta should equal 1.0"

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - model output (logits or probs)
            target: [B, 1, H, W] - ground truth in [0, 1]

        Returns:
            tversky_loss: 1 - Tversky index
        """
        # Apply sigmoid if needed (check if values are in [0, 1])
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # True positives, false positives, false negatives
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Return loss (1 - Tversky)
        return 1 - tversky


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 1

        # Create Gaussian window
        import numpy as np
        gauss = torch.tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(self.channel, 1, window_size, window_size).contiguous()
        self.register_buffer('window', window)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W]
            target: [B, 1, H, W]
        """
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Ensure window is on correct device and dtype (for AMP compatibility)
        window = self.window.to(device=pred.device, dtype=pred.dtype)

        # Mean
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Variance and covariance
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        # SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class CombinedSingleExpertLoss(nn.Module):
    """
    Combined loss for single expert training

    Components:
    - Tversky Loss (weight=2.0): Penalizes false negatives
    - BCE with pos_weight=3.0 (weight=1.0): Boosts foreground
    - SSIM Loss (weight=0.5): Structural similarity
    """

    def __init__(self,
                 tversky_weight=2.0,
                 bce_weight=1.0,
                 ssim_weight=0.5,
                 pos_weight=3.0):
        super().__init__()

        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.ssim_weight = ssim_weight
        self.pos_weight = pos_weight

        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        self.ssim_loss = SSIMLoss()

        print(f"\n{'='*70}")
        print("COMBINED LOSS FOR SINGLE EXPERT")
        print(f"{'='*70}")
        print(f"  Tversky Loss: {tversky_weight} (α=0.3, β=0.7) ⭐")
        print(f"  BCE Loss:     {bce_weight} (pos_weight={pos_weight})")
        print(f"  SSIM Loss:    {ssim_weight}")
        print(f"  Total weight: {tversky_weight + bce_weight + ssim_weight}")
        print(f"{'='*70}\n")

    def forward(self, pred, target, aux_preds=None):
        """
        Args:
            pred: Main prediction [B, 1, H, W]
            target: Ground truth [B, 1, H, W]
            aux_preds: List of auxiliary predictions for deep supervision

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Main prediction losses
        tversky = self.tversky_loss(pred, target)

        # BCE with positive weight
        pred_sigmoid = torch.sigmoid(pred) if (pred.min() < 0 or pred.max() > 1) else pred
        pos_weight_tensor = torch.ones_like(target) * self.pos_weight
        bce = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=pos_weight_tensor,
            reduction='mean'
        )

        ssim = self.ssim_loss(pred, target)

        # Weighted combination
        total_loss = (
            self.tversky_weight * tversky +
            self.bce_weight * bce +
            self.ssim_weight * ssim
        )

        # Deep supervision
        if aux_preds is not None:
            for aux_pred in aux_preds:
                # Resize aux prediction to match target size
                if aux_pred.shape != target.shape:
                    aux_pred = F.interpolate(aux_pred, size=target.shape[2:], mode='bilinear', align_corners=False)

                aux_tversky = self.tversky_loss(aux_pred, target)
                aux_bce = F.binary_cross_entropy_with_logits(
                    aux_pred, target,
                    pos_weight=pos_weight_tensor,
                    reduction='mean'
                )
                aux_ssim = self.ssim_loss(aux_pred, target)

                # Add with reduced weight
                total_loss += 0.4 * (
                    self.tversky_weight * aux_tversky +
                    self.bce_weight * aux_bce +
                    self.ssim_weight * aux_ssim
                )

        loss_dict = {
            'total': total_loss.item(),
            'tversky': tversky.item(),
            'bce': bce.item(),
            'ssim': ssim.item()
        }

        return total_loss, loss_dict


if __name__ == '__main__':
    print("Testing Tversky Loss...")

    # Create dummy data
    pred = torch.randn(2, 1, 352, 352)
    target = torch.randint(0, 2, (2, 1, 352, 352)).float()

    # Test Tversky Loss
    tversky_loss = TverskyLoss()
    loss = tversky_loss(pred, target)
    print(f"Tversky Loss: {loss:.4f}")

    # Test Combined Loss
    combined_loss = CombinedSingleExpertLoss()
    total, loss_dict = combined_loss(pred, target)
    print(f"\nCombined Loss: {total:.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ Tests passed!")
