"""
Simple Working Loss - Replaces complex CompositeLossSystem
Based on PraNet's structure loss with label smoothing for regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLossSystem(nn.Module):
    """
    Simple loss replacing the complex multi-component system.
    Uses PraNet's proven structure loss with label smoothing.
    """

    def __init__(self, label_smoothing=0.1, total_epochs=100, **kwargs):
        """
        Args:
            label_smoothing: Smoothing factor (0.1 recommended for anti-overfitting)
            total_epochs: Ignored, for interface compatibility
            **kwargs: Ignored, for interface compatibility
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def update_epoch(self, current_epoch, total_epochs=None):
        """Interface compatibility - does nothing."""
        self.current_epoch = current_epoch
        if total_epochs:
            self.total_epochs = total_epochs

    def structure_loss(self, pred, mask):
        """
        PraNet's structure loss with label smoothing.

        Args:
            pred: [B, 1, H, W] logits
            mask: [B, 1, H, W] ground truth
        """
        # Apply label smoothing: soft targets instead of hard 0/1
        if self.label_smoothing > 0:
            target = mask * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            target = mask

        # Edge-aware weight map (higher weight near boundaries)
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
        )

        # Weighted BCE
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Weighted IoU (use original mask, not smoothed)
        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

    def forward(self, pred, mask, aux_outputs=None, input_image=None, **kwargs):
        """
        Compute loss.

        Args:
            pred: Main prediction [B, 1, H, W]
            mask: Ground truth [B, 1, H, W]
            aux_outputs: Optional auxiliary predictions (use only first 2)
            input_image: Ignored (interface compatibility)
            **kwargs: Ignored
        """
        # Resize mask if needed
        if pred.shape[2:] != mask.shape[2:]:
            mask = F.interpolate(mask, size=pred.shape[2:], mode='nearest')

        # Main loss
        main_loss = self.structure_loss(pred, mask)

        # Deep supervision (only first 2 aux outputs, not all 3)
        aux_loss = 0.0
        if aux_outputs is not None and isinstance(aux_outputs, (list, tuple)):
            # Only use first 2 auxiliary outputs (reduced from 3)
            for i, aux_pred in enumerate(aux_outputs[:2]):
                if aux_pred is not None:
                    aux_mask = F.interpolate(mask, size=aux_pred.shape[2:], mode='nearest')
                    weight = 0.4 * (0.7 ** i)  # 0.4, 0.28
                    aux_loss = aux_loss + weight * self.structure_loss(aux_pred, aux_mask)

        return main_loss + aux_loss
