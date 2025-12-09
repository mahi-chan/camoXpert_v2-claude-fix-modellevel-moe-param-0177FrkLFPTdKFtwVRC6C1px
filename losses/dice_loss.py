import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice loss for handling class imbalance in binary segmentation.

    Dice coefficient = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice coefficient
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice