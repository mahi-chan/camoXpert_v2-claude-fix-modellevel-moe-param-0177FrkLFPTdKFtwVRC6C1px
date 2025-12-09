"""
Aggressive Loss Functions for Fixing Under-Confident Predictions

This loss is designed to solve the problem where the model's mean prediction
drops over time (e.g., from 0.365 to 0.190), causing poor IoU despite low loss.

The aggressive approach:
1. FocalTverskyLoss - heavily penalizes false negatives (beta=0.8)
2. AsymmetricBCELoss - heavily penalizes missing foreground (pos_weight=10.0)
3. ConfidencePenaltyLoss - penalizes uncertain predictions near 0.5

This forces the model to:
- Predict MORE positive pixels (addresses under-segmentation)
- Be MORE confident in its predictions (avoid wishy-washy 0.5 predictions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Tversky loss raised to the power of gamma

    This combines:
    - Asymmetric penalty (beta > alpha) to penalize false negatives
    - Focal weighting to focus on hard examples

    Args:
        alpha: Weight for false positives (default: 0.2 - very low FP penalty)
        beta: Weight for false negatives (default: 0.8 - very high FN penalty)
        gamma: Focal parameter (default: 2.0 - standard focal loss)
        smooth: Smoothing constant
    """

    def __init__(self, alpha=0.2, beta=0.8, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

        print(f"\n{'='*70}")
        print("FOCAL TVERSKY LOSS - AGGRESSIVE MODE")
        print(f"{'='*70}")
        print(f"  α (FP weight): {alpha} - VERY LOW (tolerate false positives)")
        print(f"  β (FN weight): {beta} - VERY HIGH (penalize missing objects)")
        print(f"  γ (focal):     {gamma} - Focus on hard examples")
        print(f"  β/α ratio:     {beta/alpha:.1f}x - FN penalized {beta/alpha:.1f}x more than FP")
        print(f"{'='*70}\n")

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - model output (logits or probs)
            target: [B, 1, H, W] - ground truth in [0, 1]

        Returns:
            focal_tversky_loss: (1 - tversky_index)^gamma
        """
        # Apply sigmoid if needed
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

        # Focal Tversky loss
        focal_tversky = torch.pow(1 - tversky, self.gamma)

        return focal_tversky


class AsymmetricBCELoss(nn.Module):
    """
    Asymmetric Binary Cross-Entropy Loss

    Heavily penalizes missing foreground pixels (false negatives)
    while being lenient on false positives.

    Args:
        pos_weight: Weight for positive (foreground) pixels (default: 10.0)
        neg_weight: Weight for negative (background) pixels (default: 0.5)

    Formula:
        loss = -pos_weight * target * log(pred) - neg_weight * (1-target) * log(1-pred)
    """

    def __init__(self, pos_weight=10.0, neg_weight=0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

        print(f"\n{'='*70}")
        print("ASYMMETRIC BCE LOSS - AGGRESSIVE MODE")
        print(f"{'='*70}")
        print(f"  Pos weight: {pos_weight} - HEAVILY penalize missing foreground")
        print(f"  Neg weight: {neg_weight} - LOW penalty for false positives")
        print(f"  Ratio:      {pos_weight/neg_weight:.1f}x - Missing FG penalized {pos_weight/neg_weight:.1f}x more")
        print(f"{'='*70}\n")

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] - model output (logits or probs)
            target: [B, 1, H, W] - ground truth in [0, 1]

        Returns:
            asym_bce_loss: Asymmetric BCE loss
        """
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Clamp to avoid log(0)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        # Asymmetric BCE
        pos_loss = -self.pos_weight * target * torch.log(pred)
        neg_loss = -self.neg_weight * (1 - target) * torch.log(1 - pred)

        loss = (pos_loss + neg_loss).mean()

        return loss


class ConfidencePenaltyLoss(nn.Module):
    """
    Confidence Penalty Loss

    Penalizes predictions that are uncertain (close to 0.5).
    Forces the model to be confident - predict either 0 or 1, not 0.5.

    This helps prevent the model from being too conservative and
    making wishy-washy predictions.

    Formula:
        penalty = relu(0.25 - abs(pred - 0.5)).mean()

    This means:
        - pred = 0.5: penalty = 0.25 (maximum penalty)
        - pred = 0.25 or 0.75: penalty = 0 (no penalty)
        - pred = 0 or 1: penalty = 0 (no penalty)
    """

    def __init__(self):
        super().__init__()

        print(f"\n{'='*70}")
        print("CONFIDENCE PENALTY LOSS")
        print(f"{'='*70}")
        print(f"  Penalizes predictions in [0.25, 0.75]")
        print(f"  Forces model to be confident (near 0 or 1)")
        print(f"  Maximum penalty at pred=0.5 (maximum uncertainty)")
        print(f"{'='*70}\n")

    def forward(self, pred, target=None):
        """
        Args:
            pred: [B, 1, H, W] - model output (logits or probs)
            target: Not used, kept for interface compatibility

        Returns:
            confidence_penalty: Penalty for uncertain predictions
        """
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Penalty for predictions close to 0.5
        # relu(0.25 - |pred - 0.5|) is 0 when |pred - 0.5| > 0.25
        # and increases as pred approaches 0.5
        uncertainty = torch.abs(pred - 0.5)
        penalty = F.relu(0.25 - uncertainty)

        return penalty.mean()


class AggressiveCombinedLoss(nn.Module):
    """
    Aggressive Combined Loss for Fixing Under-Confident Models

    Combines three aggressive losses:
    1. FocalTverskyLoss - forces more positive predictions (weight: 3.0)
    2. AsymmetricBCELoss - heavily penalizes missing foreground (weight: 1.0)
    3. ConfidencePenaltyLoss - forces confident predictions (weight: 0.5)

    Total weight: 4.5

    This is designed to fix models that become under-confident over time,
    where mean prediction drops (e.g., 0.365 -> 0.190) causing poor IoU.
    """

    def __init__(self,
                 focal_tversky_weight=3.0,
                 asym_bce_weight=1.0,
                 confidence_weight=0.5,
                 alpha=0.2,
                 beta=0.8,
                 gamma=2.0,
                 pos_weight=10.0,
                 neg_weight=0.5):
        super().__init__()

        self.focal_tversky_weight = focal_tversky_weight
        self.asym_bce_weight = asym_bce_weight
        self.confidence_weight = confidence_weight

        # Create loss functions
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.asym_bce = AsymmetricBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
        self.confidence = ConfidencePenaltyLoss()

        print(f"\n{'='*70}")
        print("AGGRESSIVE COMBINED LOSS - FINAL CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Focal Tversky: {focal_tversky_weight} (α={alpha}, β={beta}, γ={gamma})")
        print(f"  Asymmetric BCE: {asym_bce_weight} (pos_w={pos_weight}, neg_w={neg_weight})")
        print(f"  Confidence:    {confidence_weight}")
        print(f"  Total weight:  {focal_tversky_weight + asym_bce_weight + confidence_weight}")
        print(f"\n  GOAL: Force model to predict MORE and be MORE confident!")
        print(f"{'='*70}\n")

    def forward(self, pred, target, aux_preds=None):
        """
        Args:
            pred: [B, 1, H, W] - main prediction
            target: [B, 1, H, W] - ground truth
            aux_preds: List of auxiliary predictions (optional)

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        # Main losses
        focal_tversky = self.focal_tversky(pred, target)
        asym_bce = self.asym_bce(pred, target)
        confidence = self.confidence(pred)

        # Weighted combination
        total_loss = (
            self.focal_tversky_weight * focal_tversky +
            self.asym_bce_weight * asym_bce +
            self.confidence_weight * confidence
        )

        # Add auxiliary losses if provided (with lower weight)
        if aux_preds is not None and len(aux_preds) > 0:
            aux_loss = 0.0
            for aux_pred in aux_preds:
                aux_loss += self.focal_tversky(aux_pred, target)
                aux_loss += self.asym_bce(aux_pred, target)

            aux_loss = aux_loss / len(aux_preds)
            total_loss = total_loss + 0.4 * aux_loss

        # Return loss and components
        loss_dict = {
            'focal_tversky': focal_tversky.item(),
            'asym_bce': asym_bce.item(),
            'confidence': confidence.item(),
        }

        return total_loss, loss_dict
