"""
Production Loss for Camouflaged Object Detection - SOTA-Focused

Based on what actually works in SOTA methods:
- SINet: BCE + IoU loss
- PraNet: BCE + Dice + IoU
- ZoomNet: BCE + Dice + structure loss

Key principles:
1. Simple, proven loss components
2. Proper weighting for sparse masks
3. Anti-collapse mechanism
4. Light boundary supervision (not heavy TDD/GAD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss - standard for medical/COD segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


class IoULoss(nn.Module):
    """IoU/Jaccard loss - directly optimizes the metric we care about"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        return 1 - (intersection + self.smooth) / (union + self.smooth)


class FocalLoss(nn.Module):
    """Focal loss - handles class imbalance in sparse masks"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

        # Binary focal loss (autocast-safe)
        with torch.amp.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(pred.float(), target.float(), reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)

        # Alpha weighting: more weight to foreground
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class StructureLoss(nn.Module):
    """
    Structure loss from PraNet - combines weighted BCE and IoU
    This is proven to work well for COD
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Weighted BCE - more weight to foreground
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Weight: inverse frequency
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # Weighted IoU
        pred_sig = torch.sigmoid(pred)
        inter = ((pred_sig * target) * weit).sum(dim=(2, 3))
        union = ((pred_sig + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class BoundaryLoss(nn.Module):
    """Simple boundary loss - just BCE + Dice on boundary map"""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()

    def _extract_boundary(self, mask):
        # Laplacian-based boundary extraction
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        boundary = F.conv2d(mask, laplacian, padding=1)
        boundary = torch.abs(boundary)
        boundary = torch.clamp(boundary, 0, 1)
        return boundary

    def forward(self, pred_boundary, target_mask):
        # Resize if needed
        if pred_boundary.shape[2:] != target_mask.shape[2:]:
            target_mask = F.interpolate(target_mask, size=pred_boundary.shape[2:],
                                        mode='bilinear', align_corners=False)

        target_boundary = self._extract_boundary(target_mask)
        pred_boundary = torch.clamp(pred_boundary, 1e-7, 1 - 1e-7)

        # BCE with autocast safety
        with torch.amp.autocast('cuda', enabled=False):
            bce = F.binary_cross_entropy(pred_boundary.float(), target_boundary.float())
        dice = self.dice(pred_boundary, target_boundary)

        return bce + dice


class PerExpertLoss(nn.Module):
    """Ensure each expert learns independently"""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, expert_preds, target):
        if not expert_preds:
            return torch.tensor(0.0, device=target.device)

        total_loss = torch.tensor(0.0, device=target.device, dtype=torch.float32)
        for pred in expert_preds:
            if pred.shape[2:] != target.shape[2:]:
                pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

            # Apply sigmoid if logits
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

            # BCE with autocast safety
            with torch.amp.autocast('cuda', enabled=False):
                bce = F.binary_cross_entropy(pred.float(), target.float())
            dice = self.dice(pred, target)
            iou = self.iou(pred, target)

            total_loss += (bce + dice + iou)

        return total_loss / len(expert_preds)


class AntiCollapseLoss(nn.Module):
    """
    CRITICAL: Prevents model from predicting all zeros

    If the model predicts mostly zeros when there's foreground,
    this loss heavily penalizes it.
    """
    def __init__(self, min_activation_ratio=0.3):
        super().__init__()
        self.min_activation_ratio = min_activation_ratio

    def forward(self, pred, target):
        B = pred.shape[0]

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for i in range(B):
            target_fg_ratio = target[i].mean()
            pred_fg_ratio = pred[i].mean()

            # Only apply if target has significant foreground (>1%)
            if target_fg_ratio > 0.01:
                # Minimum prediction should be some fraction of target
                min_expected = target_fg_ratio * self.min_activation_ratio

                # Penalty if prediction is too sparse
                if pred_fg_ratio < min_expected:
                    shortfall = (min_expected - pred_fg_ratio) / (min_expected + 1e-7)
                    total_loss += shortfall ** 2

        return total_loss / B * 5.0  # Scale factor


class CombinedEnhancedLoss(nn.Module):
    """
    Production Loss Configuration for SOTA Performance

    Based on proven SOTA methods:
    - Main: BCE + Dice + IoU + Focal (balanced)
    - Boundary: Light supervision on BPN only
    - Per-expert: Ensures all experts learn
    - Anti-collapse: Prevents degenerate solutions

    TDD/GAD: NO DIRECT SUPERVISION (they provide features to BPN)
    """
    def __init__(
        self,
        seg_weight=1.0,
        boundary_weight=0.3,       # Light boundary supervision
        expert_weight=0.5,         # Important for MoE
        anti_collapse_weight=2.0,  # Critical for stability
        load_balance_weight=0.05,
        # These are kept for compatibility but set to 0
        discontinuity_weight=0.0,  # DISABLED
        hard_mining_weight=0.0,    # DISABLED
    ):
        super().__init__()

        self.seg_weight = seg_weight
        self.boundary_weight = boundary_weight
        self.expert_weight = expert_weight
        self.anti_collapse_weight = anti_collapse_weight
        self.load_balance_weight = load_balance_weight
        self.discontinuity_weight = discontinuity_weight

        # Core losses (proven to work)
        # Note: Using F.binary_cross_entropy instead of nn.BCELoss for autocast safety
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)

        # Auxiliary losses
        self.boundary_loss = BoundaryLoss()
        self.per_expert = PerExpertLoss()
        self.anti_collapse = AntiCollapseLoss(min_activation_ratio=0.3)

        # For logging
        self._last_loss_dict = {}

        print("\n" + "="*70)
        print("PRODUCTION LOSS - SOTA CONFIGURATION")
        print("="*70)
        print(f"  Main Segmentation: {seg_weight} × (BCE + Dice + IoU + Focal)")
        print(f"  Boundary (BPN): {boundary_weight}")
        print(f"  Per-Expert: {expert_weight}")
        print(f"  Anti-Collapse: {anti_collapse_weight} ⭐ CRITICAL")
        print(f"  Load Balance: {load_balance_weight}")
        print(f"  TDD/GAD Direct: DISABLED (features only)")
        print("="*70)
        print("  Expected loss range: 3-6 (start) → 1-2 (converged)")
        print("="*70)

    def forward(self, pred, target, aux_outputs=None, input_image=None):
        """
        Args:
            pred: Main prediction [B, 1, H, W] - can be logits or sigmoid
            target: Ground truth [B, 1, H, W]
            aux_outputs: Dict with auxiliary outputs
            input_image: Original input (unused, for compatibility)
        """
        loss_dict = {}

        # Ensure prediction is sigmoid activated
        if pred.min() < 0 or pred.max() > 1:
            pred_sig = torch.sigmoid(pred)
        else:
            pred_sig = pred
        pred_sig = torch.clamp(pred_sig, 1e-7, 1 - 1e-7)

        # ============================================================
        # 1. MAIN SEGMENTATION LOSS (proven components)
        # ============================================================
        # BCE with autocast safety
        with torch.amp.autocast('cuda', enabled=False):
            bce_loss = F.binary_cross_entropy(pred_sig.float(), target.float())
        dice_loss = self.dice(pred_sig, target)
        iou_loss = self.iou(pred_sig, target)
        focal_loss = self.focal(pred_sig, target)

        seg_loss = bce_loss + dice_loss + iou_loss + focal_loss

        loss_dict['bce'] = bce_loss.item()
        loss_dict['dice'] = dice_loss.item()
        loss_dict['iou'] = iou_loss.item()
        loss_dict['focal'] = focal_loss.item()
        loss_dict['seg_total'] = seg_loss.item()

        total_loss = self.seg_weight * seg_loss

        # ============================================================
        # 2. ANTI-COLLAPSE LOSS (critical for stability)
        # ============================================================
        anti_collapse = self.anti_collapse(pred_sig, target)
        loss_dict['anti_collapse'] = anti_collapse.item()
        total_loss += self.anti_collapse_weight * anti_collapse

        # ============================================================
        # 3. AUXILIARY LOSSES
        # ============================================================
        if aux_outputs is not None:
            # 3a. Boundary loss (BPN output only)
            if self.boundary_weight > 0:
                boundary_pred = aux_outputs.get('boundary')
                if boundary_pred is not None:
                    boundary_loss = self.boundary_loss(boundary_pred, target)
                    loss_dict['boundary'] = boundary_loss.item()
                    total_loss += self.boundary_weight * boundary_loss

            # 3b. Per-expert loss (ensures all experts learn)
            if self.expert_weight > 0:
                expert_preds = aux_outputs.get('individual_expert_preds')
                if expert_preds:
                    expert_loss = self.per_expert(expert_preds, target)
                    loss_dict['expert'] = expert_loss.item()
                    total_loss += self.expert_weight * expert_loss

            # 3c. Load balance loss (router regularization)
            if self.load_balance_weight > 0:
                lb_loss = aux_outputs.get('load_balance_loss')
                if lb_loss is not None and isinstance(lb_loss, torch.Tensor):
                    loss_dict['load_balance'] = lb_loss.item()
                    total_loss += self.load_balance_weight * lb_loss

            # TDD/GAD: NOT SUPERVISED (they just provide features to BPN)
            # Log their outputs for monitoring only
            if 'texture_disc' in aux_outputs and aux_outputs['texture_disc'] is not None:
                loss_dict['tdd_mean'] = aux_outputs['texture_disc'].mean().item()
            if 'gradient_anomaly' in aux_outputs and aux_outputs['gradient_anomaly'] is not None:
                loss_dict['gad_mean'] = aux_outputs['gradient_anomaly'].mean().item()

        loss_dict['total'] = total_loss.item()
        self._last_loss_dict = loss_dict

        return total_loss, loss_dict

    def get_last_loss_dict(self):
        return self._last_loss_dict


# Quick test
if __name__ == '__main__':
    print("Testing Production Loss...")

    criterion = CombinedEnhancedLoss()

    # Normal case
    pred = torch.sigmoid(torch.randn(2, 1, 384, 384))
    target = (torch.rand(2, 1, 384, 384) > 0.7).float()

    aux = {
        'boundary': torch.sigmoid(torch.randn(2, 1, 96, 96)),
        'individual_expert_preds': [torch.randn(2, 1, 384, 384) for _ in range(3)],
        'texture_disc': torch.sigmoid(torch.randn(2, 1, 96, 96)),
        'gradient_anomaly': torch.sigmoid(torch.randn(2, 1, 96, 96)),
        'load_balance_loss': torch.tensor(0.1),
    }

    loss, loss_dict = criterion(pred, target, aux)
    print(f"\nNormal case - Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Collapse case (all zeros)
    pred_zeros = torch.zeros_like(pred) + 0.01
    loss_zeros, _ = criterion(pred_zeros, target, aux)
    print(f"\nCollapse case - Total loss: {loss_zeros.item():.4f} (should be HIGHER)")

    loss.backward()
    print("\n✓ Gradient flow OK")
