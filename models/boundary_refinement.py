"""
Boundary Refinement Module for Camouflaged Object Detection

This module implements boundary-aware refinement with:
1. GradientSupervision - Sobel/Scharr operators for edge detection
2. CascadedRefinement - 3-stage progressive boundary enhancement
3. SignedDistanceMapLoss - Distance-based boundary loss
4. Dynamic lambda scheduling - 1.0 → 4.0 over training
5. Integration with existing model outputs and loss computation

Author: CamoXpert Team
Compatible with: PyTorch 2.0+, CompositeLossSystem
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import math
import numpy as np
from scipy.ndimage import distance_transform_edt


# ============================================================================
# Gradient-Based Supervision
# ============================================================================

class GradientSupervision(nn.Module):
    """
    Gradient-based supervision using Sobel/Scharr operators.

    Computes image gradients for edge detection and boundary supervision.
    Supports both Sobel (3×3) and Scharr (3×3 optimized) operators.

    Args:
        operator: 'sobel' or 'scharr' (default: 'scharr')
        normalize: Normalize gradient magnitude (default: True)
    """
    def __init__(self, operator: str = 'scharr', normalize: bool = True):
        super().__init__()

        self.operator = operator.lower()
        self.normalize = normalize

        # Define gradient kernels
        if self.operator == 'sobel':
            # Sobel operator (3×3)
            kernel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            kernel_y = torch.tensor([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        elif self.operator == 'scharr':
            # Scharr operator (3×3, better rotational symmetry)
            kernel_x = torch.tensor([
                [-3, 0, 3],
                [-10, 0, 10],
                [-3, 0, 3]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            kernel_y = torch.tensor([
                [-3, -10, -3],
                [ 0,   0,  0],
                [ 3,  10,  3]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        else:
            raise ValueError(f"Unknown operator: {operator}. Use 'sobel' or 'scharr'")

        # Register as buffers (non-trainable)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute gradients using Sobel/Scharr operators.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Dictionary containing:
                - 'grad_x': Gradient in x direction [B, C, H, W]
                - 'grad_y': Gradient in y direction [B, C, H, W]
                - 'grad_magnitude': Gradient magnitude [B, C, H, W]
                - 'grad_direction': Gradient direction in radians [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Compute gradients for each channel
        grad_x = F.conv2d(x, self.kernel_x.repeat(C, 1, 1, 1),
                          padding=1, groups=C)
        grad_y = F.conv2d(x, self.kernel_y.repeat(C, 1, 1, 1),
                          padding=1, groups=C)

        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Normalize if requested
        if self.normalize:
            max_val = grad_magnitude.max()
            if max_val > 0:
                grad_magnitude = grad_magnitude / max_val
                grad_x = grad_x / max_val
                grad_y = grad_y / max_val

        # Compute gradient direction (in radians)
        grad_direction = torch.atan2(grad_y, grad_x)

        return {
            'grad_x': grad_x,
            'grad_y': grad_y,
            'grad_magnitude': grad_magnitude,
            'grad_direction': grad_direction
        }

    def compute_gradient_loss(self,
                              pred: torch.Tensor,
                              target: torch.Tensor,
                              weight: float = 1.0) -> torch.Tensor:
        """
        Compute gradient-based loss between prediction and target.

        Args:
            pred: Predicted mask [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]
            weight: Loss weight (default: 1.0)

        Returns:
            Gradient loss scalar
        """
        # Compute gradients
        pred_grads = self.forward(pred)
        target_grads = self.forward(target)

        # L1 loss on gradient magnitude
        loss_mag = F.l1_loss(pred_grads['grad_magnitude'],
                             target_grads['grad_magnitude'])

        # L1 loss on gradient direction (with circular distance)
        direction_diff = torch.abs(pred_grads['grad_direction'] -
                                   target_grads['grad_direction'])
        direction_diff = torch.min(direction_diff, 2 * math.pi - direction_diff)
        loss_dir = direction_diff.mean()

        # Combined loss
        total_loss = weight * (loss_mag + 0.5 * loss_dir)

        return total_loss


# ============================================================================
# Signed Distance Map Loss
# ============================================================================

class SignedDistanceMapLoss(nn.Module):
    """
    Signed Distance Map Loss for boundary-aware training.

    Computes signed distance transform (SDT) and uses it to weight
    boundary errors more heavily than interior errors.

    Args:
        alpha: Weight for boundary region (default: 2.0)
        normalize: Normalize distance maps (default: True)
    """
    def __init__(self, alpha: float = 2.0, normalize: bool = True):
        super().__init__()
        self.alpha = alpha
        self.normalize = normalize

    def compute_sdt(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance transform.

        Args:
            mask: Binary mask [B, 1, H, W] (values in [0, 1])

        Returns:
            Signed distance map [B, 1, H, W]
            - Positive inside object
            - Negative outside object
            - Zero at boundary
        """
        B, C, H, W = mask.shape
        mask_np = mask.detach().cpu().numpy()

        sdt_maps = []
        for b in range(B):
            # Binary mask (threshold at 0.5)
            binary_mask = (mask_np[b, 0] > 0.5).astype(np.uint8)

            # Distance transform for foreground (inside object)
            dist_fg = distance_transform_edt(binary_mask)

            # Distance transform for background (outside object)
            dist_bg = distance_transform_edt(1 - binary_mask)

            # Signed distance: positive inside, negative outside
            sdt = dist_fg - dist_bg

            # Normalize if requested
            if self.normalize:
                max_dist = max(dist_fg.max(), dist_bg.max())
                if max_dist > 0:
                    sdt = sdt / max_dist

            sdt_maps.append(sdt)

        # Convert back to tensor
        sdt_tensor = torch.from_numpy(np.array(sdt_maps)).unsqueeze(1)
        sdt_tensor = sdt_tensor.to(mask.device).float()

        return sdt_tensor

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: float = 1.0) -> torch.Tensor:
        """
        Compute signed distance map loss.

        Args:
            pred: Predicted mask [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]
            weight: Loss weight (default: 1.0)

        Returns:
            SDT loss scalar
        """
        # Compute SDT for target
        target_sdt = self.compute_sdt(target)

        # Compute error map
        error = torch.abs(pred - target)

        # Weight errors based on distance from boundary
        # Errors near boundary (|SDT| small) get higher weight
        boundary_weight = torch.exp(-self.alpha * torch.abs(target_sdt))

        # Weighted error
        weighted_error = error * (1.0 + boundary_weight)

        # Mean loss
        loss = weighted_error.mean()

        return weight * loss


# ============================================================================
# Cascaded Refinement
# ============================================================================

class RefinementStage(nn.Module):
    """
    Single refinement stage with residual connections.

    Args:
        in_channels: Input channels
        mid_channels: Middle layer channels
        out_channels: Output channels
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Boundary attention
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, 1),
            nn.Sigmoid()
        )

        # Residual projection if dimensions change
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through refinement stage.

        Args:
            x: Features [B, C, H, W]
            pred: Previous prediction [B, 1, H, W]

        Returns:
            Tuple of (refined_features, refined_prediction)
        """
        # Concatenate features with previous prediction
        combined = torch.cat([x, pred], dim=1)

        # Feature extraction
        feat = self.conv1(combined)
        feat = self.conv2(feat)
        feat = self.conv3(feat)

        # Boundary attention
        attn = self.boundary_attention(feat)
        feat = feat * attn

        # Residual connection
        if self.residual_proj is not None:
            combined = self.residual_proj(combined)
        feat = feat + combined

        # Predict refinement
        refinement = torch.sigmoid(attn)

        # Refine prediction
        refined_pred = pred + refinement

        return feat, torch.clamp(refined_pred, 0, 1)


class CascadedRefinement(nn.Module):
    """
    Cascaded Refinement with 3 stages of progressive boundary enhancement.

    Stage 1: Coarse refinement (large receptive field)
    Stage 2: Medium refinement (medium receptive field)
    Stage 3: Fine refinement (small receptive field, high precision)

    Args:
        feature_channels: Input feature channels (default: 64)
        hidden_channels: Hidden layer channels (default: 32)
    """
    def __init__(self, feature_channels: int = 64, hidden_channels: int = 32):
        super().__init__()

        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels

        # Stage 1: Coarse refinement
        self.stage1 = RefinementStage(
            in_channels=feature_channels + 1,  # features + prediction
            mid_channels=hidden_channels * 2,
            out_channels=hidden_channels * 2
        )

        # Stage 2: Medium refinement
        self.stage2 = RefinementStage(
            in_channels=hidden_channels * 2 + 1,
            mid_channels=hidden_channels,
            out_channels=hidden_channels
        )

        # Stage 3: Fine refinement
        self.stage3 = RefinementStage(
            in_channels=hidden_channels + 1,
            mid_channels=hidden_channels // 2,
            out_channels=hidden_channels // 2
        )

        # Final prediction head
        self.final_pred = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self,
                features: torch.Tensor,
                initial_pred: torch.Tensor,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cascaded refinement.

        Args:
            features: Input features [B, C, H, W]
            initial_pred: Initial prediction [B, 1, H, W]
            return_intermediate: Return intermediate predictions (default: False)

        Returns:
            Dictionary containing:
                - 'final': Final refined prediction [B, 1, H, W]
                - 'stage1': Stage 1 prediction (if return_intermediate)
                - 'stage2': Stage 2 prediction (if return_intermediate)
                - 'stage3': Stage 3 prediction (if return_intermediate)
        """
        # Resize features to match initial prediction if needed
        if features.shape[-2:] != initial_pred.shape[-2:]:
            features = F.interpolate(features, size=initial_pred.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # Stage 1: Coarse refinement
        feat1, pred1 = self.stage1(features, initial_pred)

        # Stage 2: Medium refinement
        feat2, pred2 = self.stage2(feat1, pred1)

        # Stage 3: Fine refinement
        feat3, pred3 = self.stage3(feat2, pred2)

        # Final prediction
        final_pred = self.final_pred(feat3)

        output = {'final': final_pred}

        if return_intermediate:
            output.update({
                'stage1': pred1,
                'stage2': pred2,
                'stage3': pred3
            })

        return output


# ============================================================================
# Dynamic Lambda Scheduling
# ============================================================================

class DynamicLambdaScheduler:
    """
    Dynamic lambda scheduling for boundary loss weight.

    Schedules lambda from start_lambda to end_lambda over training epochs.

    Args:
        start_lambda: Starting lambda value (default: 1.0)
        end_lambda: Ending lambda value (default: 4.0)
        total_epochs: Total training epochs
        schedule_type: 'linear', 'cosine', or 'exponential' (default: 'cosine')
        warmup_epochs: Number of warmup epochs (default: 5)
    """
    def __init__(self,
                 start_lambda: float = 1.0,
                 end_lambda: float = 4.0,
                 total_epochs: int = 100,
                 schedule_type: str = 'cosine',
                 warmup_epochs: int = 5):
        self.start_lambda = start_lambda
        self.end_lambda = end_lambda
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type.lower()
        self.warmup_epochs = warmup_epochs

    def get_lambda(self, current_epoch: int) -> float:
        """
        Get lambda value for current epoch.

        Args:
            current_epoch: Current training epoch (0-indexed)

        Returns:
            Lambda value for boundary loss
        """
        # Warmup phase: use start_lambda
        if current_epoch < self.warmup_epochs:
            return self.start_lambda

        # Adjust epoch for post-warmup scheduling
        adjusted_epoch = current_epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs

        if adjusted_total <= 0:
            return self.end_lambda

        progress = adjusted_epoch / adjusted_total

        if self.schedule_type == 'linear':
            # Linear interpolation
            lambda_val = self.start_lambda + \
                        (self.end_lambda - self.start_lambda) * progress

        elif self.schedule_type == 'cosine':
            # Cosine annealing (smooth increase)
            lambda_val = self.start_lambda + \
                        (self.end_lambda - self.start_lambda) * \
                        (1 - math.cos(progress * math.pi)) / 2

        elif self.schedule_type == 'exponential':
            # Exponential increase
            lambda_val = self.start_lambda * \
                        (self.end_lambda / self.start_lambda) ** progress

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return min(lambda_val, self.end_lambda)


# ============================================================================
# Main Boundary Refinement Module
# ============================================================================

class BoundaryRefinementModule(nn.Module):
    """
    Complete Boundary Refinement Module integrating all components.

    Features:
    - GradientSupervision with Sobel/Scharr operators
    - CascadedRefinement with 3 progressive stages
    - SignedDistanceMapLoss for boundary-aware training
    - Dynamic lambda scheduling (1.0 → 4.0)
    - Compatible with existing model outputs and loss computation

    Args:
        feature_channels: Feature channels from model (default: 64)
        use_gradient_loss: Enable gradient supervision (default: True)
        use_sdt_loss: Enable SDT loss (default: True)
        gradient_weight: Base weight for gradient loss (default: 0.5)
        sdt_weight: Base weight for SDT loss (default: 1.0)
        total_epochs: Total training epochs for lambda scheduling (default: 100)
        lambda_schedule_type: 'linear', 'cosine', or 'exponential' (default: 'cosine')
    """
    def __init__(self,
                 feature_channels: int = 64,
                 use_gradient_loss: bool = True,
                 use_sdt_loss: bool = True,
                 gradient_weight: float = 0.5,
                 sdt_weight: float = 1.0,
                 total_epochs: int = 100,
                 lambda_schedule_type: str = 'cosine'):
        super().__init__()

        self.feature_channels = feature_channels
        self.use_gradient_loss = use_gradient_loss
        self.use_sdt_loss = use_sdt_loss
        self.gradient_weight = gradient_weight
        self.sdt_weight = sdt_weight

        # Cascaded refinement module
        self.refinement = CascadedRefinement(
            feature_channels=feature_channels,
            hidden_channels=32
        )

        # Gradient supervision
        if use_gradient_loss:
            self.gradient_supervision = GradientSupervision(
                operator='scharr',
                normalize=True
            )

        # Signed distance map loss
        if use_sdt_loss:
            self.sdt_loss = SignedDistanceMapLoss(
                alpha=2.0,
                normalize=True
            )

        # Lambda scheduler
        self.lambda_scheduler = DynamicLambdaScheduler(
            start_lambda=1.0,
            end_lambda=4.0,
            total_epochs=total_epochs,
            schedule_type=lambda_schedule_type,
            warmup_epochs=5
        )

        # Track current epoch
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Set current training epoch for lambda scheduling."""
        self.current_epoch = epoch

    def forward(self,
                features: torch.Tensor,
                initial_pred: torch.Tensor,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through boundary refinement.

        Args:
            features: Features from model [B, C, H, W]
            initial_pred: Initial prediction [B, 1, H, W]
            return_intermediate: Return intermediate outputs (default: False)

        Returns:
            Dictionary containing refined predictions
        """
        # Apply cascaded refinement
        refined_outputs = self.refinement(
            features,
            initial_pred,
            return_intermediate=return_intermediate
        )

        return refined_outputs

    def compute_boundary_loss(self,
                              pred: torch.Tensor,
                              target: torch.Tensor,
                              intermediate_preds: Optional[List[torch.Tensor]] = None,
                              ) -> Dict[str, torch.Tensor]:
        """
        Compute boundary-specific losses.

        Args:
            pred: Final refined prediction [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]
            intermediate_preds: List of intermediate predictions (optional)

        Returns:
            Dictionary with loss components
        """
        losses = {}
        total_loss = 0.0

        # Get current lambda
        current_lambda = self.lambda_scheduler.get_lambda(self.current_epoch)

        # Gradient loss
        if self.use_gradient_loss:
            grad_loss = self.gradient_supervision.compute_gradient_loss(
                pred, target, weight=self.gradient_weight
            )
            losses['gradient_loss'] = grad_loss
            total_loss += current_lambda * grad_loss

        # SDT loss
        if self.use_sdt_loss:
            sdt_loss = self.sdt_loss(pred, target, weight=self.sdt_weight)
            losses['sdt_loss'] = sdt_loss
            total_loss += current_lambda * sdt_loss

        # Deep supervision on intermediate predictions
        if intermediate_preds is not None:
            stage_weights = [0.4, 0.3, 0.2]  # Decreasing weights for earlier stages
            for i, (stage_pred, stage_weight) in enumerate(zip(intermediate_preds, stage_weights)):
                # Resize if needed
                if stage_pred.shape != target.shape:
                    stage_pred = F.interpolate(stage_pred, size=target.shape[-2:],
                                              mode='bilinear', align_corners=False)

                # BCE loss for intermediate stages
                stage_loss = F.binary_cross_entropy(stage_pred, target)
                losses[f'stage{i+1}_loss'] = stage_loss
                total_loss += stage_weight * stage_loss

        losses['total_boundary_loss'] = total_loss
        losses['current_lambda'] = torch.tensor(current_lambda)

        return losses


# ============================================================================
# Integration Helper
# ============================================================================

class BoundaryRefinementWrapper(nn.Module):
    """
    Wrapper to integrate BoundaryRefinementModule with existing models.

    Automatically applies boundary refinement to model predictions.

    Args:
        model: Base model (e.g., ModelLevelMoE)
        feature_channels: Feature channels from model decoder (default: 64)
        enable_refinement: Enable refinement during forward (default: True)
    """
    def __init__(self,
                 model: nn.Module,
                 feature_channels: int = 64,
                 enable_refinement: bool = True,
                 **refinement_kwargs):
        super().__init__()

        self.model = model
        self.enable_refinement = enable_refinement

        # Boundary refinement module
        self.boundary_refinement = BoundaryRefinementModule(
            feature_channels=feature_channels,
            **refinement_kwargs
        )

    def forward(self,
                x: torch.Tensor,
                return_features: bool = False,
                return_refined: bool = None):
        """
        Forward pass with optional boundary refinement.

        Args:
            x: Input images [B, 3, H, W]
            return_features: Return intermediate features (default: False)
            return_refined: Enable refinement (default: self.enable_refinement)

        Returns:
            Predictions (optionally refined)
        """
        if return_refined is None:
            return_refined = self.enable_refinement

        # Forward through base model
        # This assumes model returns (prediction, features) or just prediction
        model_output = self.model(x)

        if isinstance(model_output, tuple):
            prediction, features = model_output
        else:
            prediction = model_output
            features = None

        # Apply boundary refinement if enabled
        if return_refined and features is not None:
            refined_output = self.boundary_refinement(
                features,
                prediction,
                return_intermediate=self.training
            )
            prediction = refined_output['final']

            if return_features:
                return prediction, features, refined_output
            return prediction

        if return_features:
            return prediction, features
        return prediction


# ============================================================================
# Testing
# ============================================================================

def test_boundary_refinement():
    """Test BoundaryRefinementModule components."""
    print("=" * 80)
    print("Testing Boundary Refinement Module")
    print("=" * 80)

    # Test 1: Gradient Supervision
    print("\n1. Testing GradientSupervision...")
    grad_sup = GradientSupervision(operator='scharr')
    x = torch.randn(2, 1, 64, 64)
    grads = grad_sup(x)
    print(f"   Input: {x.shape}")
    print(f"   Gradient magnitude: {grads['grad_magnitude'].shape}")
    print(f"   Gradient direction: {grads['grad_direction'].shape}")

    # Test gradient loss
    pred = torch.rand(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    grad_loss = grad_sup.compute_gradient_loss(pred, target)
    print(f"   Gradient loss: {grad_loss.item():.4f}")

    # Test 2: Signed Distance Map Loss
    print("\n2. Testing SignedDistanceMapLoss...")
    sdt_loss_module = SignedDistanceMapLoss(alpha=2.0)
    sdt_loss = sdt_loss_module(pred, target)
    print(f"   SDT loss: {sdt_loss.item():.4f}")

    # Test 3: Cascaded Refinement
    print("\n3. Testing CascadedRefinement...")
    cascaded = CascadedRefinement(feature_channels=64, hidden_channels=32)
    features = torch.randn(2, 64, 64, 64)
    initial_pred = torch.rand(2, 1, 64, 64)
    refined = cascaded(features, initial_pred, return_intermediate=True)
    print(f"   Input features: {features.shape}")
    print(f"   Initial prediction: {initial_pred.shape}")
    print(f"   Final refined: {refined['final'].shape}")
    print(f"   Intermediate stages: {len([k for k in refined.keys() if k.startswith('stage')])}")

    # Test 4: Lambda Scheduler
    print("\n4. Testing DynamicLambdaScheduler...")
    scheduler = DynamicLambdaScheduler(
        start_lambda=1.0,
        end_lambda=4.0,
        total_epochs=100,
        schedule_type='cosine'
    )
    print("   Epoch -> Lambda:")
    for epoch in [0, 10, 25, 50, 75, 99]:
        lambda_val = scheduler.get_lambda(epoch)
        print(f"     {epoch:3d} -> {lambda_val:.3f}")

    # Test 5: Complete Module
    print("\n5. Testing BoundaryRefinementModule...")
    boundary_module = BoundaryRefinementModule(
        feature_channels=64,
        use_gradient_loss=True,
        use_sdt_loss=True,
        total_epochs=100
    )

    # Forward pass
    refined_outputs = boundary_module(features, initial_pred, return_intermediate=True)
    print(f"   Final prediction: {refined_outputs['final'].shape}")

    # Compute losses
    boundary_module.set_epoch(50)
    losses = boundary_module.compute_boundary_loss(
        refined_outputs['final'],
        target,
        intermediate_preds=[
            refined_outputs['stage1'],
            refined_outputs['stage2'],
            refined_outputs['stage3']
        ]
    )
    print(f"   Boundary losses:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in boundary_module.parameters())
    print(f"\n   Total parameters: {total_params:,}")

    print("\n✓ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_boundary_refinement()
