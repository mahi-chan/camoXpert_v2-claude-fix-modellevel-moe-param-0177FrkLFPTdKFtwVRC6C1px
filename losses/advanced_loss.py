"""
Advanced Loss Function for Camouflaged Object Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedCODLoss(nn.Module):
    """
    Combined loss for COD:
    - BCE with Logits (safe with AMP)
    - IoU Loss
    - Edge-aware Loss
    - Auxiliary MoE Loss

    PERFORMANCE OPTIMIZED: Cache sobel filters to avoid recreating on every forward pass
    """

    def __init__(self, bce_weight=1.0, iou_weight=1.0, edge_weight=0.5, aux_weight=0.1):
        super().__init__()

        # Use BCEWithLogitsLoss (safe with autocast)
        self.bce = nn.BCEWithLogitsLoss()

        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.edge_weight = edge_weight
        self.aux_weight = aux_weight

        # PERFORMANCE FIX: Register sobel kernels as buffers
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Cache for FP16 versions (avoid .to() overhead every forward)
        self._sobel_x_fp16 = None
        self._sobel_y_fp16 = None

    def iou_loss(self, pred, target):
        """IoU loss"""
        # Clamp logits to prevent NaN in mixed precision
        pred = torch.clamp(pred, min=-15, max=15)
        pred = torch.sigmoid(pred)  # Apply sigmoid here

        smooth = 1e-5
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)

        return 1 - iou.mean()

    def edge_loss(self, pred, target):
        """Edge-aware loss - OPTIMIZED: uses cached sobel filters"""
        # Clamp logits to prevent NaN in mixed precision
        pred = torch.clamp(pred, min=-15, max=15)
        pred = torch.sigmoid(pred)  # Apply sigmoid here

        # Use cached FP16 versions to avoid .to() overhead every forward
        # Cache per device to handle DDP (each rank has different device)
        if pred.dtype == torch.float16:
            if self._sobel_x_fp16 is None or self._sobel_x_fp16.device != pred.device:
                # Move buffers to correct device first, then convert to half
                sobel_x_base = self.sobel_x.to(pred.device)
                sobel_y_base = self.sobel_y.to(pred.device)
                self._sobel_x_fp16 = sobel_x_base.half()
                self._sobel_y_fp16 = sobel_y_base.half()
            sobel_x = self._sobel_x_fp16
            sobel_y = self._sobel_y_fp16
        else:
            # Ensure buffers are on correct device for FP32
            sobel_x = self.sobel_x.to(pred.device)
            sobel_y = self.sobel_y.to(pred.device)

        # Use cached sobel filters (no tensor creation overhead!)
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        # Ensure non-negative before sqrt
        pred_edge = torch.sqrt(torch.clamp(pred_edge_x ** 2 + pred_edge_y ** 2, min=0) + 1e-5)

        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-5)

        return F.mse_loss(pred_edge, target_edge)

    def forward(self, pred, target, aux_loss=None, deep_outputs=None):
        """
        Args:
            pred: Main prediction (logits, before sigmoid)
            target: Ground truth masks
            aux_loss: Auxiliary MoE loss
            deep_outputs: Deep supervision outputs (list of logits)
        """

        # Clamp main prediction logits to prevent NaN in mixed precision
        pred = torch.clamp(pred, min=-15, max=15)

        # Main losses (pred is logits)
        bce = self.bce(pred, target)
        iou = self.iou_loss(pred, target)  # IoU handles sigmoid internally
        edge = self.edge_loss(pred, target)  # Edge handles sigmoid internally

        total_loss = (
                self.bce_weight * bce +
                self.iou_weight * iou +
                self.edge_weight * edge
        )

        loss_dict = {
            'bce': bce.item(),
            'iou': iou.item(),
            'edge': edge.item()
        }

        # Add auxiliary MoE loss
        if aux_loss is not None:
            # Handle DataParallel case where aux_loss might be a vector [num_gpus]
            # DataParallel gathers scalar losses from each GPU into a tensor
            if aux_loss.dim() > 0:
                aux_loss = aux_loss.mean()
            total_loss += self.aux_weight * aux_loss
            loss_dict['aux'] = aux_loss.item()

        # Add deep supervision losses
        if deep_outputs is not None:
            deep_loss = 0
            for i, deep_pred in enumerate(deep_outputs):
                # Clamp deep supervision logits to prevent NaN
                deep_pred = torch.clamp(deep_pred, min=-15, max=15)

                # Resize target to match deep_pred size
                if deep_pred.shape[2:] != target.shape[2:]:
                    target_resized = F.interpolate(target, size=deep_pred.shape[2:],
                                                   mode='bilinear', align_corners=False)
                else:
                    target_resized = target

                deep_loss += self.bce(deep_pred, target_resized)

            deep_loss /= len(deep_outputs)
            total_loss += 0.4 * deep_loss
            loss_dict['deep'] = deep_loss.item()

        return total_loss, loss_dict


class CODSpecializedLoss(nn.Module):
    """
    100% COD-Specialized Loss Function
    Includes boundary-aware, uncertainty-aware, and reverse attention losses

    PERFORMANCE OPTIMIZED: Cache sobel/boundary filters to avoid recreating on every forward pass
    """

    def __init__(self, bce_weight=5.0, iou_weight=3.0, edge_weight=2.0, boundary_weight=3.0,
                 uncertainty_weight=0.5, reverse_attention_weight=1.0, aux_weight=0.1):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.edge_weight = edge_weight
        self.boundary_weight = boundary_weight
        self.uncertainty_weight = uncertainty_weight
        self.reverse_attention_weight = reverse_attention_weight
        self.aux_weight = aux_weight

        # PERFORMANCE FIX: Register sobel and boundary kernels as buffers
        # This avoids recreating tensors on every forward pass
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        boundary_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('boundary_kernel', boundary_kernel)

        # Cache for FP16 versions (avoid .to() overhead every forward)
        self._sobel_x_fp16 = None
        self._sobel_y_fp16 = None
        self._boundary_kernel_fp16 = None

    def iou_loss(self, pred, target):
        """IoU loss"""
        pred = torch.clamp(pred, min=-15, max=15)
        pred = torch.sigmoid(pred)

        smooth = 1e-5
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)

        return 1 - iou.mean()

    def edge_loss(self, pred, target):
        """Edge-aware loss - OPTIMIZED: uses cached sobel filters"""
        pred = torch.clamp(pred, min=-15, max=15)
        pred = torch.sigmoid(pred)

        # Use cached FP16 versions to avoid .to() overhead every forward
        # Cache per device to handle DDP (each rank has different device)
        if pred.dtype == torch.float16:
            if self._sobel_x_fp16 is None or self._sobel_x_fp16.device != pred.device:
                # Move buffers to correct device first, then convert to half
                sobel_x_base = self.sobel_x.to(pred.device)
                sobel_y_base = self.sobel_y.to(pred.device)
                self._sobel_x_fp16 = sobel_x_base.half()
                self._sobel_y_fp16 = sobel_y_base.half()
            sobel_x = self._sobel_x_fp16
            sobel_y = self._sobel_y_fp16
        else:
            # Ensure buffers are on correct device for FP32
            sobel_x = self.sobel_x.to(pred.device)
            sobel_y = self.sobel_y.to(pred.device)

        # Use cached sobel filters (no tensor creation overhead!)
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(torch.clamp(pred_edge_x ** 2 + pred_edge_y ** 2, min=0) + 1e-5)

        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-5)

        return F.mse_loss(pred_edge, target_edge)

    def boundary_loss(self, pred, target):
        """
        Boundary-aware loss: Focus heavily on boundary regions
        Boundaries are the hardest part in COD
        OPTIMIZED: uses cached boundary kernel
        """
        pred = torch.clamp(pred, min=-15, max=15)

        # Use cached FP16 version to avoid .to() overhead every forward
        # Cache per device to handle DDP (each rank has different device)
        if target.dtype == torch.float16:
            if self._boundary_kernel_fp16 is None or self._boundary_kernel_fp16.device != target.device:
                # Move buffer to correct device first, then convert to half
                boundary_kernel_base = self.boundary_kernel.to(target.device)
                self._boundary_kernel_fp16 = boundary_kernel_base.half()
            boundary_kernel = self._boundary_kernel_fp16
        else:
            # Ensure buffer is on correct device for FP32
            boundary_kernel = self.boundary_kernel.to(target.device)

        # Extract boundaries using morphological operations (use cached kernel!)
        dilated = F.conv2d(target, boundary_kernel, padding=1)
        eroded = 1 - F.conv2d(1 - target, boundary_kernel, padding=1)
        boundary = dilated - eroded

        # Higher weight on boundaries (5x)
        weight = 1.0 + boundary * 4.0

        # Weighted BCE
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_loss = (loss * weight).mean()

        return weighted_loss

    def uncertainty_loss(self, uncertainty, pred, target):
        """
        Uncertainty-aware loss
        Penalize low uncertainty when prediction is wrong
        Encourage high uncertainty at difficult boundaries
        """
        pred = torch.clamp(pred, min=-15, max=15)
        pred_prob = torch.sigmoid(pred)
        error = torch.abs(pred_prob - target)

        # Should be uncertain when error is high
        # Use log to avoid extreme values
        uncertainty_penalty = error * torch.log(1.0 / (uncertainty + 1e-8))

        return uncertainty_penalty.mean()

    def reverse_attention_loss(self, fg_map, target):
        """
        Reverse attention loss: Foreground map should match target
        fg_map is already in [0, 1] range (from sigmoid), so we need to
        disable autocast for numerical stability with AMP
        """
        # Clamp to valid probability range and add epsilon for numerical stability
        fg_map = torch.clamp(fg_map, min=1e-7, max=1.0 - 1e-7)

        # Disable autocast for this operation (AMP-safe workaround)
        with torch.cuda.amp.autocast(enabled=False):
            # Convert to fp32 for stable BCE computation
            fg_map = fg_map.float()
            target = target.float()
            return F.binary_cross_entropy(fg_map, target)

    def forward(self, pred, target, aux_loss=None, deep_outputs=None,
                uncertainty=None, fg_map=None, refinements=None, search_map=None):
        """
        Args:
            pred: Main prediction logits [B, 1, H, W]
            target: Ground truth [B, 1, H, W]
            aux_loss: MoE auxiliary loss
            deep_outputs: Deep supervision outputs (list of logits)
            uncertainty: Uncertainty map [B, 1, H, W]
            fg_map: Foreground map from reverse attention [B, 1, H, W]
            refinements: List of refined predictions from iterative refinement
            search_map: Search map from search module [B, 1, H, W]
        """
        pred = torch.clamp(pred, min=-15, max=15)

        # Main losses
        bce = self.bce(pred, target)
        iou = self.iou_loss(pred, target)
        edge = self.edge_loss(pred, target)
        boundary = self.boundary_loss(pred, target)

        total_loss = (
            self.bce_weight * bce +
            self.iou_weight * iou +
            self.edge_weight * edge +
            self.boundary_weight * boundary
        )

        loss_dict = {
            'bce': bce.item(),
            'iou': iou.item(),
            'edge': edge.item(),
            'boundary': boundary.item()
        }

        # Uncertainty loss
        if uncertainty is not None:
            uncert_loss = self.uncertainty_loss(uncertainty, pred, target)
            total_loss += self.uncertainty_weight * uncert_loss
            loss_dict['uncertainty'] = uncert_loss.item()

        # Reverse attention loss
        if fg_map is not None:
            ra_loss = self.reverse_attention_loss(fg_map, target)
            total_loss += self.reverse_attention_weight * ra_loss
            loss_dict['reverse_attention'] = ra_loss.item()

        # Search map loss (search map should highlight object regions)
        if search_map is not None:
            # Clamp to valid probability range (search_map is already sigmoid'd)
            search_map_clamped = torch.clamp(search_map, min=1e-7, max=1.0 - 1e-7)

            # Disable autocast for AMP compatibility
            with torch.cuda.amp.autocast(enabled=False):
                search_map_clamped = search_map_clamped.float()
                target_float = target.float()
                search_loss = F.binary_cross_entropy(search_map_clamped, target_float)

            total_loss += 0.5 * search_loss
            loss_dict['search'] = search_loss.item()

        # Auxiliary MoE loss
        if aux_loss is not None:
            if aux_loss.dim() > 0:
                aux_loss = aux_loss.mean()
            total_loss += self.aux_weight * aux_loss
            loss_dict['aux'] = aux_loss.item()

        # Deep supervision losses
        if deep_outputs is not None:
            deep_loss = 0
            for i, deep_pred in enumerate(deep_outputs):
                deep_pred = torch.clamp(deep_pred, min=-15, max=15)
                if deep_pred.shape[2:] != target.shape[2:]:
                    target_resized = F.interpolate(target, size=deep_pred.shape[2:],
                                                   mode='bilinear', align_corners=False)
                else:
                    target_resized = target
                deep_loss += self.bce(deep_pred, target_resized)

            deep_loss /= len(deep_outputs)
            total_loss += 0.4 * deep_loss
            loss_dict['deep'] = deep_loss.item()

        # Iterative refinement losses (decreasing weight)
        if refinements is not None:
            refinement_loss = 0
            for i, ref_pred in enumerate(refinements):
                ref_pred = torch.clamp(ref_pred, min=-15, max=15)
                weight = 0.3 / (i + 1)  # Decreasing weight for later refinements
                refinement_loss += weight * self.bce(ref_pred, target)

            total_loss += refinement_loss
            loss_dict['refinement'] = refinement_loss.item()

        return total_loss, loss_dict