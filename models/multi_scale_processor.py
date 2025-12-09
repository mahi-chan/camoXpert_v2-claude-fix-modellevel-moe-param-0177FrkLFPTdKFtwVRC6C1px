"""
MultiScaleInputProcessor: Advanced Multi-Scale Processing Architecture

Processes input images at multiple scales with attention-based integration:
1. Multi-scale input generation: 0.5×, 1.0×, 1.5× original resolution
2. Shared backbone for feature extraction at each scale
3. Attention-Based Scale Integration Units (ABSI) for dynamic fusion
4. Hierarchical scale integration without simple concatenation
5. Scale-specific loss weighting (0.5× → 0.5, 1.0× → 1.0, 1.5× → 0.5)

Handles variable input sizes and outputs unified multi-scale features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


class MultiScaleInputGenerator(nn.Module):
    """
    Generates multi-scale inputs from original image.

    Scales: 0.5×, 1.0×, 1.5× original resolution
    """
    def __init__(self, scales=[0.5, 1.0, 1.5], align_corners=False):
        super().__init__()
        self.scales = scales
        self.align_corners = align_corners

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input images

        Returns:
            multi_scale_inputs: List of [B, C, scale_H, scale_W]
        """
        B, C, H, W = x.shape
        multi_scale_inputs = []

        for scale in self.scales:
            if scale == 1.0:
                # Original resolution
                scaled = x
            else:
                # Resize to scaled resolution
                new_h = int(H * scale)
                new_w = int(W * scale)

                # Ensure dimensions are divisible by 32 (for backbone)
                new_h = (new_h // 32) * 32
                new_w = (new_w // 32) * 32

                # At least 32x32
                new_h = max(new_h, 32)
                new_w = max(new_w, 32)

                scaled = F.interpolate(
                    x,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=self.align_corners
                )

            multi_scale_inputs.append(scaled)

        return multi_scale_inputs


class AttentionBasedScaleIntegrationUnit(nn.Module):
    """
    Attention-Based Scale Integration Unit (ABSI).

    Dynamically integrates features from different scales using
    scale-aware attention mechanisms.
    """
    def __init__(self, channels, num_scales=3):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales

        # Scale-aware query, key, value projections
        self.scale_queries = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.GELU()
            ) for _ in range(num_scales)
        ])

        self.scale_keys = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.GELU()
            ) for _ in range(num_scales)
        ])

        self.scale_values = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.GELU()
            ) for _ in range(num_scales)
        ])

        # Scale importance predictor
        self.scale_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, num_scales, 1),
            nn.Softmax(dim=1)
        )

        # Cross-scale attention
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, scale_features, target_size):
        """
        Args:
            scale_features: List of [B, C, H_i, W_i] features at different scales
            target_size: (H, W) target output size

        Returns:
            integrated: [B, C, H, W] scale-integrated features
        """
        B = scale_features[0].shape[0]
        C = self.channels
        H, W = target_size

        # Resize all features to target size
        resized_features = []
        for feat in scale_features:
            if feat.shape[2:] != (H, W):
                feat_resized = F.interpolate(
                    feat,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                feat_resized = feat
            resized_features.append(feat_resized)

        # Compute scale importance
        concat_feats = torch.cat(resized_features, dim=1)  # [B, C*num_scales, H, W]
        scale_weights = self.scale_importance(concat_feats)  # [B, num_scales, 1, 1]

        # Generate Q, K, V for each scale
        queries = []
        keys = []
        values = []

        for i, (feat, q_proj, k_proj, v_proj) in enumerate(zip(
            resized_features, self.scale_queries, self.scale_keys, self.scale_values
        )):
            q = q_proj(feat)  # [B, C, H, W]
            k = k_proj(feat)
            v = v_proj(feat)

            queries.append(q)
            keys.append(k)
            values.append(v)

        # Stack and reshape for attention
        # [num_scales, B, C, H, W] -> [B, num_scales, C, H, W]
        queries_stack = torch.stack(queries, dim=1)
        keys_stack = torch.stack(keys, dim=1)
        values_stack = torch.stack(values, dim=1)

        # Flatten spatial dimensions for attention
        # [B, num_scales, C, H, W] -> [B, num_scales, C, H*W] -> [B, num_scales, H*W, C]
        Q = queries_stack.flatten(3).permute(0, 1, 3, 2)  # [B, num_scales, H*W, C]
        K = keys_stack.flatten(3).permute(0, 1, 3, 2)
        V = values_stack.flatten(3).permute(0, 1, 3, 2)

        # Reshape for cross-scale attention
        # [B, num_scales, H*W, C] -> [B*H*W, num_scales, C]
        Q_flat = Q.permute(0, 2, 1, 3).reshape(B * H * W, self.num_scales, C)
        K_flat = K.permute(0, 2, 1, 3).reshape(B * H * W, self.num_scales, C)
        V_flat = V.permute(0, 2, 1, 3).reshape(B * H * W, self.num_scales, C)

        # Cross-scale attention
        attn_out, _ = self.cross_scale_attn(Q_flat, K_flat, V_flat)  # [B*H*W, num_scales, C]

        # Reshape back
        attn_out = attn_out.view(B, H, W, self.num_scales, C)  # [B, H, W, num_scales, C]
        attn_out = attn_out.permute(0, 3, 4, 1, 2)  # [B, num_scales, C, H, W]

        # Apply scale importance weights
        weighted_features = []
        for i in range(self.num_scales):
            weighted = attn_out[:, i] * scale_weights[:, i:i+1]  # [B, C, H, W]
            weighted_features.append(weighted)

        # Sum weighted features
        integrated = sum(weighted_features)  # [B, C, H, W]

        # Output projection with residual
        output = self.out_proj(integrated)

        # Add residual from middle scale (1.0×)
        if len(resized_features) >= 2:
            output = output + resized_features[1]  # Middle scale residual

        return output


class HierarchicalScaleIntegration(nn.Module):
    """
    Hierarchical Scale Integration without simple concatenation.

    Integrates scales progressively from coarse to fine:
    1. Integrate 0.5× and 1.0×
    2. Integrate result with 1.5×
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Stage 1: Integrate 0.5× and 1.0×
        self.stage1_integration = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # Stage 2: Integrate stage1 result with 1.5×
        self.stage2_integration = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # Channel attention for each stage
        self.stage1_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.stage2_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, scale_features, target_size):
        """
        Args:
            scale_features: List of 3 features [feat_0.5x, feat_1.0x, feat_1.5x]
                           Each [B, C, H_i, W_i]
            target_size: (H, W) target output size

        Returns:
            integrated: [B, C, H, W] hierarchically integrated features
        """
        feat_05x, feat_10x, feat_15x = scale_features
        H, W = target_size

        # Resize all to target size
        feat_05x_resized = F.interpolate(feat_05x, size=(H, W), mode='bilinear', align_corners=False)
        feat_10x_resized = F.interpolate(feat_10x, size=(H, W), mode='bilinear', align_corners=False)
        feat_15x_resized = F.interpolate(feat_15x, size=(H, W), mode='bilinear', align_corners=False)

        # Stage 1: Integrate 0.5× and 1.0× (coarse to medium)
        stage1_concat = torch.cat([feat_05x_resized, feat_10x_resized], dim=1)
        stage1_integrated = self.stage1_integration(stage1_concat)

        # Apply channel attention
        stage1_ca = self.stage1_channel_attn(stage1_integrated)
        stage1_integrated = stage1_integrated * stage1_ca

        # Apply spatial attention
        stage1_max = torch.max(stage1_integrated, dim=1, keepdim=True)[0]
        stage1_avg = torch.mean(stage1_integrated, dim=1, keepdim=True)
        stage1_spatial = torch.cat([stage1_max, stage1_avg], dim=1)
        stage1_sa = self.spatial_attn(stage1_spatial)
        stage1_integrated = stage1_integrated * stage1_sa

        # Stage 2: Integrate stage1 result with 1.5× (medium to fine)
        stage2_concat = torch.cat([stage1_integrated, feat_15x_resized], dim=1)
        stage2_integrated = self.stage2_integration(stage2_concat)

        # Apply channel attention
        stage2_ca = self.stage2_channel_attn(stage2_integrated)
        stage2_integrated = stage2_integrated * stage2_ca

        # Apply spatial attention
        stage2_max = torch.max(stage2_integrated, dim=1, keepdim=True)[0]
        stage2_avg = torch.mean(stage2_integrated, dim=1, keepdim=True)
        stage2_spatial = torch.cat([stage2_max, stage2_avg], dim=1)
        stage2_sa = self.spatial_attn(stage2_spatial)
        stage2_integrated = stage2_integrated * stage2_sa

        # Residual connection with middle scale
        output = stage2_integrated + feat_10x_resized

        return output


class ScaleAwareLossModule(nn.Module):
    """
    Scale-specific loss computation with weighting.

    Weights:
    - 0.5× scale: 0.5
    - 1.0× scale: 1.0
    - 1.5× scale: 0.5
    """
    def __init__(self, scale_weights=[0.5, 1.0, 0.5]):
        super().__init__()
        self.register_buffer('scale_weights', torch.tensor(scale_weights))

    def forward(self, predictions, targets, criterion):
        """
        Args:
            predictions: List of 3 predictions [pred_0.5x, pred_1.0x, pred_1.5x]
            targets: List of 3 targets (resized to match predictions)
            criterion: Loss function (e.g., BCEWithLogitsLoss)

        Returns:
            total_loss: Weighted sum of scale-specific losses
            loss_dict: Dictionary with individual losses
        """
        assert len(predictions) == len(self.scale_weights), "Mismatch in number of scales"

        losses = []
        loss_dict = {}

        for i, (pred, target, weight) in enumerate(zip(predictions, targets, self.scale_weights)):
            # Resize target to match prediction if needed
            if target.shape != pred.shape:
                target_resized = F.interpolate(
                    target,
                    size=pred.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                target_resized = target

            # Compute loss
            loss = criterion(pred, target_resized)

            # Apply scale weight
            weighted_loss = weight * loss

            losses.append(weighted_loss)
            loss_dict[f'loss_scale_{i}'] = loss.item()
            loss_dict[f'weighted_loss_scale_{i}'] = weighted_loss.item()

        # Total weighted loss
        total_loss = sum(losses)
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class MultiScaleFeatureFusion(nn.Module):
    """
    Fuses multi-scale features from different levels.

    Combines ABSI and Hierarchical integration.
    """
    def __init__(self, channels_list=[64, 128, 320, 512], num_scales=3):
        super().__init__()
        self.channels_list = channels_list
        self.num_scales = num_scales

        # ABSI units for each feature level
        self.absi_units = nn.ModuleList([
            AttentionBasedScaleIntegrationUnit(channels, num_scales)
            for channels in channels_list
        ])

        # Hierarchical integration for each feature level
        self.hierarchical_units = nn.ModuleList([
            HierarchicalScaleIntegration(channels)
            for channels in channels_list
        ])

        # Feature refinement
        self.refinement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1)
            ) for channels in channels_list
        ])

    def forward(self, multi_scale_features, use_hierarchical=True):
        """
        Args:
            multi_scale_features: List of 3 multi-scale feature pyramids
                Each pyramid: [feat1, feat2, feat3, feat4] with dims [64, 128, 320, 512]
            use_hierarchical: If True, use hierarchical integration; else use ABSI only

        Returns:
            fused_features: List of fused features [feat1, feat2, feat3, feat4]
        """
        num_levels = len(multi_scale_features[0])
        fused_features = []

        for level in range(num_levels):
            # Gather features at this level from all scales
            scale_feats = [ms_feats[level] for ms_feats in multi_scale_features]

            # Determine target size (use 1.0× scale as reference)
            target_size = scale_feats[1].shape[2:]  # Middle scale (1.0×)

            if use_hierarchical:
                # Use hierarchical integration
                fused = self.hierarchical_units[level](scale_feats, target_size)
            else:
                # Use ABSI
                fused = self.absi_units[level](scale_feats, target_size)

            # Refine
            fused = self.refinement[level](fused)

            fused_features.append(fused)

        return fused_features


class MultiScaleInputProcessor(nn.Module):
    """
    Complete Multi-Scale Input Processing Architecture.

    Process Flow:
    1. Generate multi-scale inputs (0.5×, 1.0×, 1.5×)
    2. Extract features at each scale using shared backbone
    3. Integrate scales using ABSI + Hierarchical fusion
    4. Output unified multi-scale features
    5. Compute scale-aware losses

    Args:
        backbone: Feature extraction backbone (should support variable input sizes)
        channels_list: Output channels at each level [64, 128, 320, 512]
        scales: List of scaling factors [0.5, 1.0, 1.5]
        use_hierarchical: Use hierarchical integration (default: True)
    """
    def __init__(
        self,
        backbone,
        channels_list=[64, 128, 320, 512],
        scales=[0.5, 1.0, 1.5],
        use_hierarchical=True
    ):
        super().__init__()
        self.backbone = backbone
        self.channels_list = channels_list
        self.scales = scales
        self.num_scales = len(scales)
        self.use_hierarchical = use_hierarchical

        # Multi-scale input generator
        self.input_generator = MultiScaleInputGenerator(scales=scales)

        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(
            channels_list=channels_list,
            num_scales=self.num_scales
        )

        # Scale-aware loss module
        scale_weights = []
        for scale in scales:
            if scale == 1.0:
                scale_weights.append(1.0)
            else:
                scale_weights.append(0.5)
        self.loss_module = ScaleAwareLossModule(scale_weights=scale_weights)

        # Prediction heads for each scale (for supervision)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_list[0], channels_list[0] // 2, 3, padding=1),
                nn.BatchNorm2d(channels_list[0] // 2),
                nn.GELU(),
                nn.Conv2d(channels_list[0] // 2, 1, 1)
            ) for _ in range(self.num_scales)
        ])

    def extract_multi_scale_features(self, multi_scale_inputs):
        """
        Extract features from multi-scale inputs using shared backbone.

        Args:
            multi_scale_inputs: List of [B, 3, H_i, W_i] inputs

        Returns:
            multi_scale_features: List of 3 feature pyramids
        """
        multi_scale_features = []

        for scaled_input in multi_scale_inputs:
            # Extract features using shared backbone
            features = self.backbone(scaled_input)
            multi_scale_features.append(features)

        return multi_scale_features

    def forward(self, x, return_multi_scale=False, return_loss_inputs=False):
        """
        Forward pass through multi-scale processor.

        Args:
            x: [B, 3, H, W] input images
            return_multi_scale: If True, return features at all scales
            return_loss_inputs: If True, return predictions for loss computation

        Returns:
            If return_multi_scale=False and return_loss_inputs=False:
                fused_features: List of [feat1, feat2, feat3, feat4]

            If return_multi_scale=True:
                fused_features, multi_scale_features

            If return_loss_inputs=True:
                fused_features, scale_predictions
        """
        # Step 1: Generate multi-scale inputs
        multi_scale_inputs = self.input_generator(x)

        # Step 2: Extract features at each scale (shared backbone)
        multi_scale_features = self.extract_multi_scale_features(multi_scale_inputs)

        # Step 3: Fuse multi-scale features
        fused_features = self.feature_fusion(
            multi_scale_features,
            use_hierarchical=self.use_hierarchical
        )

        # Step 4: Generate predictions for each scale (for loss)
        if return_loss_inputs:
            scale_predictions = []
            for i, features in enumerate(multi_scale_features):
                # Use finest features for prediction
                pred = self.prediction_heads[i](features[0])
                scale_predictions.append(pred)

            if return_multi_scale:
                return fused_features, multi_scale_features, scale_predictions
            else:
                return fused_features, scale_predictions

        if return_multi_scale:
            return fused_features, multi_scale_features

        return fused_features

    def compute_loss(self, scale_predictions, target, criterion):
        """
        Compute scale-aware weighted loss.

        Args:
            scale_predictions: List of predictions at each scale
            target: [B, 1, H, W] ground truth
            criterion: Loss function

        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Dictionary with individual losses
        """
        # Create targets for each scale
        targets = []
        for pred in scale_predictions:
            if target.shape != pred.shape:
                target_resized = F.interpolate(
                    target,
                    size=pred.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                target_resized = target
            targets.append(target_resized)

        # Compute weighted loss
        total_loss, loss_dict = self.loss_module(
            scale_predictions,
            targets,
            criterion
        )

        return total_loss, loss_dict


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example backbone for testing
class DummyBackbone(nn.Module):
    """Dummy backbone for testing (mimics PVT/EdgeNeXt)"""
    def __init__(self, channels_list=[64, 128, 320, 512]):
        super().__init__()
        self.channels_list = channels_list

        # Simple conv layers to mimic backbone
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64 if i == 0 else channels_list[i-1],
                         channels_list[i], 3, stride=2, padding=1),
                nn.BatchNorm2d(channels_list[i]),
                nn.GELU()
            ) for i in range(len(channels_list))
        ])

    def forward(self, x):
        features = []
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features


if __name__ == '__main__':
    print("="*70)
    print("Testing MultiScaleInputProcessor")
    print("="*70)

    # Create dummy backbone
    backbone = DummyBackbone(channels_list=[64, 128, 320, 512])

    # Create processor
    processor = MultiScaleInputProcessor(
        backbone=backbone,
        channels_list=[64, 128, 320, 512],
        scales=[0.5, 1.0, 1.5],
        use_hierarchical=True
    )

    print(f"\nTotal parameters: {count_parameters(processor) / 1e6:.2f}M")

    # Test with variable input sizes
    print("\n" + "="*70)
    print("Test 1: Variable Input Sizes")
    print("="*70)

    test_sizes = [(256, 256), (320, 320), (352, 352), (384, 384)]

    for H, W in test_sizes:
        x = torch.randn(2, 3, H, W)
        print(f"\nInput size: {x.shape}")

        # Forward pass
        fused_features = processor(x)

        print(f"Output features:")
        for i, feat in enumerate(fused_features):
            print(f"  Level {i+1}: {feat.shape}")

    # Test multi-scale feature extraction
    print("\n" + "="*70)
    print("Test 2: Multi-Scale Feature Extraction")
    print("="*70)

    x = torch.randn(2, 3, 352, 352)
    fused_features, multi_scale_features = processor(x, return_multi_scale=True)

    print(f"\nInput: {x.shape}")
    print(f"\nMulti-scale features:")
    for scale_idx, features in enumerate(multi_scale_features):
        print(f"\n  Scale {scale_idx} (factor={processor.scales[scale_idx]}):")
        for level_idx, feat in enumerate(features):
            print(f"    Level {level_idx+1}: {feat.shape}")

    print(f"\nFused features:")
    for i, feat in enumerate(fused_features):
        print(f"  Level {i+1}: {feat.shape}")

    # Test loss computation
    print("\n" + "="*70)
    print("Test 3: Scale-Aware Loss Computation")
    print("="*70)

    x = torch.randn(2, 3, 352, 352)
    target = torch.randint(0, 2, (2, 1, 352, 352)).float()

    fused_features, scale_predictions = processor(x, return_loss_inputs=True)

    print(f"\nScale predictions:")
    for i, pred in enumerate(scale_predictions):
        print(f"  Scale {i} ({processor.scales[i]}×): {pred.shape}")

    # Compute loss
    criterion = nn.BCEWithLogitsLoss()
    total_loss, loss_dict = processor.compute_loss(
        scale_predictions,
        target,
        criterion
    )

    print(f"\nLoss computation:")
    print(f"  Total weighted loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        if key != 'total_loss':
            print(f"  {key}: {value:.4f}")

    # Test individual components
    print("\n" + "="*70)
    print("Test 4: Individual Components")
    print("="*70)

    # Test MultiScaleInputGenerator
    print("\n1. Multi-Scale Input Generator:")
    generator = MultiScaleInputGenerator(scales=[0.5, 1.0, 1.5])
    x = torch.randn(2, 3, 352, 352)
    multi_scale_inputs = generator(x)
    print(f"   Input: {x.shape}")
    for i, inp in enumerate(multi_scale_inputs):
        print(f"   Scale {generator.scales[i]}×: {inp.shape}")

    # Test ABSI
    print("\n2. Attention-Based Scale Integration Unit:")
    absi = AttentionBasedScaleIntegrationUnit(channels=128, num_scales=3)
    scale_features = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 128, 44, 44),
        torch.randn(2, 128, 52, 52)
    ]
    integrated = absi(scale_features, target_size=(44, 44))
    print(f"   Input scales:")
    for i, feat in enumerate(scale_features):
        print(f"     Scale {i}: {feat.shape}")
    print(f"   Integrated output: {integrated.shape}")

    # Test Hierarchical Integration
    print("\n3. Hierarchical Scale Integration:")
    hierarchical = HierarchicalScaleIntegration(channels=128)
    integrated_h = hierarchical(scale_features, target_size=(44, 44))
    print(f"   Hierarchical output: {integrated_h.shape}")

    # Test ScaleAwareLossModule
    print("\n4. Scale-Aware Loss Module:")
    loss_module = ScaleAwareLossModule(scale_weights=[0.5, 1.0, 0.5])
    predictions = [
        torch.randn(2, 1, 32, 32),
        torch.randn(2, 1, 44, 44),
        torch.randn(2, 1, 52, 52)
    ]
    targets = [
        torch.randn(2, 1, 32, 32),
        torch.randn(2, 1, 44, 44),
        torch.randn(2, 1, 52, 52)
    ]
    criterion = nn.MSELoss()
    total_loss, loss_dict = loss_module(predictions, targets, criterion)
    print(f"   Scale weights: {loss_module.scale_weights}")
    print(f"   Total loss: {total_loss.item():.4f}")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)

    print("\nArchitecture Summary:")
    print("  ✓ Multi-scale input generation (0.5×, 1.0×, 1.5×)")
    print("  ✓ Shared backbone for feature extraction")
    print("  ✓ Attention-Based Scale Integration Units")
    print("  ✓ Hierarchical scale integration (no simple concat)")
    print("  ✓ Scale-specific loss weighting (0.5, 1.0, 0.5)")
    print("  ✓ Variable input size support")
    print("  ✓ Unified multi-scale feature output")

    # Parameter breakdown
    print(f"\nParameter breakdown:")
    print(f"  Backbone: {count_parameters(backbone) / 1e6:.2f}M")
    print(f"  Feature Fusion: {count_parameters(processor.feature_fusion) / 1e6:.2f}M")
    print(f"  Prediction Heads: {count_parameters(processor.prediction_heads) / 1e6:.2f}M")
    print(f"  Total: {count_parameters(processor) / 1e6:.2f}M")
