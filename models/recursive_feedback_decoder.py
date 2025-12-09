"""
RecursiveFeedbackDecoder: Iterative refinement decoder with high-resolution preservation.

Features:
1. Multi-resolution iterative refinement with global loop connections
2. PVT-inspired memory-efficient attention blocks
3. Iteration weight schemes to prevent feature corruption
4. 3-5 refinement passes with residual connections
5. High-resolution feature maintenance throughout decoding

Addresses detail degradation from resolution loss in standard decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


class SpatialReductionAttention(nn.Module):
    """
    PVT-style spatial reduction attention for memory efficiency.

    Reduces spatial dimensions of K and V while keeping Q at full resolution.
    This significantly reduces memory and computation while maintaining
    high-resolution query features.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        sr_ratio: Spatial reduction ratio (e.g., 8 means 8x reduction)
        qkv_bias: Add bias to Q,K,V projections
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        qkv_bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        # Q projection (full resolution)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # K, V projections
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Spatial reduction for K, V
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply spatial reduction attention.

        Args:
            x: [B, N, C] where N = H * W
            H: Height
            W: Width

        Returns:
            out: [B, N, C] attended features
        """
        B, N, C = x.shape

        # Q at full resolution
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B, num_heads, N, head_dim]

        # K, V with spatial reduction
        if self.sr_ratio > 1:
            # Reshape to spatial
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

            # Apply spatial reduction
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)

            # K, V projections
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # [B, num_heads, N_reduced, head_dim]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class PVTFeedForward(nn.Module):
    """
    PVT-style feed-forward network with depthwise convolution.

    Adds spatial awareness to standard FFN through 3x3 depthwise conv.

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4x dim)
        dropout: Dropout rate
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply feed-forward with depthwise conv.

        Args:
            x: [B, N, C]
            H: Height
            W: Width

        Returns:
            out: [B, N, C]
        """
        B, N, C = x.shape

        x = self.fc1(x)

        # Apply depthwise conv (requires spatial layout)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, -1, N).permute(0, 2, 1)

        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class PVTBlock(nn.Module):
    """
    PVT transformer block with spatial reduction attention.

    Combines spatial reduction attention and depthwise FFN for
    memory-efficient high-resolution feature processing.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        sr_ratio: Spatial reduction ratio
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = PVTFeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply PVT block.

        Args:
            x: [B, N, C]
            H: Height
            W: Width

        Returns:
            out: [B, N, C]
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x), H, W)

        # FFN with residual
        x = x + self.mlp(self.norm2(x), H, W)

        return x


class IterationWeightingScheme(nn.Module):
    """
    Learnable iteration weighting to prevent feature corruption.

    As refinement iterations progress, earlier iterations may become
    stale or corrupted. This module learns to weight iterations
    adaptively based on their quality and relevance.

    Args:
        num_iterations: Number of refinement iterations
        channels: Feature channels
        scheme: Weighting scheme ('learned', 'exponential', 'uniform')
    """
    def __init__(
        self,
        num_iterations: int,
        channels: int,
        scheme: str = 'learned'
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.channels = channels
        self.scheme = scheme

        if scheme == 'learned':
            # Learnable weights per iteration
            self.iteration_weights = nn.Parameter(torch.ones(num_iterations))

            # Quality assessment network
            self.quality_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            )

        elif scheme == 'exponential':
            # Exponential decay: later iterations weighted less
            # weight[i] = exp(-decay * i)
            self.decay = nn.Parameter(torch.tensor(0.1))

        # Scheme 'uniform' requires no parameters

    def forward(
        self,
        iteration_features: List[torch.Tensor],
        current_iteration: int
    ) -> torch.Tensor:
        """
        Weight and combine iteration features.

        Args:
            iteration_features: List of [B, C, H, W] features from each iteration
            current_iteration: Current iteration index

        Returns:
            weighted: [B, C, H, W] weighted combination
        """
        if self.scheme == 'uniform':
            # Simple averaging
            return torch.stack(iteration_features).mean(dim=0)

        elif self.scheme == 'exponential':
            # Exponential decay weighting
            weights = []
            for i in range(len(iteration_features)):
                weight = torch.exp(-self.decay * i)
                weights.append(weight)

            weights = torch.stack(weights)
            weights = weights / weights.sum()

            # Weighted sum
            weighted = torch.zeros_like(iteration_features[0])
            for i, feat in enumerate(iteration_features):
                weighted = weighted + weights[i] * feat

            return weighted

        elif self.scheme == 'learned':
            # Learned quality-aware weighting
            weights = []

            # Compute quality score for each iteration
            for i, feat in enumerate(iteration_features):
                quality = self.quality_net(feat)  # [B, 1, 1, 1]

                # Combine with learned iteration weight
                iter_weight = torch.sigmoid(self.iteration_weights[i])
                weight = quality * iter_weight
                weights.append(weight)

            # Normalize weights
            weights = torch.cat(weights, dim=1)  # [B, num_iter, 1, 1]
            weights = F.softmax(weights, dim=1)

            # Weighted sum
            weighted = torch.zeros_like(iteration_features[0])
            for i, feat in enumerate(iteration_features):
                weighted = weighted + weights[:, i:i+1, :, :] * feat

            return weighted

        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")


class HighResolutionFusionModule(nn.Module):
    """
    Maintains high-resolution features during multi-scale fusion.

    Unlike standard decoders that aggressively downsample, this module
    preserves spatial resolution through:
    1. Minimal downsampling (only when necessary)
    2. High-res feature prioritization
    3. Detail-preserving upsampling

    Args:
        in_channels: List of input channels at each scale
        out_channels: Output channels
        base_resolution: Base spatial resolution to maintain
    """
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        base_resolution: int = 64
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_resolution = base_resolution
        self.num_scales = len(in_channels)

        # Channel alignment convolutions
        self.channel_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ch in in_channels
        ])

        # Detail-preserving upsampling (pixel shuffle)
        self.upsample_modules = nn.ModuleList()
        for i in range(self.num_scales):
            # Determine upsampling factor
            # Assume scales are: 1/8, 1/4, 1/2, 1 of base resolution
            upsample_factor = 2 ** (self.num_scales - 1 - i)

            if upsample_factor > 1:
                # Pixel shuffle upsampling
                self.upsample_modules.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels * (upsample_factor ** 2), 3, 1, 1),
                        nn.PixelShuffle(upsample_factor),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.upsample_modules.append(nn.Identity())

        # High-res feature enhancement
        self.high_res_enhance = nn.Sequential(
            nn.Conv2d(out_channels * self.num_scales, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features while preserving high resolution.

        Args:
            multi_scale_features: List of [B, C_i, H_i, W_i] at different scales

        Returns:
            fused: [B, out_channels, H_max, W_max] high-res fused features
        """
        # Determine target resolution (highest resolution input)
        target_h = max(feat.size(2) for feat in multi_scale_features)
        target_w = max(feat.size(3) for feat in multi_scale_features)

        aligned_features = []

        for i, feat in enumerate(multi_scale_features):
            # Align channels
            feat = self.channel_align[i](feat)

            # Upsample to target resolution
            if feat.size(2) != target_h or feat.size(3) != target_w:
                # Use learned upsampling if available
                feat = self.upsample_modules[i](feat)

                # Additional interpolation if needed
                if feat.size(2) != target_h or feat.size(3) != target_w:
                    feat = F.interpolate(
                        feat,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )

            aligned_features.append(feat)

        # Concatenate and enhance
        concat_feat = torch.cat(aligned_features, dim=1)
        fused = self.high_res_enhance(concat_feat)

        return fused


class RefinementBlock(nn.Module):
    """
    Single refinement block with residual connection.

    Refines features through:
    1. PVT-style efficient attention
    2. High-resolution feature processing
    3. Residual connections to preserve information

    Args:
        channels: Feature channels
        num_heads: Number of attention heads
        sr_ratio: Spatial reduction ratio for attention
        use_residual: Whether to use residual connection
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        sr_ratio: int = 2,
        use_residual: bool = True
    ):
        super().__init__()

        self.use_residual = use_residual

        # PVT block for efficient refinement
        self.pvt_block = PVTBlock(
            dim=channels,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            mlp_ratio=4.0
        )

        # Convolutional refinement
        self.conv_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Feature gating for adaptive refinement
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refine features.

        Args:
            x: [B, C, H, W] input features

        Returns:
            refined: [B, C, H, W] refined features
        """
        B, C, H, W = x.shape
        identity = x

        # PVT-based attention refinement
        # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)
        x_attn = self.pvt_block(x_flat, H, W)
        x_attn = x_attn.transpose(1, 2).reshape(B, C, H, W)

        # Convolutional refinement
        x_conv = self.conv_refine(x_attn)

        # Gated fusion
        gate = self.gate(x_conv)
        refined = x_conv * gate + x_attn * (1 - gate)

        # Residual connection
        if self.use_residual:
            refined = refined + identity

        return refined


class RecursiveFeedbackDecoder(nn.Module):
    """
    Recursive feedback decoder with iterative refinement.

    Features:
    1. Multi-resolution iterative refinement with global loop connections
    2. PVT-inspired memory-efficient attention
    3. Iteration weight schemes preventing feature corruption
    4. 3-5 refinement passes with residual connections
    5. High-resolution feature maintenance

    Addresses detail degradation from resolution loss in standard decoders.

    Args:
        encoder_channels: List of encoder output channels [C1, C2, C3, C4]
        decoder_channels: Decoder feature channels
        num_iterations: Number of refinement iterations (3-5)
        num_classes: Number of output classes (1 for binary COD)
        iteration_scheme: Iteration weighting scheme ('learned', 'exponential', 'uniform')
        sr_ratios: Spatial reduction ratios for each scale [8, 4, 2, 1]
        use_global_feedback: Enable global loop connections
    """
    def __init__(
        self,
        encoder_channels: List[int] = [64, 128, 320, 512],
        decoder_channels: int = 256,
        num_iterations: int = 4,
        num_classes: int = 1,
        iteration_scheme: str = 'learned',
        sr_ratios: Optional[List[int]] = None,
        use_global_feedback: bool = True
    ):
        super().__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.num_iterations = num_iterations
        self.num_classes = num_classes
        self.iteration_scheme = iteration_scheme
        self.use_global_feedback = use_global_feedback

        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]  # Default: more reduction at coarser scales
        self.sr_ratios = sr_ratios

        # High-resolution fusion module
        self.high_res_fusion = HighResolutionFusionModule(
            in_channels=encoder_channels,
            out_channels=decoder_channels
        )

        # Refinement blocks for each iteration
        self.refinement_blocks = nn.ModuleList([
            RefinementBlock(
                channels=decoder_channels,
                num_heads=8,
                sr_ratio=2,  # Moderate reduction for refinement
                use_residual=True
            ) for _ in range(num_iterations)
        ])

        # Iteration weighting scheme
        self.iteration_weighting = IterationWeightingScheme(
            num_iterations=num_iterations,
            channels=decoder_channels,
            scheme=iteration_scheme
        )

        # Global feedback connection
        if use_global_feedback:
            self.feedback_conv = nn.Sequential(
                nn.Conv2d(num_classes, decoder_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True)
            )

            # Feedback gating
            self.feedback_gate = nn.Sequential(
                nn.Conv2d(decoder_channels * 2, decoder_channels, 1),
                nn.Sigmoid()
            )

        # Prediction heads for each iteration
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels // 2, 3, 1, 1),
                nn.BatchNorm2d(decoder_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels // 2, num_classes, 1)
            ) for _ in range(num_iterations)
        ])

        # Final fusion head
        self.final_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, num_classes, 1)
        )

    def forward(
        self,
        encoder_features: List[torch.Tensor],
        return_iterations: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Recursive feedback decoding with iterative refinement.

        Args:
            encoder_features: List of [B, C_i, H_i, W_i] from encoder
            return_iterations: Whether to return intermediate iterations

        Returns:
            outputs: Dictionary containing:
                - 'final_prediction': [B, num_classes, H, W] final output
                - 'iteration_predictions': List of iteration outputs (if return_iterations=True)
                - 'iteration_features': List of iteration features (if return_iterations=True)
        """
        # Initial high-resolution fusion
        fused_features = self.high_res_fusion(encoder_features)

        # Store iteration features and predictions
        iteration_features = []
        iteration_predictions = []

        # Current features for refinement
        current_features = fused_features
        previous_prediction = None

        # Iterative refinement loop
        for iteration in range(self.num_iterations):
            # Global feedback connection
            if self.use_global_feedback and previous_prediction is not None:
                # Convert prediction to features
                feedback_feat = self.feedback_conv(previous_prediction)

                # Resize if needed
                if feedback_feat.shape[2:] != current_features.shape[2:]:
                    feedback_feat = F.interpolate(
                        feedback_feat,
                        size=current_features.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Gated fusion with feedback
                concat = torch.cat([current_features, feedback_feat], dim=1)
                gate = self.feedback_gate(concat)
                current_features = current_features * gate + feedback_feat * (1 - gate)

            # Refinement
            refined_features = self.refinement_blocks[iteration](current_features)

            # Generate prediction for this iteration
            prediction = self.prediction_heads[iteration](refined_features)

            # Store
            iteration_features.append(refined_features)
            iteration_predictions.append(prediction)

            # Update for next iteration
            current_features = refined_features
            previous_prediction = prediction

        # Weight and combine iterations
        weighted_features = self.iteration_weighting(
            iteration_features,
            current_iteration=self.num_iterations - 1
        )

        # Final prediction
        final_prediction = self.final_head(weighted_features)

        # Prepare outputs
        outputs = {
            'final_prediction': final_prediction,
        }

        if return_iterations:
            outputs['iteration_predictions'] = iteration_predictions
            outputs['iteration_features'] = iteration_features

        return outputs

    def get_iteration_count(self) -> int:
        """Get current number of refinement iterations."""
        return self.num_iterations

    def set_iteration_count(self, num_iterations: int):
        """
        Dynamically adjust number of refinement iterations.

        Useful for:
        - Progressive training (start with fewer iterations)
        - Inference speed-accuracy trade-off

        Args:
            num_iterations: New number of iterations (must be <= original)
        """
        if num_iterations > len(self.refinement_blocks):
            raise ValueError(
                f"Cannot set iterations to {num_iterations}, "
                f"max is {len(self.refinement_blocks)}"
            )

        self.num_iterations = num_iterations


# Example usage
if __name__ == '__main__':
    print("RecursiveFeedbackDecoder Test")
    print("=" * 60)

    # Create decoder
    decoder = RecursiveFeedbackDecoder(
        encoder_channels=[64, 128, 320, 512],
        decoder_channels=256,
        num_iterations=4,
        num_classes=1,
        iteration_scheme='learned',
        use_global_feedback=True
    )

    print(f"Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Create dummy encoder features
    batch_size = 2
    encoder_features = [
        torch.randn(batch_size, 64, 64, 64),   # 1/4 resolution
        torch.randn(batch_size, 128, 32, 32),  # 1/8 resolution
        torch.randn(batch_size, 320, 16, 16),  # 1/16 resolution
        torch.randn(batch_size, 512, 8, 8)     # 1/32 resolution
    ]

    print("\nEncoder features:")
    for i, feat in enumerate(encoder_features):
        print(f"  Level {i}: {feat.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = decoder(encoder_features, return_iterations=True)

    print("\nOutput shapes:")
    print(f"  Final prediction: {outputs['final_prediction'].shape}")
    print(f"  Iteration predictions: {len(outputs['iteration_predictions'])} outputs")
    for i, pred in enumerate(outputs['iteration_predictions']):
        print(f"    Iteration {i}: {pred.shape}")

    print("\n✓ RecursiveFeedbackDecoder test completed successfully!")
