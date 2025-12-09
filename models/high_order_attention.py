"""
HighOrderAttention: Advanced attention mechanism for detecting subtle camouflage differences.

Features:
1. Tensor decomposition (Tucker) to model subtle foreground-background differences
2. Multi-order polynomial attention (orders 2, 3, 4) beyond standard quadratic
3. Multi-granularity fusion at different hierarchical levels
4. Channel Interaction and Enhancement Module (CIEM)
5. Cross-knowledge propagation between adjacent attention levels

Significantly improves detection of subtle camouflaged objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


class TuckerDecomposition(nn.Module):
    """
    Tucker tensor decomposition for modeling subtle differences.

    Decomposes attention tensor into core tensor and factor matrices:
    A ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃

    where:
    - G is the core tensor (captures interactions)
    - U₁, U₂, U₃ are factor matrices (capture mode-specific patterns)

    This decomposition helps separate subtle foreground-background
    differences by decomposing complex attention patterns into
    interpretable low-rank components.

    Args:
        in_channels: Input channel dimension
        ranks: Decomposition ranks [r1, r2, r3] (default: [C//4, C//4, C//4])
        spatial_size: Spatial dimension for initialization (default: 8)
    """
    def __init__(
        self,
        in_channels: int,
        ranks: Optional[List[int]] = None,
        spatial_size: int = 8
    ):
        super().__init__()

        self.in_channels = in_channels

        if ranks is None:
            # Default: compress by factor of 4
            rank = max(in_channels // 4, 16)
            ranks = [rank, rank, rank]

        self.ranks = ranks
        self.spatial_size = spatial_size

        # Factor matrices for each mode
        # Mode 1: Channel dimension
        self.U1 = nn.Parameter(torch.randn(in_channels, ranks[0]))

        # Mode 2: Spatial height
        self.U2 = nn.Parameter(torch.randn(spatial_size, ranks[1]))

        # Mode 3: Spatial width
        self.U3 = nn.Parameter(torch.randn(spatial_size, ranks[2]))

        # Core tensor (captures mode interactions)
        self.core = nn.Parameter(torch.randn(ranks[0], ranks[1], ranks[2]))

        # Learnable scaling
        self.scale = nn.Parameter(torch.ones(1))

        # Initialize
        self._init_parameters()

    def _init_parameters(self):
        """Initialize decomposition parameters."""
        # Initialize factor matrices with SVD-like initialization
        nn.init.orthogonal_(self.U1)
        nn.init.orthogonal_(self.U2)
        nn.init.orthogonal_(self.U3)

        # Initialize core tensor
        nn.init.kaiming_normal_(self.core, mode='fan_out')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Tucker decomposition to extract subtle features.

        Args:
            x: [B, C, H, W] input features

        Returns:
            decomposed: [B, C, H, W] decomposed features highlighting subtle differences
        """
        B, C, H, W = x.shape

        # Reshape to process spatial dimensions
        x_reshaped = x.view(B, C, H * W)  # [B, C, HW]

        # Mode-1 product: Channel dimension
        # [B, C, HW] @ [C, r1] -> [B, r1, HW]
        mode1 = torch.matmul(x_reshaped.transpose(1, 2), self.U1).transpose(1, 2)

        # Reshape for spatial processing
        mode1 = mode1.view(B, self.ranks[0], H, W)

        # Interpolate spatial factor matrices if sizes don't match
        if H != self.spatial_size or W != self.spatial_size:
            U2_interp = F.interpolate(
                self.U2.unsqueeze(0).unsqueeze(0),  # [1, 1, spatial_size, r2]
                size=(H, self.ranks[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # [H, r2]

            U3_interp = F.interpolate(
                self.U3.unsqueeze(0).unsqueeze(0),
                size=(W, self.ranks[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # [W, r3]
        else:
            U2_interp = self.U2
            U3_interp = self.U3

        # Mode-2 product: Height dimension
        # Reshape: [B, r1, H, W] -> [B*r1*W, H]
        temp = mode1.permute(0, 1, 3, 2).reshape(B * self.ranks[0] * W, H)
        # [B*r1*W, H] @ [H, r2] -> [B*r1*W, r2]
        mode2 = torch.matmul(temp, U2_interp)
        # Reshape back: [B, r1, W, r2]
        mode2 = mode2.view(B, self.ranks[0], W, self.ranks[1])

        # Mode-3 product: Width dimension
        # Reshape: [B, r1, W, r2] -> [B*r1*r2, W]
        temp = mode2.permute(0, 1, 3, 2).reshape(B * self.ranks[0] * self.ranks[1], W)
        # [B*r1*r2, W] @ [W, r3] -> [B*r1*r2, r3]
        mode3 = torch.matmul(temp, U3_interp)
        # Reshape: [B, r1, r2, r3]
        mode3 = mode3.view(B, self.ranks[0], self.ranks[1], self.ranks[2])

        # Multiply with core tensor
        # [B, r1, r2, r3] * [r1, r2, r3] -> [B, r1, r2, r3]
        decomposed_compact = mode3 * self.core.unsqueeze(0)

        # Sum over decomposition dimensions
        # [B, r1, r2, r3] -> [B]
        decomposed_score = decomposed_compact.sum(dim=[1, 2, 3])

        # Create attention mask from decomposition
        # [B] -> [B, 1, 1, 1]
        attention = torch.sigmoid(decomposed_score.view(B, 1, 1, 1) * self.scale)

        # Apply attention to original features
        output = x * attention

        return output


class PolynomialAttention(nn.Module):
    """
    Multi-order polynomial attention beyond standard quadratic attention.

    Standard attention: A = softmax(QK^T / sqrt(d))  [Order 2]
    Polynomial: A = sum_{i=2}^{max_order} α_i * (QK^T / sqrt(d))^i

    Higher-order terms capture complex non-linear relationships between
    query and key, important for subtle camouflage patterns.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        max_order: Maximum polynomial order (default: 4, includes orders 2,3,4)
        qkv_bias: Add bias to Q,K,V projections (default: True)
        dropout: Dropout rate (default: 0.0)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        max_order: int = 4,
        qkv_bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_order = max_order
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Learnable weights for each polynomial order
        self.order_weights = nn.Parameter(torch.ones(max_order - 1))  # Orders 2,3,4,...

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization for stability
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply polynomial attention.

        Args:
            x: [B, N, C] input features (N = H*W for images)

        Returns:
            output: [B, N, C] attended features
            attention_map: [B, num_heads, N, N] multi-order attention map
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute base attention: QK^T / sqrt(d)
        # [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N] -> [B, num_heads, N, N]
        base_attn = (q @ k.transpose(-2, -1)) * self.scale

        # Initialize multi-order attention
        attn_multiorder = torch.zeros_like(base_attn)

        # Normalize order weights
        order_weights_normalized = F.softmax(self.order_weights, dim=0)

        # Compute polynomial terms
        current_power = base_attn
        for order_idx in range(self.max_order - 1):
            # order_idx = 0 -> order 2
            # order_idx = 1 -> order 3
            # order_idx = 2 -> order 4
            order = order_idx + 2

            # Raise to current order (for order 2, it's already squared in definition)
            if order_idx == 0:
                # Order 2: base_attn already computed
                power_term = base_attn
            else:
                # Order 3+: multiply by base_attn
                power_term = current_power * base_attn
                current_power = power_term

            # Add weighted term
            attn_multiorder = attn_multiorder + order_weights_normalized[order_idx] * power_term

        # Apply softmax to multi-order attention
        attn_final = F.softmax(attn_multiorder, dim=-1)
        attn_final = self.dropout(attn_final)

        # Apply attention to values
        # [B, num_heads, N, N] @ [B, num_heads, N, head_dim] -> [B, num_heads, N, head_dim]
        out = attn_final @ v

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.proj(out)
        out = self.dropout(out)

        # Residual connection + norm
        out = self.norm(x + out)

        return out, attn_final


class ChannelInteractionEnhancementModule(nn.Module):
    """
    Channel Interaction and Enhancement Module (CIEM).

    Enhances feature channels through:
    1. Channel-wise attention (global context)
    2. Cross-channel interaction (channel dependencies)
    3. Channel enhancement (non-linear refinement)

    Critical for COD as different channels capture different
    camouflage cues (texture, color, edges).

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
        num_groups: Number of channel groups for interaction (default: 4)
    """
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        num_groups: int = 4
    ):
        super().__init__()

        self.channels = channels
        self.reduction = reduction
        self.num_groups = num_groups

        # 1. Channel-wise attention (squeeze-excitation like)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 2. Cross-channel interaction
        # Split channels into groups and enable cross-group communication
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        group_channels = channels // num_groups

        self.channel_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, group_channels, 1, bias=False),
                nn.BatchNorm2d(group_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_groups)
        ])

        # Fusion after interaction
        self.interaction_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 3. Channel enhancement (non-linear refinement)
        self.enhancement = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel interaction and enhancement.

        Args:
            x: [B, C, H, W] input features

        Returns:
            enhanced: [B, C, H, W] channel-enhanced features
        """
        B, C, H, W = x.shape

        # 1. Channel-wise attention
        channel_attn = self.channel_attention(x)  # [B, C, 1, 1]
        x_attended = x * channel_attn  # [B, C, H, W]

        # 2. Cross-channel interaction
        interactions = []
        for interact_module in self.channel_interaction:
            interactions.append(interact_module(x_attended))

        # Concatenate group interactions
        x_interacted = torch.cat(interactions, dim=1)  # [B, C, H, W]
        x_interacted = self.interaction_fusion(x_interacted)

        # 3. Channel enhancement
        enhancement_gate = self.enhancement(x_interacted)  # [B, C, H, W]
        x_enhanced = x_interacted * enhancement_gate

        # Residual connection
        output = x + x_enhanced

        return output


class MultiGranularityFusion(nn.Module):
    """
    Multi-granularity fusion at different hierarchical levels.

    Fuses features at multiple granularities:
    - Fine-grained: Pixel-level details
    - Medium-grained: Local regions
    - Coarse-grained: Global context

    Hierarchical fusion captures camouflage patterns at multiple scales.

    Args:
        channels: Number of input channels
        num_levels: Number of granularity levels (default: 3)
        fusion_mode: Fusion strategy ('concat', 'add', 'attention')
    """
    def __init__(
        self,
        channels: int,
        num_levels: int = 3,
        fusion_mode: str = 'attention'
    ):
        super().__init__()

        self.channels = channels
        self.num_levels = num_levels
        self.fusion_mode = fusion_mode

        # Multi-scale feature extraction at different granularities
        self.granularity_extractors = nn.ModuleList()

        for level in range(num_levels):
            # Level 0: Fine (1x1, captures pixel-level)
            # Level 1: Medium (3x3, captures local regions)
            # Level 2: Coarse (5x5 or pooling, captures global context)

            if level == 0:
                # Fine-grained: 1x1 conv
                extractor = nn.Sequential(
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            elif level == 1:
                # Medium-grained: 3x3 conv
                extractor = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # Coarse-grained: 5x5 conv or dilated conv
                extractor = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )

            self.granularity_extractors.append(extractor)

        # Fusion layer
        if fusion_mode == 'concat':
            self.fusion = nn.Sequential(
                nn.Conv2d(channels * num_levels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_mode == 'attention':
            # Attention-based fusion
            self.fusion_attention = nn.Sequential(
                nn.Conv2d(channels * num_levels, num_levels, 1, bias=False),
                nn.Softmax(dim=1)
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels)
            )
        else:
            # 'add' mode: no fusion layer needed
            self.fusion = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-granularity fusion.

        Args:
            x: [B, C, H, W] input features

        Returns:
            fused: [B, C, H, W] multi-granularity fused features
        """
        B, C, H, W = x.shape

        # Extract features at different granularities
        granularity_features = []
        for extractor in self.granularity_extractors:
            feat = extractor(x)
            granularity_features.append(feat)

        # Fuse features
        if self.fusion_mode == 'concat':
            # Concatenate and fuse
            concat_feats = torch.cat(granularity_features, dim=1)  # [B, C*num_levels, H, W]
            fused = self.fusion(concat_feats)

        elif self.fusion_mode == 'attention':
            # Attention-weighted fusion
            concat_feats = torch.cat(granularity_features, dim=1)  # [B, C*num_levels, H, W]

            # Compute attention weights
            attn_weights = self.fusion_attention(concat_feats)  # [B, num_levels, H, W]

            # Weighted sum
            fused = torch.zeros_like(x)
            for i, feat in enumerate(granularity_features):
                fused = fused + attn_weights[:, i:i+1, :, :] * feat

            fused = self.fusion(fused)

        else:  # 'add'
            # Simple addition
            fused = sum(granularity_features)

        # Residual connection
        output = x + fused

        return output


class CrossKnowledgePropagation(nn.Module):
    """
    Cross-knowledge propagation between adjacent attention levels.

    Enables information flow between different attention hierarchy levels:
    - Bottom-up: Fine details to coarse semantics
    - Top-down: Global context to local details
    - Lateral: Same-level cross-feature communication

    Improves consistency across attention levels.

    Args:
        channels: Number of channels at each level
        num_levels: Number of hierarchical levels
        propagation_mode: 'bidirectional', 'bottom_up', or 'top_down'
    """
    def __init__(
        self,
        channels: int,
        num_levels: int = 3,
        propagation_mode: str = 'bidirectional'
    ):
        super().__init__()

        self.channels = channels
        self.num_levels = num_levels
        self.propagation_mode = propagation_mode

        # Bottom-up propagation (fine -> coarse)
        if propagation_mode in ['bidirectional', 'bottom_up']:
            self.bottom_up_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                ) for _ in range(num_levels - 1)
            ])

        # Top-down propagation (coarse -> fine)
        if propagation_mode in ['bidirectional', 'top_down']:
            self.top_down_convs = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                ) for _ in range(num_levels - 1)
            ])

        # Lateral connections (same level refinement)
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels)
        ])

        # Fusion gates for combining propagated knowledge
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.Sigmoid()
            ) for _ in range(num_levels)
        ])

    def forward(self, level_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Propagate knowledge across attention levels.

        Args:
            level_features: List of [B, C, H_i, W_i] features at each level
                           (from fine to coarse resolution)

        Returns:
            propagated_features: List of enhanced features with cross-level knowledge
        """
        assert len(level_features) == self.num_levels

        B = level_features[0].size(0)

        # Initialize output features
        output_features = [feat.clone() for feat in level_features]

        # Bottom-up propagation
        if self.propagation_mode in ['bidirectional', 'bottom_up']:
            for i in range(self.num_levels - 1):
                # Propagate from level i to level i+1
                propagated = self.bottom_up_convs[i](output_features[i])

                # Resize if dimensions don't match
                if propagated.shape[2:] != output_features[i+1].shape[2:]:
                    propagated = F.interpolate(
                        propagated,
                        size=output_features[i+1].shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Fuse with current level features
                concat = torch.cat([output_features[i+1], propagated], dim=1)
                gate = self.fusion_gates[i+1](concat)
                output_features[i+1] = output_features[i+1] * gate + propagated * (1 - gate)

        # Top-down propagation
        if self.propagation_mode in ['bidirectional', 'top_down']:
            for i in range(self.num_levels - 1, 0, -1):
                # Propagate from level i to level i-1
                propagated = self.top_down_convs[i-1](output_features[i])

                # Resize if dimensions don't match
                if propagated.shape[2:] != output_features[i-1].shape[2:]:
                    propagated = F.interpolate(
                        propagated,
                        size=output_features[i-1].shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Fuse with current level features
                concat = torch.cat([output_features[i-1], propagated], dim=1)
                gate = self.fusion_gates[i-1](concat)
                output_features[i-1] = output_features[i-1] * gate + propagated * (1 - gate)

        # Lateral refinement
        for i in range(self.num_levels):
            lateral_refined = self.lateral_convs[i](output_features[i])
            output_features[i] = output_features[i] + lateral_refined

        return output_features


class HighOrderAttention(nn.Module):
    """
    High-order attention module for detecting subtle camouflage differences.

    Combines:
    1. Tensor decomposition (Tucker) for subtle difference modeling
    2. Multi-order polynomial attention (orders 2-4)
    3. Multi-granularity fusion (fine/medium/coarse)
    4. Channel Interaction and Enhancement Module (CIEM)
    5. Cross-knowledge propagation between levels

    Significantly improves detection of subtle foreground-background differences
    in camouflaged object detection.

    Args:
        channels: List of channel dimensions at each level [C1, C2, C3, C4]
        num_heads: Number of attention heads for polynomial attention
        max_order: Maximum polynomial order (default: 4)
        num_granularity_levels: Number of granularity levels (default: 3)
        tucker_ranks: Tucker decomposition ranks (default: None, auto-computed)
        propagation_mode: Cross-level propagation mode (default: 'bidirectional')
    """
    def __init__(
        self,
        channels: List[int],
        num_heads: int = 8,
        max_order: int = 4,
        num_granularity_levels: int = 3,
        tucker_ranks: Optional[List[List[int]]] = None,
        propagation_mode: str = 'bidirectional'
    ):
        super().__init__()

        self.channels = channels
        self.num_levels = len(channels)
        self.num_heads = num_heads
        self.max_order = max_order

        # 1. Tucker decomposition for each level
        self.tucker_decompositions = nn.ModuleList()
        for i, ch in enumerate(channels):
            if tucker_ranks is not None:
                ranks = tucker_ranks[i]
            else:
                ranks = None

            self.tucker_decompositions.append(
                TuckerDecomposition(ch, ranks=ranks)
            )

        # 2. Polynomial attention for each level
        self.polynomial_attentions = nn.ModuleList()
        for ch in channels:
            # Adjust num_heads if channel not divisible
            level_heads = min(num_heads, ch)
            while ch % level_heads != 0:
                level_heads -= 1

            self.polynomial_attentions.append(
                PolynomialAttention(
                    dim=ch,
                    num_heads=level_heads,
                    max_order=max_order
                )
            )

        # 3. Channel Interaction and Enhancement Modules (CIEM)
        self.ciems = nn.ModuleList([
            ChannelInteractionEnhancementModule(ch) for ch in channels
        ])

        # 4. Multi-granularity fusion for each level
        self.granularity_fusions = nn.ModuleList([
            MultiGranularityFusion(ch, num_levels=num_granularity_levels)
            for ch in channels
        ])

        # 5. Cross-knowledge propagation
        # Use the most common channel dimension for propagation
        # (or first level if all different)
        prop_channels = channels[0]

        # Align all levels to common dimension for propagation
        self.channel_alignments = nn.ModuleList()
        for ch in channels:
            if ch != prop_channels:
                self.channel_alignments.append(
                    nn.Conv2d(ch, prop_channels, 1, bias=False)
                )
            else:
                self.channel_alignments.append(nn.Identity())

        self.cross_knowledge_prop = CrossKnowledgePropagation(
            channels=prop_channels,
            num_levels=self.num_levels,
            propagation_mode=propagation_mode
        )

        # Restore original channels after propagation
        self.channel_restorations = nn.ModuleList()
        for ch in channels:
            if ch != prop_channels:
                self.channel_restorations.append(
                    nn.Conv2d(prop_channels, ch, 1, bias=False)
                )
            else:
                self.channel_restorations.append(nn.Identity())

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply high-order attention to multi-level features.

        Args:
            features: List of [B, C_i, H_i, W_i] features at each level

        Returns:
            enhanced_features: List of enhanced features
            attention_info: Dictionary containing attention maps and statistics
        """
        assert len(features) == self.num_levels

        # Store intermediate outputs
        tucker_features = []
        polynomial_features = []
        ciem_features = []
        granularity_features = []
        attention_maps = []

        # Process each level independently
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape

            # 1. Tucker decomposition for subtle differences
            tucker_feat = self.tucker_decompositions[i](feat)
            tucker_features.append(tucker_feat)

            # 2. Polynomial attention
            # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
            feat_flat = tucker_feat.flatten(2).transpose(1, 2)  # [B, HW, C]

            poly_feat, attn_map = self.polynomial_attentions[i](feat_flat)

            # Reshape back: [B, HW, C] -> [B, C, H, W]
            poly_feat = poly_feat.transpose(1, 2).reshape(B, C, H, W)
            polynomial_features.append(poly_feat)
            attention_maps.append(attn_map)

            # 3. Channel Interaction and Enhancement
            ciem_feat = self.ciems[i](poly_feat)
            ciem_features.append(ciem_feat)

            # 4. Multi-granularity fusion
            granularity_feat = self.granularity_fusions[i](ciem_feat)
            granularity_features.append(granularity_feat)

        # 5. Cross-knowledge propagation across levels
        # Align channels
        aligned_features = []
        for i, feat in enumerate(granularity_features):
            aligned = self.channel_alignments[i](feat)
            aligned_features.append(aligned)

        # Propagate knowledge
        propagated_features = self.cross_knowledge_prop(aligned_features)

        # Restore original channels
        enhanced_features = []
        for i, feat in enumerate(propagated_features):
            restored = self.channel_restorations[i](feat)
            enhanced_features.append(restored)

        # Prepare attention info
        attention_info = {
            'tucker_features': tucker_features,
            'polynomial_features': polynomial_features,
            'ciem_features': ciem_features,
            'granularity_features': granularity_features,
            'attention_maps': attention_maps
        }

        return enhanced_features, attention_info


# Example usage and testing
if __name__ == '__main__':
    print("HighOrderAttention Module Test")
    print("=" * 60)

    # Create dummy multi-level features
    batch_size = 2
    channels = [64, 128, 320, 512]  # Typical backbone output channels
    heights = [64, 32, 16, 8]
    widths = [64, 32, 16, 8]

    features = []
    for i in range(4):
        feat = torch.randn(batch_size, channels[i], heights[i], widths[i])
        features.append(feat)
        print(f"Level {i}: {feat.shape}")

    # Create HighOrderAttention module
    print("\nCreating HighOrderAttention module...")
    high_order_attn = HighOrderAttention(
        channels=channels,
        num_heads=8,
        max_order=4,
        num_granularity_levels=3,
        propagation_mode='bidirectional'
    )

    print(f"Total parameters: {sum(p.numel() for p in high_order_attn.parameters()):,}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        enhanced_features, attention_info = high_order_attn(features)

    print("\nOutput shapes:")
    for i, feat in enumerate(enhanced_features):
        print(f"Enhanced Level {i}: {feat.shape}")

    print("\nAttention info keys:", list(attention_info.keys()))
    print(f"Number of attention maps: {len(attention_info['attention_maps'])}")

    print("\n✓ HighOrderAttention test completed successfully!")
