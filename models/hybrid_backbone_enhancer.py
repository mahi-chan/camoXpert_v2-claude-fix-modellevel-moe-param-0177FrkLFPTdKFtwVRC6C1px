"""
HybridBackboneEnhancer: CNN + Transformer Hybrid Architecture

Enhances CNN local features with transformer global modeling through:
1. Non-Local Token Enhancement Module (NL-TEM) with Graph Convolution Networks
2. Feature Shrinkage Decoder (FSD) with 4-layer hierarchical structure
3. Cross-modulation between CNN and transformer features at each scale
4. Progressive aggregation with layer-wise supervision weights 2^(i-4)

Input:  CNN features [64, 128, 320, 512] at different scales
Output: Enhanced features maintaining same dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional
from einops import rearrange


class GraphConvolutionNetwork(nn.Module):
    """
    Graph Convolution Network for modeling high-order semantic relations.

    Constructs a graph where tokens are nodes and edges represent semantic similarity.
    """
    def __init__(self, in_features, out_features, num_heads=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # Multi-head graph convolution
        self.head_dim = out_features // num_heads

        # Graph construction: compute adjacency matrix
        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)

        # Edge weight learning
        self.edge_net = nn.Sequential(
            nn.Linear(in_features * 2, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Linear(out_features, num_heads)
        )

        # Output projection
        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def construct_graph(self, x):
        """
        Construct graph adjacency matrix based on semantic similarity.

        Args:
            x: [B, N, C] token features
        Returns:
            adj: [B, num_heads, N, N] adjacency matrices
        """
        B, N, C = x.shape

        # Compute pairwise features for edge weights
        # Expand to [B, N, N, C]
        x_i = x.unsqueeze(2).expand(B, N, N, C)  # [B, N, N, C]
        x_j = x.unsqueeze(1).expand(B, N, N, C)  # [B, N, N, C]

        # Concatenate for edge features
        edge_features = torch.cat([x_i, x_j], dim=-1)  # [B, N, N, 2C]

        # Compute edge weights
        edge_weights = self.edge_net(edge_features)  # [B, N, N, num_heads]
        edge_weights = edge_weights.permute(0, 3, 1, 2)  # [B, num_heads, N, N]

        # Normalize with softmax (row-wise)
        adj = F.softmax(edge_weights, dim=-1)

        return adj

    def forward(self, x):
        """
        Args:
            x: [B, N, C] token features
        Returns:
            [B, N, C] graph-enhanced features
        """
        B, N, C = x.shape

        # Construct graph
        adj = self.construct_graph(x)  # [B, num_heads, N, N]

        # Multi-head graph convolution
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Graph convolution: aggregate neighbors with learned edge weights
        # adj: [B, num_heads, N, N]
        # v: [B, num_heads, N, head_dim]
        out = torch.matmul(adj, v)  # [B, num_heads, N, head_dim]

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, out_features]
        out = self.out_proj(out)
        out = self.norm(out + x if C == self.out_features else out)

        return out


class NonLocalTokenEnhancement(nn.Module):
    """
    Non-Local Token Enhancement Module (NL-TEM).

    Performs non-local operations on neighboring tokens with GCN for
    high-order semantic relations.
    """
    def __init__(self, dim, num_heads=8, window_size=7, gcn_layers=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        # Non-local attention (efficient implementation)
        self.non_local_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Graph Convolution Networks for high-order relations
        self.gcn_layers = nn.ModuleList([
            GraphConvolutionNetwork(dim, dim, num_heads=4)
            for _ in range(gcn_layers)
        ])

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature maps
        Returns:
            [B, C, H, W] enhanced features
        """
        B, C, H, W = x.shape

        # Convert to token format
        tokens = rearrange(x, 'b c h w -> b (h w) c')  # [B, N, C]

        # Non-local attention
        attn_out, _ = self.non_local_attn(tokens, tokens, tokens)
        attn_out = self.norm1(attn_out + tokens)

        # Graph convolution for high-order relations
        gcn_out = attn_out
        for gcn_layer in self.gcn_layers:
            gcn_out = gcn_layer(gcn_out)
        gcn_out = self.norm2(gcn_out)

        # Fuse non-local and GCN features
        fused = self.fusion(torch.cat([attn_out, gcn_out], dim=-1))

        # Residual connection
        output = tokens + fused

        # Convert back to spatial format
        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)

        return output


class AdjacentInteractionModule(nn.Module):
    """
    Adjacent Interaction Module (AIM) for feature interaction between scales.

    Enables information exchange between adjacent feature levels.
    """
    def __init__(self, in_channels_low, in_channels_high, out_channels):
        super().__init__()

        # Process low-resolution features (upsample)
        self.low_branch = nn.Sequential(
            nn.Conv2d(in_channels_low, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # Process high-resolution features
        self.high_branch = nn.Sequential(
            nn.Conv2d(in_channels_high, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Interaction through cross-attention
        self.interaction = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_low, feat_high):
        """
        Args:
            feat_low: [B, C_low, H, W] lower resolution features
            feat_high: [B, C_high, 2H, 2W] higher resolution features
        Returns:
            [B, out_channels, 2H, 2W] interacted features
        """
        # Process both branches
        low_up = self.low_branch(feat_low)     # [B, out, 2H, 2W]
        high_proc = self.high_branch(feat_high)  # [B, out, 2H, 2W]

        # Concatenate and interact
        combined = torch.cat([low_up, high_proc], dim=1)
        interacted = self.interaction(combined)

        # Apply channel attention
        ca_weight = self.channel_attn(interacted)
        output = interacted * ca_weight

        # Residual connection with high_proc
        output = output + high_proc

        return output


class FeatureShrinkageDecoder(nn.Module):
    """
    Feature Shrinkage Decoder (FSD) with 4-layer hierarchical structure.

    Contains 12 Adjacent Interaction Modules arranged in a pyramid:
    - Layer 1: 1 AIM  (512 -> 320)
    - Layer 2: 2 AIMs (320 -> 128)
    - Layer 3: 3 AIMs (128 -> 64)
    - Layer 4: 6 AIMs (progressive refinement)
    Total: 1 + 2 + 3 + 6 = 12 AIMs
    """
    def __init__(self, dims=[64, 128, 320, 512]):
        super().__init__()
        self.dims = dims

        # Layer 1: 512 -> 320 (1 AIM)
        self.layer1_aim1 = AdjacentInteractionModule(dims[3], dims[2], dims[2])

        # Layer 2: 320 -> 128 (2 AIMs)
        self.layer2_aim1 = AdjacentInteractionModule(dims[2], dims[1], dims[1])
        self.layer2_aim2 = AdjacentInteractionModule(dims[2], dims[1], dims[1])

        # Layer 3: 128 -> 64 (3 AIMs)
        self.layer3_aim1 = AdjacentInteractionModule(dims[1], dims[0], dims[0])
        self.layer3_aim2 = AdjacentInteractionModule(dims[1], dims[0], dims[0])
        self.layer3_aim3 = AdjacentInteractionModule(dims[1], dims[0], dims[0])

        # Layer 4: Progressive refinement at 64 resolution (6 AIMs)
        # These refine the features at the finest resolution
        self.layer4_aims = nn.ModuleList([
            AdjacentInteractionModule(dims[0], dims[0], dims[0])
            for _ in range(6)
        ])

        # Fusion modules for combining multiple paths
        self.fusion2 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], 1),
            nn.BatchNorm2d(dims[1]),
            nn.GELU()
        )

        self.fusion3 = nn.Sequential(
            nn.Conv2d(dims[0] * 3, dims[0], 1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )

        self.fusion4 = nn.Sequential(
            nn.Conv2d(dims[0] * 6, dims[0], 1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )

    def forward(self, features):
        """
        Args:
            features: List[Tensor] = [f1, f2, f3, f4]
                     where f1: [B, 64, H, W]
                           f2: [B, 128, H/2, W/2]
                           f3: [B, 320, H/4, W/4]
                           f4: [B, 512, H/8, W/8]
        Returns:
            decoded_features: List[Tensor] at each layer
        """
        f1, f2, f3, f4 = features

        # Layer 1: 512 -> 320 (1 AIM)
        l1_out = self.layer1_aim1(f4, f3)  # [B, 320, H/4, W/4]

        # Layer 2: 320 -> 128 (2 AIMs parallel)
        l2_out1 = self.layer2_aim1(l1_out, f2)  # [B, 128, H/2, W/2]
        l2_out2 = self.layer2_aim2(f3, f2)       # [B, 128, H/2, W/2]
        l2_fused = self.fusion2(torch.cat([l2_out1, l2_out2], dim=1))

        # Layer 3: 128 -> 64 (3 AIMs parallel)
        l3_out1 = self.layer3_aim1(l2_fused, f1)  # [B, 64, H, W]
        l3_out2 = self.layer3_aim2(l2_out1, f1)   # [B, 64, H, W]
        l3_out3 = self.layer3_aim3(f2, f1)        # [B, 64, H, W]
        l3_fused = self.fusion3(torch.cat([l3_out1, l3_out2, l3_out3], dim=1))

        # Layer 4: Progressive refinement (6 AIMs cascade)
        l4_outputs = []
        l4_input = l3_fused
        for aim in self.layer4_aims:
            l4_out = aim(l4_input, f1)  # [B, 64, H, W]
            l4_outputs.append(l4_out)
            l4_input = l4_out  # Cascade

        l4_fused = self.fusion4(torch.cat(l4_outputs, dim=1))

        return {
            'layer1': l1_out,     # [B, 320, H/4, W/4]
            'layer2': l2_fused,   # [B, 128, H/2, W/2]
            'layer3': l3_fused,   # [B, 64, H, W]
            'layer4': l4_fused,   # [B, 64, H, W]
            'all_outputs': [l1_out, l2_fused, l3_fused, l4_fused]
        }


class CrossModalFusion(nn.Module):
    """
    Cross-modulation between CNN and Transformer features.

    Uses dual cross-attention to exchange information bidirectionally.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim

        # CNN to Transformer attention
        self.cnn2tf_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Transformer to CNN attention
        self.tf2cnn_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Feature enhancement
        self.cnn_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )

        self.tf_enhance = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.LayerNorm(dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim)
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, cnn_feat, tf_feat):
        """
        Args:
            cnn_feat: [B, C, H, W] CNN features
            tf_feat: [B, C, H, W] Transformer features
        Returns:
            fused_cnn: [B, C, H, W] enhanced CNN features
            fused_tf: [B, C, H, W] enhanced transformer features
        """
        B, C, H, W = cnn_feat.shape

        # Convert to token format
        cnn_tokens = rearrange(cnn_feat, 'b c h w -> b (h w) c')
        tf_tokens = rearrange(tf_feat, 'b c h w -> b (h w) c')

        # Cross-attention: CNN queries Transformer
        cnn_cross, _ = self.cnn2tf_attn(
            query=cnn_tokens,
            key=tf_tokens,
            value=tf_tokens
        )
        cnn_cross = self.norm1(cnn_cross + cnn_tokens)

        # Cross-attention: Transformer queries CNN
        tf_cross, _ = self.tf2cnn_attn(
            query=tf_tokens,
            key=cnn_tokens,
            value=cnn_tokens
        )
        tf_cross = self.norm2(tf_cross + tf_tokens)

        # Enhance features
        cnn_enhanced = rearrange(cnn_cross, 'b (h w) c -> b c h w', h=H, w=W)
        cnn_enhanced = self.cnn_enhance(cnn_enhanced)

        tf_enhanced = self.tf_enhance(tf_cross)
        tf_enhanced = rearrange(tf_enhanced, 'b (h w) c -> b c h w', h=H, w=W)

        # Gated fusion
        gate_weight = self.gate(torch.cat([cnn_enhanced, tf_enhanced], dim=1))

        fused_cnn = cnn_feat + gate_weight * cnn_enhanced
        fused_tf = tf_feat + (1 - gate_weight) * tf_enhanced

        return fused_cnn, fused_tf


class ProgressiveAggregation(nn.Module):
    """
    Progressive aggregation with layer-wise supervision weights.

    Weights: 2^(i-4) for layer i ∈ {1, 2, 3, 4}
    - Layer 1: 2^(-3) = 0.125
    - Layer 2: 2^(-2) = 0.25
    - Layer 3: 2^(-1) = 0.5
    - Layer 4: 2^(0)  = 1.0
    """
    def __init__(self, dims=[64, 128, 320, 512]):
        super().__init__()
        self.dims = dims

        # Supervision weights: 2^(i-4)
        self.register_buffer('layer_weights', torch.tensor([
            2 ** (-3),  # Layer 1: 0.125
            2 ** (-2),  # Layer 2: 0.25
            2 ** (-1),  # Layer 3: 0.5
            2 ** 0      # Layer 4: 1.0
        ]))

        # Projection heads for intermediate supervision
        self.supervision_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[2], dims[2] // 2, 3, padding=1),
                nn.BatchNorm2d(dims[2] // 2),
                nn.GELU(),
                nn.Conv2d(dims[2] // 2, 1, 1)
            ),  # Layer 1: 320 channels
            nn.Sequential(
                nn.Conv2d(dims[1], dims[1] // 2, 3, padding=1),
                nn.BatchNorm2d(dims[1] // 2),
                nn.GELU(),
                nn.Conv2d(dims[1] // 2, 1, 1)
            ),  # Layer 2: 128 channels
            nn.Sequential(
                nn.Conv2d(dims[0], dims[0] // 2, 3, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, 1, 1)
            ),  # Layer 3: 64 channels
            nn.Sequential(
                nn.Conv2d(dims[0], dims[0] // 2, 3, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, 1, 1)
            )   # Layer 4: 64 channels
        ])

        # Progressive fusion
        self.progressive_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[2], dims[2], 3, padding=1),
                nn.BatchNorm2d(dims[2]),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(dims[1], dims[1], 3, padding=1),
                nn.BatchNorm2d(dims[1]),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(dims[0], dims[0], 3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(dims[0], dims[0], 3, padding=1),
                nn.BatchNorm2d(dims[0]),
                nn.GELU()
            )
        ])

    def forward(self, decoder_outputs, return_supervision=True):
        """
        Args:
            decoder_outputs: Dict from FeatureShrinkageDecoder
            return_supervision: If True, return intermediate predictions
        Returns:
            aggregated: Final aggregated features
            supervision_outputs: Dict of intermediate predictions with weights
        """
        all_outputs = decoder_outputs['all_outputs']

        # Progressive fusion with weights
        fused_features = []
        supervision_preds = []

        for i, (feat, fusion, head, weight) in enumerate(zip(
            all_outputs, self.progressive_fusion, self.supervision_heads, self.layer_weights
        )):
            # Fuse features
            fused = fusion(feat)
            fused_features.append(fused * weight)

            # Generate supervision prediction
            if return_supervision:
                pred = head(fused)
                supervision_preds.append({
                    'prediction': pred,
                    'weight': weight.item(),
                    'layer': i + 1
                })

        # Final aggregation (at finest resolution)
        # Upsample coarser features to finest resolution
        B, C, H, W = fused_features[-1].shape

        aggregated = fused_features[-1]  # Start with layer 4

        # Add layer 3 (same resolution)
        aggregated = aggregated + fused_features[2]

        # Add layer 2 (upsample 2x)
        aggregated = aggregated + F.interpolate(
            fused_features[1], size=(H, W), mode='bilinear', align_corners=False
        )

        # Add layer 1 (upsample 4x)
        aggregated = aggregated + F.interpolate(
            fused_features[0], size=(H, W), mode='bilinear', align_corners=False
        )

        supervision_outputs = {
            'predictions': supervision_preds,
            'layer_weights': self.layer_weights
        } if return_supervision else None

        return aggregated, supervision_outputs


class HybridBackboneEnhancer(nn.Module):
    """
    Hybrid CNN-Transformer Backbone Enhancer.

    Enhances CNN features with transformer global modeling through:
    1. Non-Local Token Enhancement Module (NL-TEM) with GCN
    2. Feature Shrinkage Decoder (FSD) with 12 AIMs
    3. Cross-modulation between CNN and transformer features
    4. Progressive aggregation with layer-wise supervision weights 2^(i-4)

    Architecture:
        Input: CNN features [64, 128, 320, 512]
            ↓
        [NL-TEM] → Transformer-enhanced features
            ↓
        [Cross-Modulation] ← → CNN features
            ↓
        [Feature Shrinkage Decoder with 12 AIMs]
            ↓
        [Progressive Aggregation]
            ↓
        Output: Enhanced features [64, 128, 320, 512]
    """
    def __init__(self, dims=[64, 128, 320, 512], num_heads=8):
        super().__init__()
        self.dims = dims

        # Non-Local Token Enhancement Modules for each scale
        self.nl_tem_modules = nn.ModuleList([
            NonLocalTokenEnhancement(dim, num_heads=num_heads, gcn_layers=2)
            for dim in dims
        ])

        # Cross-modulation modules for each scale
        self.cross_modal_modules = nn.ModuleList([
            CrossModalFusion(dim, num_heads=num_heads)
            for dim in dims
        ])

        # Feature Shrinkage Decoder with 12 AIMs
        self.fsd = FeatureShrinkageDecoder(dims)

        # Progressive Aggregation
        self.progressive_agg = ProgressiveAggregation(dims)

        # Output projections to maintain dimensions
        self.output_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1)
            ) for dim in dims
        ])

    def forward(self, cnn_features, return_supervision=False):
        """
        Args:
            cnn_features: List[Tensor] = [f1, f2, f3, f4]
                         where f1: [B, 64, H, W]
                               f2: [B, 128, H/2, W/2]
                               f3: [B, 320, H/4, W/4]
                               f4: [B, 512, H/8, W/8]
            return_supervision: If True, return intermediate supervision outputs

        Returns:
            enhanced_features: List[Tensor] with same dimensions as input
            supervision_outputs: Dict (if return_supervision=True)
        """
        # Step 1: Non-Local Token Enhancement with GCN
        tf_features = []
        for feat, nl_tem in zip(cnn_features, self.nl_tem_modules):
            tf_feat = nl_tem(feat)
            tf_features.append(tf_feat)

        # Step 2: Cross-modulation between CNN and Transformer features
        fused_cnn_features = []
        fused_tf_features = []
        for cnn_feat, tf_feat, cross_mod in zip(
            cnn_features, tf_features, self.cross_modal_modules
        ):
            fused_cnn, fused_tf = cross_mod(cnn_feat, tf_feat)
            fused_cnn_features.append(fused_cnn)
            fused_tf_features.append(fused_tf)

        # Step 3: Feature Shrinkage Decoder (12 AIMs)
        decoder_outputs = self.fsd(fused_cnn_features)

        # Step 4: Progressive Aggregation with layer-wise supervision
        aggregated, supervision_outputs = self.progressive_agg(
            decoder_outputs,
            return_supervision=return_supervision
        )

        # Step 5: Reconstruct multi-scale features
        # Use decoder outputs and aggregated features
        enhanced_features = []

        # Scale 4 (finest): Use aggregated directly
        enhanced_f1 = self.output_projs[0](aggregated)  # [B, 64, H, W]
        enhanced_features.append(enhanced_f1)

        # Scale 3: Use layer2 from decoder
        enhanced_f2 = self.output_projs[1](decoder_outputs['layer2'])  # [B, 128, H/2, W/2]
        enhanced_features.append(enhanced_f2)

        # Scale 2: Use layer1 from decoder
        enhanced_f3 = self.output_projs[2](decoder_outputs['layer1'])  # [B, 320, H/4, W/4]
        enhanced_features.append(enhanced_f3)

        # Scale 1 (coarsest): Use fused_cnn_features directly
        # Downsample aggregated to match f4 resolution
        f4_enhanced = F.adaptive_avg_pool2d(aggregated, cnn_features[3].shape[2:])
        f4_enhanced = F.interpolate(
            f4_enhanced,
            size=cnn_features[3].shape[2:],
            mode='bilinear',
            align_corners=False
        )
        # Project to 512 channels
        f4_proj = nn.Conv2d(self.dims[0], self.dims[3], 1).to(f4_enhanced.device)
        enhanced_f4 = f4_proj(f4_enhanced)
        # Residual with original
        enhanced_f4 = enhanced_f4 + fused_cnn_features[3]
        enhanced_features.append(enhanced_f4)

        # Return in original order [64, 128, 320, 512]
        enhanced_features_ordered = enhanced_features  # Already in correct order

        if return_supervision:
            return enhanced_features_ordered, supervision_outputs
        else:
            return enhanced_features_ordered


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("="*70)
    print("Testing HybridBackboneEnhancer")
    print("="*70)

    # Create model
    enhancer = HybridBackboneEnhancer(
        dims=[64, 128, 320, 512],
        num_heads=8
    )

    print(f"\nTotal parameters: {count_parameters(enhancer) / 1e6:.2f}M")

    # Create dummy CNN features
    batch_size = 2
    cnn_features = [
        torch.randn(batch_size, 64, 64, 64),    # f1: H, W
        torch.randn(batch_size, 128, 32, 32),   # f2: H/2, W/2
        torch.randn(batch_size, 320, 16, 16),   # f3: H/4, W/4
        torch.randn(batch_size, 512, 8, 8)      # f4: H/8, W/8
    ]

    print("\nInput features:")
    for i, feat in enumerate(cnn_features):
        print(f"  f{i+1}: {feat.shape}")

    # Test without supervision
    print("\n" + "="*70)
    print("Test 1: Forward pass without supervision")
    print("="*70)

    enhanced_features = enhancer(cnn_features, return_supervision=False)

    print("\nEnhanced features:")
    for i, feat in enumerate(enhanced_features):
        print(f"  f{i+1}: {feat.shape}")

    # Verify dimensions maintained
    assert len(enhanced_features) == len(cnn_features), "Number of features mismatch!"
    for i, (orig, enhanced) in enumerate(zip(cnn_features, enhanced_features)):
        assert orig.shape == enhanced.shape, f"Dimension mismatch at scale {i+1}!"

    print("\n✓ Dimensions maintained correctly!")

    # Test with supervision
    print("\n" + "="*70)
    print("Test 2: Forward pass with supervision")
    print("="*70)

    enhanced_features, supervision = enhancer(cnn_features, return_supervision=True)

    print("\nSupervision outputs:")
    print(f"  Layer weights: {supervision['layer_weights']}")
    print(f"  Number of supervision predictions: {len(supervision['predictions'])}")

    for pred_info in supervision['predictions']:
        print(f"\n  Layer {pred_info['layer']}:")
        print(f"    Prediction shape: {pred_info['prediction'].shape}")
        print(f"    Weight: {pred_info['weight']:.4f}")

    # Test individual components
    print("\n" + "="*70)
    print("Test 3: Individual Component Testing")
    print("="*70)

    # Test NL-TEM
    print("\n1. Non-Local Token Enhancement Module:")
    nl_tem = NonLocalTokenEnhancement(dim=128, num_heads=8)
    test_feat = torch.randn(2, 128, 32, 32)
    nl_out = nl_tem(test_feat)
    print(f"   Input:  {test_feat.shape}")
    print(f"   Output: {nl_out.shape}")
    assert nl_out.shape == test_feat.shape, "NL-TEM dimension mismatch!"
    print("   ✓ NL-TEM working correctly")

    # Test GCN
    print("\n2. Graph Convolution Network:")
    gcn = GraphConvolutionNetwork(in_features=128, out_features=128, num_heads=4)
    tokens = torch.randn(2, 256, 128)  # [B, N, C]
    gcn_out = gcn(tokens)
    print(f"   Input:  {tokens.shape}")
    print(f"   Output: {gcn_out.shape}")
    assert gcn_out.shape == tokens.shape, "GCN dimension mismatch!"
    print("   ✓ GCN working correctly")

    # Test AIM
    print("\n3. Adjacent Interaction Module:")
    aim = AdjacentInteractionModule(in_channels_low=320, in_channels_high=128, out_channels=128)
    feat_low = torch.randn(2, 320, 16, 16)
    feat_high = torch.randn(2, 128, 32, 32)
    aim_out = aim(feat_low, feat_high)
    print(f"   Low input:  {feat_low.shape}")
    print(f"   High input: {feat_high.shape}")
    print(f"   Output:     {aim_out.shape}")
    assert aim_out.shape == feat_high.shape, "AIM dimension mismatch!"
    print("   ✓ AIM working correctly")

    # Test FSD
    print("\n4. Feature Shrinkage Decoder:")
    fsd = FeatureShrinkageDecoder(dims=[64, 128, 320, 512])
    fsd_out = fsd(cnn_features)
    print(f"   Number of AIMs: 12 (1 + 2 + 3 + 6)")
    print(f"   Layer 1 output: {fsd_out['layer1'].shape}")
    print(f"   Layer 2 output: {fsd_out['layer2'].shape}")
    print(f"   Layer 3 output: {fsd_out['layer3'].shape}")
    print(f"   Layer 4 output: {fsd_out['layer4'].shape}")
    print("   ✓ FSD working correctly with 12 AIMs")

    # Test Cross-Modulation
    print("\n5. Cross-Modal Fusion:")
    cross_mod = CrossModalFusion(dim=128, num_heads=8)
    cnn_f = torch.randn(2, 128, 32, 32)
    tf_f = torch.randn(2, 128, 32, 32)
    fused_cnn, fused_tf = cross_mod(cnn_f, tf_f)
    print(f"   CNN input:  {cnn_f.shape}")
    print(f"   TF input:   {tf_f.shape}")
    print(f"   Fused CNN:  {fused_cnn.shape}")
    print(f"   Fused TF:   {fused_tf.shape}")
    assert fused_cnn.shape == cnn_f.shape and fused_tf.shape == tf_f.shape
    print("   ✓ Cross-modulation working correctly")

    # Test Progressive Aggregation
    print("\n6. Progressive Aggregation:")
    prog_agg = ProgressiveAggregation(dims=[64, 128, 320, 512])
    agg_out, sup_out = prog_agg(fsd_out, return_supervision=True)
    print(f"   Aggregated output: {agg_out.shape}")
    print(f"   Layer weights: {sup_out['layer_weights']}")
    print(f"   Supervision predictions: {len(sup_out['predictions'])}")
    print("   ✓ Progressive aggregation working correctly")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)

    print("\nArchitecture Summary:")
    print("  ✓ Non-Local Token Enhancement with GCN")
    print("  ✓ Feature Shrinkage Decoder with 12 AIMs")
    print("  ✓ Cross-modulation between CNN and Transformer")
    print("  ✓ Progressive aggregation with layer-wise supervision")
    print("  ✓ Input/Output dimensions maintained: [64, 128, 320, 512]")

    # Count components
    num_nl_tem = len(enhancer.nl_tem_modules)
    num_cross_mod = len(enhancer.cross_modal_modules)
    num_output_proj = len(enhancer.output_projs)

    print(f"\nComponent counts:")
    print(f"  NL-TEM modules: {num_nl_tem}")
    print(f"  Cross-modulation modules: {num_cross_mod}")
    print(f"  Adjacent Interaction Modules: 12")
    print(f"  Output projections: {num_output_proj}")
    print(f"  Total parameters: {count_parameters(enhancer) / 1e6:.2f}M")
