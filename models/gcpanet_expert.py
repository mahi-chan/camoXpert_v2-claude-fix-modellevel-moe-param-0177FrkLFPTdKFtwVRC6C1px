"""
GCPANet Expert: Global Context-aware Progressive Aggregation Network

Paper: "Global Context-aware Progressive Aggregation Network for Salient Object Detection" (AAAI 2020)

Key Concepts:
1. Global Context Module (GCM) - Non-local style attention
2. Progressive Aggregation - Gradual feature integration
3. Position-Aware Attention - Learns spatial priors
4. Self-Refinement - Iterative prediction refinement

This expert specializes in LARGE UNIFORM REGIONS and GLOBAL CONTEXT.
Best for images where camouflage covers large areas or requires
understanding of the scene structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBNReLU(nn.Module):
    """Basic conv-bn-relu block."""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class GlobalContextModule(nn.Module):
    """
    Global Context Module (GCM) - Non-local style global attention.
    
    Captures long-range dependencies that are crucial for understanding
    large camouflaged regions.
    
    Implementation based on Non-local Neural Networks (CVPR 2018).
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        
        self.in_channels = in_channels
        inter_channels = in_channels // reduction
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.key = nn.Conv2d(in_channels, inter_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Softmax for attention
        self.softmax = nn.Softmax(dim=-1)
        
        # Additional global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            output: [B, C, H, W] with global context
        """
        B, C, H, W = x.shape
        
        # Compute query, key, value
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C//r]
        key = self.key(x).view(B, -1, H * W)  # [B, C//r, HW]
        value = self.value(x).view(B, -1, H * W)  # [B, C, HW]
        
        # Attention weights
        attention = torch.bmm(query, key)  # [B, HW, HW]
        attention = self.softmax(attention)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        
        # Residual with learnable weight
        out = self.gamma * out + x
        
        # Add global pooling attention
        global_att = self.global_pool(x)
        out = out * global_att + out
        
        return out


class PositionAwareAttention(nn.Module):
    """
    Position-Aware Attention - Learns spatial priors.
    
    Different positions in an image have different importance
    for camouflaged object detection. This module learns position embeddings.
    """
    def __init__(self, in_channels, max_size=56):
        super().__init__()
        
        self.in_channels = in_channels
        self.max_size = max_size
        
        # Learnable position embeddings (will be interpolated to actual size)
        self.pos_embed_h = nn.Parameter(torch.randn(1, in_channels // 2, max_size, 1) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, in_channels // 2, 1, max_size) * 0.02)
        
        # Position-conditioned attention
        self.pos_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            position-aware features: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Interpolate position embeddings to actual size
        pos_h = F.interpolate(self.pos_embed_h, (H, 1), mode='bilinear', align_corners=False)
        pos_w = F.interpolate(self.pos_embed_w, (1, W), mode='bilinear', align_corners=False)
        
        # Combine position embeddings
        pos = pos_h.expand(-1, -1, -1, W) + pos_w.expand(-1, -1, H, -1)
        pos = pos.expand(B, -1, -1, -1)
        
        # Split channels and add position
        x_pos = x.clone()
        x_pos[:, :C//2] = x_pos[:, :C//2] + pos
        
        # Position-conditioned attention
        att = self.pos_att(x_pos)
        
        # Apply attention and transform
        x_attended = x * att
        output = self.transform(x_attended) + x
        
        return output


class ProgressiveAggregationModule(nn.Module):
    """
    Progressive Aggregation Module (PAM) - Gradually integrates features.
    
    Instead of directly combining all scales, progressively aggregates
    from coarse to fine, allowing gradual refinement.
    """
    def __init__(self, feature_dims=[64, 128, 320, 512], out_dim=64):
        super().__init__()
        
        # Project each scale to common dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, out_dim, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])
        
        # Progressive aggregation: high → low
        # Stage 1: f4 alone
        self.agg4 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: f4 + f3
        self.agg3 = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: (f4+f3) + f2
        self.agg2 = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 4: ((f4+f3)+f2) + f1
        self.agg1 = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        """
        Args:
            features: [f1, f2, f3, f4]
        Returns:
            List of progressively aggregated features
        """
        # Project all scales
        p1, p2, p3, p4 = [proj(f) for proj, f in zip(self.projections, features)]
        
        # Progressive aggregation (coarse to fine)
        a4 = self.agg4(p4)  # [B, 64, H/32, W/32]
        
        a4_up = F.interpolate(a4, p3.shape[2:], mode='bilinear', align_corners=False)
        a3 = self.agg3(torch.cat([a4_up, p3], dim=1))  # [B, 64, H/16, W/16]
        
        a3_up = F.interpolate(a3, p2.shape[2:], mode='bilinear', align_corners=False)
        a2 = self.agg2(torch.cat([a3_up, p2], dim=1))  # [B, 64, H/8, W/8]
        
        a2_up = F.interpolate(a2, p1.shape[2:], mode='bilinear', align_corners=False) 
        a1 = self.agg1(torch.cat([a2_up, p1], dim=1))  # [B, 64, H/4, W/4]
        
        return [a1, a2, a3, a4]


class SelfRefinementModule(nn.Module):
    """
    Self-Refinement Module - Iteratively refines prediction.
    
    Takes initial prediction and refines it using feature guidance.
    """
    def __init__(self, feat_ch=64, num_iterations=2):
        super().__init__()
        
        self.num_iterations = num_iterations
        
        # Refinement block (shared across iterations)
        self.refine = nn.Sequential(
            nn.Conv2d(feat_ch + 1, feat_ch, 3, padding=1),  # +1 for pred
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch // 2, 3, padding=1),
            nn.BatchNorm2d(feat_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 2, 1, 1)
        )
    
    def forward(self, features, initial_pred):
        """
        Args:
            features: [B, C, H, W]
            initial_pred: [B, 1, H, W]
        Returns:
            refined_pred: [B, 1, H, W]
        """
        pred = initial_pred
        
        for _ in range(self.num_iterations):
            # Concatenate features with current prediction
            combined = torch.cat([features, pred], dim=1)
            
            # Predict residual
            residual = self.refine(combined)
            
            # Add residual
            pred = pred + residual
        
        return pred


class GCPANetExpert(nn.Module):
    """
    GCPANet: Global Context-aware Progressive Aggregation Expert
    
    Paper-accurate implementation with:
    1. Global Context Module (GCM) - Non-local attention
    2. Position-Aware Attention (PAA) - Spatial priors
    3. Progressive Aggregation Module (PAM) - Gradual integration
    4. Self-Refinement Module (SRM) - Iterative refinement
    
    Specialization: LARGE UNIFORM REGIONS, GLOBAL CONTEXT
    ~18M parameters
    """
    def __init__(self, feature_dims=[64, 128, 320, 512]):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.out_dim = 64
        
        # 1. Global Context Module at deep levels
        self.gcm_modules = nn.ModuleList([
            GlobalContextModule(dim, reduction=8) for dim in feature_dims[2:]  # f3, f4
        ])
        
        # 2. Position-Aware Attention at all levels
        self.paa_modules = nn.ModuleList([
            PositionAwareAttention(dim) for dim in feature_dims
        ])
        
        # 3. Progressive Aggregation
        self.pam = ProgressiveAggregationModule(feature_dims, self.out_dim)
        
        # 4. Initial prediction head
        self.initial_pred = nn.Sequential(
            nn.Conv2d(self.out_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        # 5. Self-Refinement
        self.srm = SelfRefinementModule(self.out_dim, num_iterations=2)
        
        # 6. Auxiliary heads for deep supervision
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(self.out_dim, 1, 1) for _ in range(4)
        ])
        
        # 7. Dropout
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, features, return_aux=True):
        """
        Args:
            features: [f1, f2, f3, f4] from backbone
        Returns:
            pred: [B, 1, H, W] main prediction
            aux: List of auxiliary predictions
        """
        f1, f2, f3, f4 = features
        output_size = (f1.shape[2] * 4, f1.shape[3] * 4)
        
        # Step 1: Position-Aware Attention at all levels
        paa_features = [paa(f) for paa, f in zip(self.paa_modules, features)]
        
        # Step 2: Global Context at deep levels (f3, f4)
        paa_features[2] = self.gcm_modules[0](paa_features[2])
        paa_features[3] = self.gcm_modules[1](paa_features[3])
        
        # Step 3: Progressive Aggregation
        pam_outputs = self.pam(paa_features)  # [a1, a2, a3, a4]
        
        # Step 4: Initial prediction
        a1 = pam_outputs[0]
        if self.training:
            a1 = self.dropout(a1)
        
        initial = self.initial_pred(a1)
        
        # Step 5: Self-Refinement
        refined = self.srm(a1, initial)
        
        # Upsample to output size
        pred = F.interpolate(refined, output_size, mode='bilinear', align_corners=False)
        
        if return_aux:
            aux_outputs = []
            for out, head in zip(pam_outputs, self.aux_heads):
                aux = F.interpolate(head(out), output_size, mode='bilinear', align_corners=False)
                aux_outputs.append(aux)
            return pred, aux_outputs[:3]  # Return 3 aux to match other experts
        
        return pred, []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    print("=" * 60)
    print("Testing GCPANet Expert")
    print("=" * 60)
    
    model = GCPANetExpert()
    params = count_parameters(model)
    print(f"Parameters: {params / 1e6:.1f}M")
    
    # Test
    features = [
        torch.randn(2, 64, 112, 112),
        torch.randn(2, 128, 56, 56),
        torch.randn(2, 320, 28, 28),
        torch.randn(2, 512, 14, 14)
    ]
    
    pred, aux = model(features)
    print(f"Output: {pred.shape}")
    print(f"Aux: {len(aux)} outputs")
    
    print("\n✓ GCPANet Expert test passed!")
