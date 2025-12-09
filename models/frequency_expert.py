"""
Production-Ready FrequencyExpert for Camouflaged Object Detection

This module implements a frequency-domain expert architecture with:
1. DeepWaveletDecomposition - Learnable Haar wavelets for frequency separation
2. HighFrequencyAttention - Texture and edge enhancement
3. LowFrequencyAttention - Semantic content processing
4. ODEEdgeReconstruction - Second-order Runge-Kutta edge evolution
5. Multi-scale integration for PVT backbone features [64, 128, 320, 512]
6. Deep supervision with auxiliary outputs

Author: CamoXpert Team
Compatible with: PyTorch 2.0+, PVT-v2 backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional


# ============================================================================
# Helper Modules
# ============================================================================

class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D feature maps (channels-first format).

    Normalizes over the channel dimension while preserving spatial structure.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        return x


class ChannelAttention(nn.Module):
    """
    Channel attention module using global pooling and squeeze-excitation.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module using channel pooling and convolution.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


# ============================================================================
# Core Frequency Components
# ============================================================================

class DeepWaveletDecomposition(nn.Module):
    """
    Deep Wavelet Decomposition using learnable Haar wavelets.

    Decomposes input into 4 frequency subbands:
    - LL: Low-low (approximation, semantic content)
    - LH: Low-high (horizontal edges)
    - HL: High-low (vertical edges)
    - HH: High-high (diagonal edges)

    Args:
        channels: Number of input channels
        learnable: If True, wavelet filters are learnable (default: True)
    """
    def __init__(self, channels: int, learnable: bool = True):
        super().__init__()
        self.channels = channels
        self.learnable = learnable

        # Learnable wavelet filters initialized with Haar basis
        self.ll_conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1,
                                 bias=False, groups=channels)
        self.lh_conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1,
                                 bias=False, groups=channels)
        self.hl_conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1,
                                 bias=False, groups=channels)
        self.hh_conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1,
                                 bias=False, groups=channels)

        # Initialize with Haar wavelet patterns
        self._initialize_haar_wavelets()

        # Optional: Make wavelets non-learnable
        if not learnable:
            for conv in [self.ll_conv, self.lh_conv, self.hl_conv, self.hh_conv]:
                conv.weight.requires_grad = False

    def _initialize_haar_wavelets(self):
        """Initialize convolution kernels with Haar wavelet basis."""
        # Define 3x3 Haar-like wavelet patterns
        # LL: Low-pass (averaging)
        ll_kernel = torch.ones(3, 3) / 9.0

        # LH: Horizontal edges (vertical high-pass)
        lh_kernel = torch.tensor([
            [-1.0, -1.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  1.0,  1.0]
        ]) / 6.0

        # HL: Vertical edges (horizontal high-pass)
        hl_kernel = torch.tensor([
            [-1.0,  0.0,  1.0],
            [-1.0,  0.0,  1.0],
            [-1.0,  0.0,  1.0]
        ]) / 6.0

        # HH: Diagonal edges (both high-pass)
        hh_kernel = torch.tensor([
            [-1.0,  0.0,  1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  0.0, -1.0]
        ]) / 4.0

        # Apply kernels to all channels (depthwise)
        kernels = [ll_kernel, lh_kernel, hl_kernel, hh_kernel]
        convs = [self.ll_conv, self.lh_conv, self.hl_conv, self.hh_conv]

        for conv, kernel in zip(convs, kernels):
            # Initialize each channel with the same kernel (depthwise)
            for i in range(self.channels):
                conv.weight.data[i, 0] = kernel

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose input into frequency subbands.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Dictionary with keys 'll', 'lh', 'hl', 'hh' containing subbands
        """
        return {
            'll': self.ll_conv(x),  # Low-frequency approximation
            'lh': self.lh_conv(x),  # Horizontal details
            'hl': self.hl_conv(x),  # Vertical details
            'hh': self.hh_conv(x)   # Diagonal details
        }


class HighFrequencyAttention(nn.Module):
    """
    High-Frequency Attention for texture and edge enhancement.

    Uses residual blocks with joint spatial-channel attention to enhance
    high-frequency details (edges, textures) while suppressing noise.

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio for attention (default: 16)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # Residual feature extraction blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels)
        )

        # Joint spatial-channel attention
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size=7)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhance high-frequency features.

        Args:
            x: High-frequency features [B, C, H, W]

        Returns:
            Enhanced high-frequency features [B, C, H, W]
        """
        # Residual block 1
        identity = x
        out = self.conv1(x) + identity
        out = self.gelu(out)

        # Residual block 2
        identity = out
        out = self.conv2(out) + identity
        out = self.gelu(out)

        # Apply joint attention
        ca = self.channel_att(out)
        sa = self.spatial_att(out)
        out = out * ca * sa

        return out


class LowFrequencyAttention(nn.Module):
    """
    Low-Frequency Attention for semantic content processing.

    Uses instance normalization and positional encoding to process low-frequency
    components while suppressing redundant background information.

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 4)
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        # Instance normalization to reduce color/illumination bias
        self.instance_norm = nn.InstanceNorm2d(channels, affine=True)

        # Positional normalization
        self.pos_norm = nn.Sequential(
            nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
            nn.GELU()
        )

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels)
        )

        # Suppression gate to reduce redundancy
        self.suppression_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process low-frequency features.

        Args:
            x: Low-frequency features [B, C, H, W]

        Returns:
            Refined low-frequency features [B, C, H, W]
        """
        # Normalize to reduce illumination/color bias
        out = self.instance_norm(x)
        out = self.pos_norm(out)

        # Refine features with residual connection
        identity = out
        out = self.refine(out) + identity
        out = self.gelu(out)

        # Apply suppression gate
        gate = self.suppression_gate(out)
        out = out * gate

        return out


class ODEEdgeReconstruction(nn.Module):
    """
    ODE-based Edge Reconstruction using 2nd-order Runge-Kutta solver.

    Models edge evolution as an ODE: dx/dt = f(x, t)
    Solved using RK2 (Heun's method) for numerical stability.

    The dynamics function f(x, t) learns to evolve edge features toward
    cleaner, more coherent edge maps.

    Args:
        channels: Number of input channels
        num_steps: Number of ODE integration steps (default: 2)
        dt: Time step size (default: 0.1)
    """
    def __init__(self, channels: int, num_steps: int = 2, dt: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_steps = num_steps

        # Edge dynamics function f(x, t)
        self.dynamics_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels)
        )

        # Learnable time step (bounded to ensure stability)
        self.dt = nn.Parameter(torch.tensor(dt))

        # Damping factor for stability
        self.damping = nn.Parameter(torch.tensor(0.1))

        # Potential energy term (Hamiltonian formulation)
        self.potential = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct edges using RK2 ODE solver.

        Args:
            x: High-frequency edge features [B, C, H, W]

        Returns:
            Reconstructed edge features [B, C, H, W]
        """
        # Ensure dt is positive and bounded
        dt = torch.clamp(self.dt, 0.01, 0.5)

        # Initial state
        x_curr = x

        # RK2 integration steps
        for _ in range(self.num_steps):
            # Step 1: Compute k1 = f(x_n)
            k1 = self.dynamics_net(x_curr)

            # Step 2: Compute k2 = f(x_n + dt * k1)
            x_temp = x_curr + dt * k1
            k2 = self.dynamics_net(x_temp)

            # Step 3: Update x_{n+1} = x_n + dt/2 * (k1 + k2)
            x_next = x_curr + (dt / 2.0) * (k1 + k2)

            # Apply potential energy for Hamiltonian stability
            potential_term = self.potential(x_next)
            x_next = x_next + potential_term * torch.sigmoid(self.damping)

            x_curr = x_next

        return x_curr


# ============================================================================
# Main FrequencyExpert Architecture
# ============================================================================

class FrequencyExpert(nn.Module):
    """
    Complete Frequency Expert for Camouflaged Object Detection.

    Processes features through frequency decomposition, specialized attention,
    and ODE-based edge reconstruction to enhance camouflaged object detection.

    Architecture Flow:
        Input [B, C, H, W]
            ↓
        [Wavelet Decomposition] → {LL, LH, HL, HH}
            ↓                          ↓
        [Low-Freq Attention]    [High-Freq Attention] × 3
            ↓                          ↓
        [ODE Edge Reconstruction]
            ↓
        [Feature Fusion]
            ↓
        Output [B, C, H, W] + Auxiliary Outputs

    Args:
        in_channels: Input feature channels
        out_channels: Output feature channels (default: same as input)
        reduction: Attention reduction ratio (default: 16)
        ode_steps: ODE integration steps (default: 2)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 reduction: int = 16,
                 ode_steps: int = 2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        # 1. Wavelet Decomposition
        self.wavelet_decomp = DeepWaveletDecomposition(in_channels, learnable=True)

        # 2. Low-Frequency Attention
        self.low_freq_att = LowFrequencyAttention(in_channels, reduction=4)

        # 3. High-Frequency Attention (separate for each subband)
        self.high_freq_att_lh = HighFrequencyAttention(in_channels, reduction)
        self.high_freq_att_hl = HighFrequencyAttention(in_channels, reduction)
        self.high_freq_att_hh = HighFrequencyAttention(in_channels, reduction)

        # 4. ODE Edge Reconstruction
        self.ode_edge_recon = ODEEdgeReconstruction(in_channels, num_steps=ode_steps)

        # 5. Feature Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1, bias=False),
            LayerNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1, bias=False),
            LayerNorm2d(self.out_channels)
        )

        # 6. Deep Supervision Heads
        self.aux_head_low = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        self.aux_head_high = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
            LayerNorm2d(self.out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Forward pass through FrequencyExpert.

        Args:
            x: Input features [B, C, H, W]
            return_aux: If True, return auxiliary outputs for deep supervision

        Returns:
            If return_aux=False:
                Enhanced features [B, C_out, H, W]
            If return_aux=True:
                Tuple of (enhanced_features, aux_outputs) where aux_outputs contains:
                    - 'low_freq_pred': Low-frequency prediction [B, 1, H, W]
                    - 'high_freq_pred': High-frequency edge map [B, 1, H, W]
                    - 'decomposition': Wavelet components dict
        """
        # Step 1: Wavelet Decomposition
        decomp = self.wavelet_decomp(x)
        ll = decomp['ll']  # Low-frequency
        lh = decomp['lh']  # Horizontal edges
        hl = decomp['hl']  # Vertical edges
        hh = decomp['hh']  # Diagonal edges

        # Step 2: Low-Frequency Processing
        ll_enhanced = self.low_freq_att(ll)

        # Step 3: High-Frequency Processing
        lh_enhanced = self.high_freq_att_lh(lh)
        hl_enhanced = self.high_freq_att_hl(hl)
        hh_enhanced = self.high_freq_att_hh(hh)

        # Combine high-frequency components
        high_freq_combined = lh_enhanced + hl_enhanced + hh_enhanced

        # Step 4: ODE Edge Reconstruction
        edges_reconstructed = self.ode_edge_recon(high_freq_combined)

        # Step 5: Feature Fusion
        # Concatenate all frequency components
        fused = torch.cat([ll_enhanced, lh_enhanced, hl_enhanced, edges_reconstructed], dim=1)
        output = self.fusion(fused)

        # Add residual connection if dimensions match
        if self.in_channels == self.out_channels:
            output = output + x

        # Final refinement
        output = self.final_refine(output)

        # Generate auxiliary outputs for deep supervision
        if return_aux:
            aux_outputs = {
                'low_freq_pred': self.aux_head_low(ll_enhanced),
                'high_freq_pred': self.aux_head_high(edges_reconstructed),
                'decomposition': decomp
            }
            return output, aux_outputs

        return output


class MultiScaleFrequencyExpert(nn.Module):
    """
    Multi-Scale Frequency Expert for PVT backbone integration.

    Processes features at multiple scales [64, 128, 320, 512] from PVT backbone
    with separate FrequencyExpert modules for each scale.

    Args:
        in_channels: List of input channels for each scale [64, 128, 320, 512]
        reduction: Attention reduction ratio (default: 16)
        ode_steps: ODE integration steps (default: 2)
    """
    def __init__(self,
                 in_channels: List[int] = [64, 128, 320, 512],
                 reduction: int = 16,
                 ode_steps: int = 2):
        super().__init__()

        self.in_channels = in_channels
        self.num_scales = len(in_channels)

        # Create FrequencyExpert for each scale
        self.experts = nn.ModuleList([
            FrequencyExpert(ch, ch, reduction, ode_steps)
            for ch in in_channels
        ])

        # Cross-scale feature fusion
        self.cross_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                LayerNorm2d(ch),
                nn.GELU()
            ) for ch in in_channels
        ])

        # Deep supervision prediction heads
        self.ds_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, padding=1),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 2, 1, 1),
                nn.Sigmoid()
            ) for ch in in_channels
        ])

    def forward(self,
                features: List[torch.Tensor],
                return_aux: bool = False):
        """
        Process multi-scale features from PVT backbone.

        Args:
            features: List of feature tensors [f1, f2, f3, f4]
                      with channels [64, 128, 320, 512]
            return_aux: If True, return auxiliary predictions

        Returns:
            If return_aux=False:
                List of enhanced features
            If return_aux=True:
                Tuple of (enhanced_features, aux_dict) where aux_dict contains:
                    - 'predictions': Deep supervision predictions for each scale
                    - 'aux_outputs': Auxiliary outputs from each expert
        """
        assert len(features) == self.num_scales, \
            f"Expected {self.num_scales} features, got {len(features)}"

        enhanced_features = []
        aux_predictions = []
        aux_outputs_all = []

        # Process each scale
        for i, (feat, expert) in enumerate(zip(features, self.experts)):
            # Apply frequency expert
            if return_aux:
                enhanced, aux = expert(feat, return_aux=True)
                aux_outputs_all.append(aux)
            else:
                enhanced = expert(feat)

            # Cross-scale fusion
            enhanced = self.cross_scale_fusion[i](enhanced)
            enhanced_features.append(enhanced)

            # Generate deep supervision prediction
            if return_aux:
                pred = self.ds_heads[i](enhanced)
                aux_predictions.append(pred)

        if return_aux:
            aux_dict = {
                'predictions': aux_predictions,
                'aux_outputs': aux_outputs_all
            }
            return enhanced_features, aux_dict

        return enhanced_features


# ============================================================================
# Testing and Validation
# ============================================================================

def test_frequency_expert():
    """Test FrequencyExpert with sample inputs."""
    print("=" * 80)
    print("Testing FrequencyExpert")
    print("=" * 80)

    # Test single-scale expert
    print("\n1. Testing single-scale FrequencyExpert...")
    expert = FrequencyExpert(in_channels=128, reduction=16, ode_steps=2)
    x = torch.randn(2, 128, 32, 32)

    # Forward without aux
    output = expert(x, return_aux=False)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"

    # Forward with aux
    output, aux = expert(x, return_aux=True)
    print(f"   Low-freq prediction: {aux['low_freq_pred'].shape}")
    print(f"   High-freq prediction: {aux['high_freq_pred'].shape}")
    print(f"   Decomposition keys: {list(aux['decomposition'].keys())}")

    # Test multi-scale expert
    print("\n2. Testing MultiScaleFrequencyExpert...")
    multi_expert = MultiScaleFrequencyExpert(
        in_channels=[64, 128, 320, 512],
        reduction=16,
        ode_steps=2
    )

    features = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 320, 16, 16),
        torch.randn(2, 512, 8, 8)
    ]

    # Forward with aux
    enhanced, aux_dict = multi_expert(features, return_aux=True)

    print(f"   Enhanced features:")
    for i, feat in enumerate(enhanced):
        print(f"      Scale {i} ({[64, 128, 320, 512][i]} ch): {feat.shape}")

    print(f"   Deep supervision predictions:")
    for i, pred in enumerate(aux_dict['predictions']):
        print(f"      Scale {i}: {pred.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in multi_expert.parameters())
    trainable_params = sum(p.numel() for p in multi_expert.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Memory (FP32): ~{total_params * 4 / 1024**2:.2f} MB")

    print("\n✓ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_frequency_expert()
