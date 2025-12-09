"""
Usage Examples for FrequencyExpert

Demonstrates how to use FrequencyExpert for camouflaged object detection
with PVT backbone integration and deep supervision.
"""

import torch
from models.frequency_expert import (
    FrequencyExpert,
    MultiScaleFrequencyExpert,
    DeepWaveletDecomposition,
    HighFrequencyAttention,
    LowFrequencyAttention,
    ODEEdgeReconstruction
)


def example_1_single_scale():
    """
    Example 1: Using FrequencyExpert for single-scale features.

    This is useful when you want to process features at a specific resolution
    with frequency decomposition and specialized attention.
    """
    print("="*80)
    print("Example 1: Single-Scale FrequencyExpert")
    print("="*80)

    # Create a FrequencyExpert for 256-channel features
    expert = FrequencyExpert(
        in_channels=256,
        out_channels=256,
        reduction=16,      # Attention reduction ratio
        ode_steps=2        # ODE integration steps
    )

    # Sample input features from a backbone [B, C, H, W]
    features = torch.randn(4, 256, 32, 32)

    # Forward pass WITHOUT deep supervision
    enhanced_features = expert(features, return_aux=False)
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {enhanced_features.shape}")

    # Forward pass WITH deep supervision
    enhanced_features, aux_outputs = expert(features, return_aux=True)
    print(f"\nWith deep supervision:")
    print(f"  Low-freq prediction: {aux_outputs['low_freq_pred'].shape}")
    print(f"  High-freq edge map: {aux_outputs['high_freq_pred'].shape}")
    print(f"  Wavelet components: {list(aux_outputs['decomposition'].keys())}")

    print("\n✓ Example 1 complete\n")


def example_2_multi_scale_pvt():
    """
    Example 2: Using MultiScaleFrequencyExpert with PVT backbone.

    This is the recommended approach for integrating FrequencyExpert
    with PVT-v2 backbones that produce multi-scale features.
    """
    print("="*80)
    print("Example 2: Multi-Scale FrequencyExpert (PVT Integration)")
    print("="*80)

    # Create MultiScaleFrequencyExpert for PVT-v2-b2 backbone
    # PVT-v2-b2 produces features at [64, 128, 320, 512] channels
    multi_expert = MultiScaleFrequencyExpert(
        in_channels=[64, 128, 320, 512],
        reduction=16,
        ode_steps=2
    )

    # Sample features from PVT backbone at different scales
    backbone_features = [
        torch.randn(4, 64, 88, 88),    # Stage 1: H/4 × W/4
        torch.randn(4, 128, 44, 44),   # Stage 2: H/8 × W/8
        torch.randn(4, 320, 22, 22),   # Stage 3: H/16 × W/16
        torch.randn(4, 512, 11, 11)    # Stage 4: H/32 × W/32
    ]

    # Process all scales
    enhanced_features, aux_dict = multi_expert(
        backbone_features,
        return_aux=True
    )

    print("Enhanced features:")
    for i, feat in enumerate(enhanced_features):
        print(f"  Scale {i}: {feat.shape}")

    print("\nDeep supervision predictions:")
    for i, pred in enumerate(aux_dict['predictions']):
        print(f"  Scale {i}: {pred.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in multi_expert.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Memory (FP32): ~{total_params * 4 / 1024**2:.2f} MB")

    print("\n✓ Example 2 complete\n")


def example_3_custom_integration():
    """
    Example 3: Custom integration with your own architecture.

    Shows how to use individual components (wavelet decomposition,
    attention modules, ODE reconstruction) in a custom architecture.
    """
    print("="*80)
    print("Example 3: Custom Component Usage")
    print("="*80)

    channels = 128
    x = torch.randn(4, channels, 32, 32)

    # 1. Use Wavelet Decomposition only
    print("\n1. Wavelet Decomposition:")
    wavelet = DeepWaveletDecomposition(channels, learnable=True)
    decomp = wavelet(x)
    print(f"  LL (low-freq): {decomp['ll'].shape}")
    print(f"  LH (horizontal): {decomp['lh'].shape}")
    print(f"  HL (vertical): {decomp['hl'].shape}")
    print(f"  HH (diagonal): {decomp['hh'].shape}")

    # 2. Use High-Frequency Attention only
    print("\n2. High-Frequency Attention:")
    hf_att = HighFrequencyAttention(channels, reduction=16)
    edges = decomp['lh'] + decomp['hl'] + decomp['hh']
    enhanced_edges = hf_att(edges)
    print(f"  Input edges: {edges.shape}")
    print(f"  Enhanced: {enhanced_edges.shape}")

    # 3. Use Low-Frequency Attention only
    print("\n3. Low-Frequency Attention:")
    lf_att = LowFrequencyAttention(channels, reduction=4)
    enhanced_content = lf_att(decomp['ll'])
    print(f"  Input content: {decomp['ll'].shape}")
    print(f"  Enhanced: {enhanced_content.shape}")

    # 4. Use ODE Edge Reconstruction only
    print("\n4. ODE Edge Reconstruction:")
    ode_recon = ODEEdgeReconstruction(channels, num_steps=3)
    reconstructed_edges = ode_recon(enhanced_edges)
    print(f"  Input edges: {enhanced_edges.shape}")
    print(f"  Reconstructed: {reconstructed_edges.shape}")

    print("\n✓ Example 3 complete\n")


def example_4_training_setup():
    """
    Example 4: Training setup with deep supervision.

    Shows how to use auxiliary outputs for deep supervision during training.
    """
    print("="*80)
    print("Example 4: Training with Deep Supervision")
    print("="*80)

    # Model
    expert = FrequencyExpert(in_channels=256, reduction=16, ode_steps=2)

    # Sample batch
    images = torch.randn(4, 256, 32, 32)
    targets = torch.randint(0, 2, (4, 1, 32, 32)).float()

    # Forward pass with deep supervision
    enhanced_features, aux_outputs = expert(images, return_aux=True)

    # Compute losses
    print("\nComputing losses for deep supervision:")

    # You would use these predictions with your loss function
    low_freq_pred = aux_outputs['low_freq_pred']    # [B, 1, H, W]
    high_freq_pred = aux_outputs['high_freq_pred']  # [B, 1, H, W]

    print(f"  Low-freq prediction: {low_freq_pred.shape}")
    print(f"  High-freq prediction: {high_freq_pred.shape}")
    print(f"  Target shape: {targets.shape}")

    # Example loss computation (pseudo-code):
    # loss_main = criterion(final_prediction, targets)
    # loss_low = criterion(low_freq_pred, targets) * 0.3
    # loss_high = criterion(high_freq_pred, edge_targets) * 0.3
    # total_loss = loss_main + loss_low + loss_high

    print("\n  Training loss = main_loss + 0.3*low_freq_loss + 0.3*high_freq_loss")

    print("\n✓ Example 4 complete\n")


def example_5_full_model_integration():
    """
    Example 5: Full model integration with decoder.

    Shows how to integrate FrequencyExpert into a complete COD model.
    """
    print("="*80)
    print("Example 5: Full COD Model Integration")
    print("="*80)

    import torch.nn as nn

    class SimpleCODModel(nn.Module):
        """Example COD model with FrequencyExpert."""
        def __init__(self):
            super().__init__()

            # Frequency expert for each scale
            self.freq_expert = MultiScaleFrequencyExpert(
                in_channels=[64, 128, 320, 512],
                reduction=16,
                ode_steps=2
            )

            # Simple decoder (example)
            self.decoder = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ) for ch in [64, 128, 320, 512]
            ])

            # Final prediction head
            self.final_head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            )

        def forward(self, backbone_features, return_aux=False):
            # Enhance features with frequency expert
            if return_aux:
                enhanced_features, aux_dict = self.freq_expert(
                    backbone_features, return_aux=True
                )
            else:
                enhanced_features = self.freq_expert(
                    backbone_features, return_aux=False
                )

            # Decode features
            decoded = []
            for feat, decoder in zip(enhanced_features, self.decoder):
                decoded.append(decoder(feat))

            # Upsample and fuse all scales
            target_size = decoded[0].shape[-2:]
            fused = sum(
                torch.nn.functional.interpolate(
                    d, size=target_size, mode='bilinear', align_corners=False
                ) for d in decoded
            )

            # Final prediction
            prediction = self.final_head(fused)

            if return_aux:
                return prediction, aux_dict
            return prediction

    # Create model and test
    model = SimpleCODModel()

    # Sample backbone features
    backbone_features = [
        torch.randn(2, 64, 88, 88),
        torch.randn(2, 128, 44, 44),
        torch.randn(2, 320, 22, 22),
        torch.randn(2, 512, 11, 11)
    ]

    # Inference
    prediction, aux_dict = model(backbone_features, return_aux=True)

    print(f"Final prediction: {prediction.shape}")
    print(f"Auxiliary predictions: {len(aux_dict['predictions'])} scales")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")

    print("\n✓ Example 5 complete\n")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FrequencyExpert Usage Examples")
    print("="*80 + "\n")

    # Run all examples
    example_1_single_scale()
    example_2_multi_scale_pvt()
    example_3_custom_integration()
    example_4_training_setup()
    example_5_full_model_integration()

    print("="*80)
    print("All examples completed successfully!")
    print("="*80)
