"""
Demo script for FrequencyExpert usage in CamoXpert

This script demonstrates:
1. Single-scale FrequencyExpert usage
2. Multi-scale FrequencyExpert usage
3. Integration with existing models
4. Deep supervision training example
5. Visualization of frequency decomposition
"""

import torch
import torch.nn as nn
import sys
sys.path.append('..')

from models.frequency_expert import (
    FrequencyExpert,
    MultiScaleFrequencyExpert,
    LearnableWaveletDecomposition
)


def demo_single_scale():
    """Demonstrate single-scale FrequencyExpert"""
    print("="*70)
    print("Demo 1: Single-Scale FrequencyExpert")
    print("="*70)

    # Create expert
    dim = 128
    expert = FrequencyExpert(dim=dim)

    # Create sample input
    batch_size = 2
    height, width = 32, 32
    x = torch.randn(batch_size, dim, height, width)

    print(f"\nInput shape: {x.shape}")

    # Basic forward pass
    output = expert(x)
    print(f"Output shape: {output.shape}")
    print(f"✓ Shape preserved: {output.shape == x.shape}")

    # Forward pass with auxiliary outputs
    output, aux = expert(x, return_aux=True)
    print(f"\nAuxiliary outputs:")
    print(f"  - Low-frequency prediction: {aux['low_freq_pred'].shape}")
    print(f"  - High-frequency prediction: {aux['high_freq_pred'].shape}")
    print(f"  - Decomposition components: {list(aux['decomposition'].keys())}")

    # Check decomposition shapes
    for key, value in aux['decomposition'].items():
        print(f"    - {key.upper()}: {value.shape}")

    # Count parameters
    params = sum(p.numel() for p in expert.parameters())
    print(f"\nTotal parameters: {params:,}")

    return expert, output, aux


def demo_multi_scale():
    """Demonstrate multi-scale FrequencyExpert"""
    print("\n" + "="*70)
    print("Demo 2: Multi-Scale FrequencyExpert")
    print("="*70)

    # Feature dimensions (typical for PVT or similar backbones)
    dims = [64, 128, 320, 512]
    expert = MultiScaleFrequencyExpert(dims=dims)

    # Create multi-scale features (simulating backbone output)
    batch_size = 2
    features = [
        torch.randn(batch_size, 64, 64, 64),   # 1/4 resolution
        torch.randn(batch_size, 128, 32, 32),  # 1/8 resolution
        torch.randn(batch_size, 320, 16, 16),  # 1/16 resolution
        torch.randn(batch_size, 512, 8, 8)     # 1/32 resolution
    ]

    print(f"\nInput features:")
    for i, feat in enumerate(features):
        print(f"  Scale {i}: {feat.shape}")

    # Forward pass with auxiliary outputs
    enhanced_features, aux = expert(features, return_aux=True)

    print(f"\nEnhanced features:")
    for i, feat in enumerate(enhanced_features):
        print(f"  Scale {i}: {feat.shape}")

    print(f"\nDeep supervision predictions:")
    for i, pred in enumerate(aux['predictions']):
        print(f"  Scale {i}: {pred.shape}")

    # Count parameters
    params = sum(p.numel() for p in expert.parameters())
    print(f"\nTotal parameters: {params:,}")

    return expert, enhanced_features, aux


def demo_training_with_deep_supervision():
    """Demonstrate training with deep supervision"""
    print("\n" + "="*70)
    print("Demo 3: Training with Deep Supervision")
    print("="*70)

    # Setup
    dim = 128
    expert = FrequencyExpert(dim=dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(expert.parameters(), lr=1e-4)

    # Simulate training data
    batch_size = 4
    x = torch.randn(batch_size, dim, 32, 32)
    target = torch.randint(0, 2, (batch_size, 1, 32, 32)).float()
    edge_target = torch.randint(0, 2, (batch_size, 1, 32, 32)).float()

    print(f"\nInput: {x.shape}")
    print(f"Target: {target.shape}")
    print(f"Edge target: {edge_target.shape}")

    # Forward pass
    expert.train()
    output, aux = expert(x, return_aux=True)

    # Compute losses
    # Note: In practice, you'd need a proper prediction head
    # This is simplified for demonstration
    loss_main = torch.mean((output - x) ** 2)  # Simplified
    loss_low_freq = criterion(aux['low_freq_pred'], target)
    loss_high_freq = criterion(aux['high_freq_pred'], edge_target)

    # Weighted combination
    alpha_low = 0.3
    alpha_high = 0.3
    loss_total = loss_main + alpha_low * loss_low_freq + alpha_high * loss_high_freq

    print(f"\nLosses:")
    print(f"  Main loss: {loss_main.item():.4f}")
    print(f"  Low-freq loss: {loss_low_freq.item():.4f}")
    print(f"  High-freq loss: {loss_high_freq.item():.4f}")
    print(f"  Total loss: {loss_total.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    print(f"\n✓ Backward pass successful")

    return expert, loss_total


def demo_wavelet_decomposition():
    """Demonstrate wavelet decomposition visualization"""
    print("\n" + "="*70)
    print("Demo 4: Wavelet Decomposition Analysis")
    print("="*70)

    # Create decomposition module
    channels = 64
    decomp = LearnableWaveletDecomposition(channels)

    # Create sample input
    batch_size = 1
    x = torch.randn(batch_size, channels, 64, 64)

    print(f"\nInput: {x.shape}")

    # Perform decomposition
    components = decomp(x)

    print(f"\nWavelet decomposition components:")
    for key, value in components.items():
        print(f"  {key.upper()}: {value.shape}")
        print(f"    - Mean: {value.mean().item():.4f}")
        print(f"    - Std: {value.std().item():.4f}")
        print(f"    - Min: {value.min().item():.4f}")
        print(f"    - Max: {value.max().item():.4f}")

    # Visualize energy distribution
    print(f"\nEnergy distribution:")
    total_energy = sum(torch.sum(c ** 2) for c in components.values())
    for key, value in components.items():
        energy = torch.sum(value ** 2)
        percentage = (energy / total_energy * 100).item()
        print(f"  {key.upper()}: {percentage:.2f}%")

    return components


def demo_integration_example():
    """Show how to integrate into existing model"""
    print("\n" + "="*70)
    print("Demo 5: Integration Example")
    print("="*70)

    class SimpleModelWithFrequency(nn.Module):
        """Example model integrating FrequencyExpert"""
        def __init__(self):
            super().__init__()

            # Backbone (simplified - normally you'd use EdgeNeXt/PVT)
            self.backbone = nn.ModuleList([
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.Conv2d(128, 320, 3, stride=2, padding=1),
                nn.Conv2d(320, 512, 3, stride=2, padding=1)
            ])

            # Multi-scale frequency expert
            self.freq_expert = MultiScaleFrequencyExpert(
                dims=[64, 128, 320, 512]
            )

            # Decoder (simplified)
            self.decoder = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=16, mode='bilinear'),
                nn.Conv2d(256, 1, 1)
            )

        def forward(self, x):
            # Extract features
            features = []
            feat = x
            for layer in self.backbone:
                feat = layer(feat)
                features.append(feat)

            # Apply frequency expert
            enhanced_features, aux = self.freq_expert(
                features,
                return_aux=True
            )

            # Decode
            output = self.decoder(enhanced_features[-1])

            return output, aux['predictions']

    # Create model
    model = SimpleModelWithFrequency()

    # Test forward pass
    batch_size = 2
    img = torch.randn(batch_size, 3, 256, 256)

    print(f"\nInput image: {img.shape}")

    output, aux_preds = model(img)

    print(f"Output: {output.shape}")
    print(f"Auxiliary predictions: {len(aux_preds)} scales")
    for i, pred in enumerate(aux_preds):
        print(f"  Scale {i}: {pred.shape}")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {params:,}")

    return model


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" FrequencyExpert Demo Suite")
    print("="*70)
    print("\nThis demo showcases the FEDER-inspired FrequencyExpert")
    print("for camouflaged object detection.\n")

    # Run demos
    try:
        demo_single_scale()
        demo_multi_scale()
        demo_training_with_deep_supervision()
        demo_wavelet_decomposition()
        demo_integration_example()

        print("\n" + "="*70)
        print(" ✓ All demos completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run demos
    main()

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Review FREQUENCY_EXPERT_GUIDE.md for detailed documentation")
    print("2. Integrate FrequencyExpert into your model (see Demo 5)")
    print("3. Train with deep supervision (see Demo 3)")
    print("4. Visualize frequency decomposition during inference")
    print("5. Tune hyperparameters (ODE dt, damping, attention reduction)")
    print("="*70)
