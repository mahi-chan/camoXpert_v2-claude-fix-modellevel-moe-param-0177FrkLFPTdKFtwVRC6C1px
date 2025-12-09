"""
Verification script for FEDER expert architecture.

This script validates:
1. Parameter count (~12-15M to match other experts)
2. Output shapes (main pred + 3 auxiliary outputs)
3. All components are present and functional
4. Forward pass works correctly
"""

import torch
import torch.nn as nn
from models.expert_architectures import FEDERFrequencyExpert


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_feder_architecture():
    print("="*70)
    print("FEDER Expert Architecture Verification")
    print("="*70)

    # Create FEDER expert
    print("\n1. Creating FEDER expert...")
    feature_dims = [64, 128, 320, 512]
    feder = FEDERFrequencyExpert(feature_dims=feature_dims)

    # Count parameters by component
    print("\n2. Parameter count by component:")
    print("-" * 70)

    total_params = count_parameters(feder)

    wavelet_params = count_parameters(feder.wavelet_decompositions)
    high_att_params = count_parameters(feder.high_freq_attentions)
    low_att_params = count_parameters(feder.low_freq_attentions)
    ode_params = count_parameters(feder.ode_edge_modules)
    fusion_params = count_parameters(feder.frequency_fusion)
    decoder_params = count_parameters(feder.decoder)

    print(f"  Wavelet Decompositions (4x):     {wavelet_params:>10,} params")
    print(f"  High-Freq Attentions (4x):       {high_att_params:>10,} params")
    print(f"  Low-Freq Attentions (4x):        {low_att_params:>10,} params")
    print(f"  ODE Edge Reconstruction (4x):    {ode_params:>10,} params")
    print(f"  Frequency Fusion (4x):           {fusion_params:>10,} params")
    print(f"  Decoder + Deep Supervision:      {decoder_params:>10,} params")
    print("-" * 70)
    print(f"  TOTAL FEDER PARAMETERS:          {total_params:>10,} params")
    print(f"                                   ({total_params/1e6:>10.2f}M)")
    print("-" * 70)

    # Check if in target range
    if 12e6 <= total_params <= 15e6:
        print("  ✓ Parameter count is in target range (12-15M)")
    else:
        print(f"  ⚠ Parameter count is outside target range (12-15M)")

    # Verify components
    print("\n3. Component verification:")
    print("-" * 70)

    components = {
        "DeepWaveletDecomposition": len(feder.wavelet_decompositions) == 4,
        "HighFrequencyAttention": len(feder.high_freq_attentions) == 4,
        "LowFrequencyAttention": len(feder.low_freq_attentions) == 4,
        "ODEEdgeReconstruction": len(feder.ode_edge_modules) == 4,
        "FrequencyFusion": len(feder.frequency_fusion) == 4,
        "DeepSupervisionDecoder": hasattr(feder, 'decoder'),
    }

    for comp_name, present in components.items():
        status = "✓" if present else "✗"
        print(f"  {status} {comp_name}")

    all_present = all(components.values())
    if all_present:
        print("\n  ✓ All required components are present")
    else:
        print("\n  ✗ Some components are missing!")

    # Test forward pass
    print("\n4. Testing forward pass...")
    print("-" * 70)

    # Create dummy input features (simulating PVT-v2 backbone output)
    batch_size = 2
    img_size = 416  # User's training size

    features = [
        torch.randn(batch_size, 64, img_size // 4, img_size // 4),   # f1: H/4
        torch.randn(batch_size, 128, img_size // 8, img_size // 8),  # f2: H/8
        torch.randn(batch_size, 320, img_size // 16, img_size // 16), # f3: H/16
        torch.randn(batch_size, 512, img_size // 32, img_size // 32)  # f4: H/32
    ]

    print(f"  Input feature shapes:")
    for i, f in enumerate(features):
        print(f"    f{i+1}: {list(f.shape)}")

    # Forward pass without aux
    print("\n  Testing forward pass (return_aux=False)...")
    with torch.no_grad():
        pred, aux = feder(features, return_aux=False)

    print(f"    Main prediction shape: {list(pred.shape)}")
    print(f"    Auxiliary outputs: {len(aux)}")

    expected_shape = [batch_size, 1, img_size, img_size]
    if list(pred.shape) == expected_shape:
        print(f"    ✓ Output shape matches expected {expected_shape}")
    else:
        print(f"    ✗ Output shape mismatch! Expected {expected_shape}, got {list(pred.shape)}")

    # Forward pass with aux
    print("\n  Testing forward pass (return_aux=True)...")
    with torch.no_grad():
        pred, aux_outputs = feder(features, return_aux=True)

    print(f"    Main prediction shape: {list(pred.shape)}")
    print(f"    Number of auxiliary outputs: {len(aux_outputs)}")

    if len(aux_outputs) == 3:
        print("    ✓ Correct number of auxiliary outputs (3)")
        for i, aux_pred in enumerate(aux_outputs):
            print(f"      aux{i+1}: {list(aux_pred.shape)}")
            if list(aux_pred.shape) == expected_shape:
                print(f"        ✓ Matches expected shape")
            else:
                print(f"        ✗ Shape mismatch!")
    else:
        print(f"    ✗ Wrong number of auxiliary outputs! Expected 3, got {len(aux_outputs)}")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    print("-" * 70)

    feder.train()
    pred, aux_outputs = feder(features, return_aux=True)

    # Dummy loss
    target = torch.ones_like(pred)
    loss = nn.functional.binary_cross_entropy_with_logits(pred, target)

    # Add auxiliary losses
    for aux_pred in aux_outputs:
        loss += 0.5 * nn.functional.binary_cross_entropy_with_logits(aux_pred, target)

    # Backward
    loss.backward()

    # Check if gradients exist
    has_grads = all(p.grad is not None for p in feder.parameters() if p.requires_grad)

    if has_grads:
        print("  ✓ Gradients computed successfully")

        # Check gradient magnitudes
        grad_norm = torch.sqrt(sum((p.grad**2).sum() for p in feder.parameters() if p.grad is not None))
        print(f"  ✓ Total gradient norm: {grad_norm.item():.6f}")
    else:
        print("  ✗ Some parameters did not receive gradients!")

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    checks = [
        ("Parameter count in range (12-15M)", 12e6 <= total_params <= 15e6),
        ("All components present", all_present),
        ("Forward pass successful", True),
        ("Output shape correct", list(pred.shape) == expected_shape),
        ("3 auxiliary outputs", len(aux_outputs) == 3),
        ("Gradient flow working", has_grads),
    ]

    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(passed for _, passed in checks)

    print("="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - FEDER is ready for training!")
    else:
        print("✗ SOME CHECKS FAILED - Please review the issues above")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    verify_feder_architecture()
