"""
Quick validation script to test all expert architectures for dimension mismatches.
Run this before training to catch any dimension issues early.
"""

import torch
import torch.nn as nn
from models.expert_architectures import (
    SINetExpert,
    PraNetExpert,
    ZoomNetExpert,
    FEDERFrequencyExpert
)


def test_expert(expert_class, expert_name):
    """Test a single expert architecture"""
    print(f"\n{'='*70}")
    print(f"Testing {expert_name}")
    print('='*70)

    try:
        # Create expert
        expert = expert_class(feature_dims=[64, 128, 320, 512])
        expert.eval()

        # Count parameters
        total_params = sum(p.numel() for p in expert.parameters())
        print(f"✓ Expert created successfully")
        print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

        # Create dummy features (simulating PVT-v2 backbone output)
        batch_size = 2
        img_size = 416

        features = [
            torch.randn(batch_size, 64, img_size // 4, img_size // 4),   # f1: H/4
            torch.randn(batch_size, 128, img_size // 8, img_size // 8),  # f2: H/8
            torch.randn(batch_size, 320, img_size // 16, img_size // 16), # f3: H/16
            torch.randn(batch_size, 512, img_size // 32, img_size // 32)  # f4: H/32
        ]

        print(f"✓ Input features created:")
        for i, f in enumerate(features):
            print(f"    f{i+1}: {list(f.shape)}")

        # Test forward pass without aux
        with torch.no_grad():
            pred, aux = expert(features, return_aux=False)

        print(f"✓ Forward pass (return_aux=False) successful")
        print(f"    Output: {list(pred.shape)}")
        print(f"    Aux outputs: {len(aux)}")

        # Test forward pass with aux
        with torch.no_grad():
            pred, aux_outputs = expert(features, return_aux=True)

        print(f"✓ Forward pass (return_aux=True) successful")
        print(f"    Main prediction: {list(pred.shape)}")
        print(f"    Aux outputs: {len(aux_outputs)}")
        for i, aux in enumerate(aux_outputs):
            print(f"      aux{i+1}: {list(aux.shape)}")

        # Verify output shape matches input resolution
        expected_shape = [batch_size, 1, img_size, img_size]
        if list(pred.shape) == expected_shape:
            print(f"✓ Output shape correct: {expected_shape}")
        else:
            print(f"✗ Output shape mismatch!")
            print(f"    Expected: {expected_shape}")
            print(f"    Got: {list(pred.shape)}")
            return False

        # Verify all aux outputs match main prediction size
        all_match = all(list(aux.shape) == expected_shape for aux in aux_outputs)
        if all_match:
            print(f"✓ All auxiliary outputs match expected shape")
        else:
            print(f"✗ Some auxiliary outputs have wrong shape!")
            return False

        print(f"\n{'='*70}")
        print(f"✓✓✓ {expert_name} PASSED ALL CHECKS ✓✓✓")
        print('='*70)

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗✗✗ {expert_name} FAILED ✗✗✗")
        print('='*70)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print('='*70)
        return False


def main():
    """Test all expert architectures"""
    print("\n" + "="*70)
    print("COMPREHENSIVE EXPERT ARCHITECTURE VALIDATION")
    print("="*70)

    experts_to_test = [
        (SINetExpert, "SINet (Search & Identification)"),
        (PraNetExpert, "PraNet (Parallel Reverse Attention)"),
        (ZoomNetExpert, "ZoomNet (Multi-scale Zoom)"),
        (FEDERFrequencyExpert, "FEDER (Frequency Expert)"),
    ]

    results = {}
    for expert_class, expert_name in experts_to_test:
        results[expert_name] = test_expert(expert_class, expert_name)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for expert_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {expert_name}")

    all_passed = all(results.values())

    print("="*70)
    if all_passed:
        print("✓✓✓ ALL EXPERTS PASSED - READY FOR TRAINING ✓✓✓")
    else:
        print("✗✗✗ SOME EXPERTS FAILED - FIX ISSUES BEFORE TRAINING ✗✗✗")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
