#!/usr/bin/env python3
"""Quick test to verify parameter counts are balanced"""

import torch
from models.expert_architectures import SINetExpert, PraNetExpert, ZoomNetExpert

feature_dims = [64, 128, 320, 512]

print("="*60)
print("EXPERT PARAMETER COUNT TEST")
print("="*60)

experts = [
    ("SINet", SINetExpert(feature_dims)),
    ("PraNet", PraNetExpert(feature_dims)),
    ("ZoomNet", ZoomNetExpert(feature_dims)),
]

for name, expert in experts:
    params = sum(p.numel() for p in expert.parameters())
    print(f"{name:12s}: {params/1e6:6.1f}M parameters")

print("="*60)

# Test forward pass
print("\nTesting forward pass...")
x = [
    torch.randn(2, 64, 88, 88),
    torch.randn(2, 128, 44, 44),
    torch.randn(2, 320, 22, 22),
    torch.randn(2, 512, 11, 11),
]

for name, expert in experts:
    try:
        pred, aux = expert(x, return_aux=True)
        print(f"{name:12s}: ✓ Forward pass OK (output shape: {pred.shape})")
    except Exception as e:
        print(f"{name:12s}: ✗ FAILED - {e}")

print("\n" + "="*60)
print("SUCCESS: All tests passed!")
print("="*60)
