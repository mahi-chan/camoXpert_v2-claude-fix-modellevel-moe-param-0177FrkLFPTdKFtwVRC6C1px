"""
Test script to verify TDD, GAD, BPN integration in Model-Level MoE
"""

import torch
import sys
sys.path.insert(0, '/home/user/camoXpert_v2')

from models.model_level_moe import ModelLevelMoE

print("="*70)
print("Testing Discontinuity Detection Integration")
print("="*70)

# Create model with discontinuity detection
print("\n[Test 1] Creating ModelLevelMoE with discontinuity modules...")
model = ModelLevelMoE(
    backbone_name='pvt_v2_b2',
    num_experts=3,
    top_k=2,
    pretrained=False
)

# Verify modules exist
print("\n[Test 2] Verifying discontinuity modules...")
assert hasattr(model, 'tdd'), "TDD module not found"
assert hasattr(model, 'gad'), "GAD module not found"
assert hasattr(model, 'bpn'), "BPN module not found"
print("✓ TDD module exists")
print("✓ GAD module exists")
print("✓ BPN module exists")

# Test forward pass
print("\n[Test 3] Testing forward pass with discontinuity detection...")
x = torch.randn(2, 3, 416, 416)
model.eval()

with torch.no_grad():
    pred, info = model(x, return_routing_info=True)

print(f"✓ Prediction shape: {pred.shape}")
assert pred.shape == (2, 1, 416, 416), f"Expected [2, 1, 416, 416], got {pred.shape}"

# Verify discontinuity outputs
print("\n[Test 4] Verifying discontinuity outputs in routing_info...")
assert 'texture_disc' in info, "texture_disc not in routing_info"
assert 'gradient_anomaly' in info, "gradient_anomaly not in routing_info"
assert 'boundary' in info, "boundary not in routing_info"
assert 'boundary_scales' in info, "boundary_scales not in routing_info"
assert 'texture_descriptors' in info, "texture_descriptors not in routing_info"
assert 'gradient_features' in info, "gradient_features not in routing_info"

print(f"✓ Texture discontinuity shape: {info['texture_disc'].shape}")
print(f"✓ Gradient anomaly shape: {info['gradient_anomaly'].shape}")
print(f"✓ Boundary shape: {info['boundary'].shape}")
print(f"✓ Number of boundary scales: {len(info['boundary_scales'])}")
print(f"✓ Texture descriptors shape: {info['texture_descriptors'].shape}")
print(f"✓ Gradient features shape: {info['gradient_features'].shape}")

# Verify value ranges
print("\n[Test 5] Verifying output value ranges...")
assert info['texture_disc'].min() >= 0 and info['texture_disc'].max() <= 1, "texture_disc out of [0,1] range"
assert info['gradient_anomaly'].min() >= 0 and info['gradient_anomaly'].max() <= 1, "gradient_anomaly out of [0,1] range"
assert info['boundary'].min() >= 0 and info['boundary'].max() <= 1, "boundary out of [0,1] range"
print(f"✓ Texture discontinuity range: [{info['texture_disc'].min():.3f}, {info['texture_disc'].max():.3f}]")
print(f"✓ Gradient anomaly range: [{info['gradient_anomaly'].min():.3f}, {info['gradient_anomaly'].max():.3f}]")
print(f"✓ Boundary range: [{info['boundary'].min():.3f}, {info['boundary'].max():.3f}]")

# Test training mode (should have individual_expert_preds)
print("\n[Test 6] Testing training mode...")
model.train()
pred_train, info_train = model(x, return_routing_info=True)
assert 'individual_expert_preds' in info_train, "individual_expert_preds not in training mode"
assert info_train['individual_expert_preds'] is not None, "individual_expert_preds is None in training mode"
assert len(info_train['individual_expert_preds']) == 3, f"Expected 3 expert predictions, got {len(info_train['individual_expert_preds'])}"
print(f"✓ Individual expert predictions: {len(info_train['individual_expert_preds'])} experts")

# Test gradient flow through discontinuity modules
print("\n[Test 7] Testing gradient flow...")
model.train()
pred, info = model(x, return_routing_info=True)
loss = pred.mean() + info['texture_disc'].mean() + info['gradient_anomaly'].mean() + info['boundary'].mean()
loss.backward()

# Check gradients exist
tdd_has_grad = any(p.grad is not None for p in model.tdd.parameters())
gad_has_grad = any(p.grad is not None for p in model.gad.parameters())
bpn_has_grad = any(p.grad is not None for p in model.bpn.parameters())

assert tdd_has_grad, "TDD has no gradients"
assert gad_has_grad, "GAD has no gradients"
assert bpn_has_grad, "BPN has no gradients"
print("✓ TDD gradient flow OK")
print("✓ GAD gradient flow OK")
print("✓ BPN gradient flow OK")

print("\n" + "="*70)
print("✅ All discontinuity integration tests passed!")
print("="*70)
print("\nSummary:")
print(f"  • TDD, GAD, BPN modules integrated successfully")
print(f"  • All outputs available in routing_info")
print(f"  • Gradient flow verified through all modules")
print(f"  • Training mode provides individual expert predictions")
print("="*70)
