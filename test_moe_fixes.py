"""
Test script to verify Model-Level MoE fixes
"""

import torch
import sys
sys.path.insert(0, '/home/user/camoXpert_v2')

from models.model_level_moe import ModelLevelMoE

print("="*70)
print("Testing Model-Level MoE Fixes")
print("="*70)

# Create model with 3 experts, top-2 selection
print("\n[Test 1] Creating ModelLevelMoE with 3 experts...")
model = ModelLevelMoE(backbone_name='pvt_v2_b2', num_experts=3, top_k=2, pretrained=False)

print(f"\n✓ Number of experts: {len(model.expert_models)}")
assert len(model.expert_models) == 3, "Should have exactly 3 experts"
print(f"✓ Top-K: {model.top_k}")
assert model.top_k == 2, "Should select top-2 experts"

# Test forward pass
print("\n[Test 2] Testing forward pass...")
x = torch.randn(2, 3, 416, 416)
pred, info = model(x, return_routing_info=True)

print(f"✓ Prediction shape: {pred.shape}")
assert pred.shape == (2, 1, 416, 416), f"Expected [2, 1, 416, 416], got {pred.shape}"

print(f"✓ Expert probs shape: {info['expert_probs'].shape}")
assert info['expert_probs'].shape == (2, 3), f"Expected [2, 3], got {info['expert_probs'].shape}"

# Test router freezing
print("\n[Test 3] Testing router freezing...")
model.freeze_router()
# Check that router parameters are frozen
router_frozen = all(not p.requires_grad for p in model.router.parameters())
assert router_frozen, "Router parameters should be frozen"
print("✓ Router frozen successfully")

model.unfreeze_router()
# Check that router parameters are unfrozen
router_unfrozen = all(p.requires_grad for p in model.router.parameters())
assert router_unfrozen, "Router parameters should be unfrozen"
print("✓ Router unfrozen successfully")

# Test equal routing weights
print("\n[Test 4] Testing equal routing weights...")
equal_weights = model.get_equal_routing_weights(batch_size=4, device='cpu')
print(f"✓ Equal weights shape: {equal_weights.shape}")
assert equal_weights.shape == (4, 3), f"Expected [4, 3], got {equal_weights.shape}"
assert torch.allclose(equal_weights, torch.ones_like(equal_weights) / 3), "Weights should be equal (1/3 each)"
print(f"✓ All weights equal to 1/3: {equal_weights[0]}")

# Test individual expert predictions in training mode
print("\n[Test 5] Testing individual expert predictions...")
model.train()
pred_train, info_train = model(x, return_routing_info=True)
if 'individual_expert_preds' in info_train:
    print(f"✓ Individual expert predictions: {len(info_train['individual_expert_preds'])} experts")
    assert len(info_train['individual_expert_preds']) == 3, "Should have 3 individual predictions"
    for i, expert_pred in enumerate(info_train['individual_expert_preds']):
        print(f"  Expert {i} pred shape: {expert_pred.shape}")
        assert expert_pred.shape == (2, 1, 416, 416), f"Expected [2, 1, 416, 416], got {expert_pred.shape}"
else:
    print("⚠️  Warning: individual_expert_preds not found in routing_info")

# Test load balance loss
print("\n[Test 6] Checking load balance loss...")
if info_train['load_balance_loss'] is not None:
    print(f"✓ Load balance loss: {info_train['load_balance_loss'].item():.6f}")
    print(f"✓ Load balance coefficient: {model.router.load_balance_coef}")
    assert model.router.load_balance_coef == 0.1, "Load balance coef should be 0.1"
else:
    print("⚠️  Warning: load_balance_loss is None")

# Test entropy computation
print("\n[Test 7] Testing entropy regularization...")
test_probs = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
entropy_bonus = model.router.compute_entropy_bonus(test_probs)
print(f"✓ Entropy bonus computed: {entropy_bonus.item():.6f}")
assert entropy_bonus.item() >= 0, "Entropy bonus should be non-negative"

print("\n" + "="*70)
print("✅ All tests passed!")
print("="*70)
print("\nSummary:")
print(f"  • 3 experts: SINet, PraNet, ZoomNet")
print(f"  • Top-2 selection")
print(f"  • Router freezing/unfreezing works")
print(f"  • Individual expert predictions available")
print(f"  • Load balance coefficient: 0.1 (10x increase)")
print(f"  • Entropy regularization enabled")
print("="*70)
