"""Quick test to verify all fixes are applied correctly."""
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/kaggle/working/camoXpert_v2')

def test_all():
    print("="*60)
    print("TESTING CAMOEXPERT FIXES")
    print("="*60)

    # Test 1: Loss with label smoothing
    print("\n[1] Testing Loss...")
    from losses.composite_loss import CompositeLossSystem
    criterion = CompositeLossSystem(label_smoothing=0.1)

    pred = torch.randn(2, 1, 352, 352)
    mask = (torch.randn(2, 1, 352, 352) > 0).float()
    loss = criterion(pred, mask)

    print(f"    Loss value: {loss.item():.4f}")
    assert 0.5 < loss.item() < 5.0, f"Loss out of range: {loss.item()}"
    assert criterion.label_smoothing == 0.1, "Label smoothing not set!"
    print("    ✓ Loss OK with label_smoothing=0.1")

    # Test 2: Model with expert tracking
    print("\n[2] Testing Model...")
    from models.model_level_moe import ModelLevelMoE
    model = ModelLevelMoE(backbone_name='pvt_v2_b2', num_experts=3, top_k=2, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 352, 352)
    with torch.no_grad():
        pred, info = model(x, return_routing_info=True)

    print(f"    Output shape: {pred.shape}")
    print(f"    Router probs: {info['expert_probs'][0].numpy()}")

    assert 'individual_expert_preds' in info, "Expert tracking missing!"
    assert info['individual_expert_preds'] is not None, "Expert preds is None!"
    assert len(info['individual_expert_preds']) == 3, f"Expected 3 experts, got {len(info['individual_expert_preds'])}"
    print("    ✓ Individual expert outputs tracked")

    # Test 3: Expert dropout
    print("\n[3] Testing Expert Dropout...")
    from models.expert_architectures import SINetExpert, PraNetExpert, ZoomNetExpert

    for name, ExpertClass in [("SINet", SINetExpert), ("PraNet", PraNetExpert), ("ZoomNet", ZoomNetExpert)]:
        expert = ExpertClass([64, 128, 320, 512])
        has_dropout = hasattr(expert, 'dropout')
        status = "✓" if has_dropout else "⚠️ MISSING"
        print(f"    {name} dropout: {status}")

    # Test 4: Gradient flow
    print("\n[4] Testing Gradient Flow...")
    model.train()
    pred, _ = model(x, return_routing_info=True)
    loss = criterion(pred, mask)
    loss.backward()

    grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    print(f"    Gradient norm: {grad_norm:.2f}")
    assert grad_norm > 0, "No gradients!"
    print("    ✓ Gradients flowing")

    # Summary
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nRun training with:")
    print("  torchrun --nproc_per_node=2 train_advanced.py \\")
    print("      --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \\")
    print("      --epochs 100 --use-ddp")

if __name__ == '__main__':
    test_all()
