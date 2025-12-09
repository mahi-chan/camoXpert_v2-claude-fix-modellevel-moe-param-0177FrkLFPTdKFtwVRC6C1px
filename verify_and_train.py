"""
Verification and Training Launch Script

Run this to:
1. Verify all modules work correctly
2. Test forward/backward passes
3. Start training if all checks pass
"""

import os
import sys
import torch
import torch.nn as nn

def verify_modules():
    """Verify all modules are working"""
    print("="*70)
    print("VERIFICATION: Testing all modules")
    print("="*70)

    errors = []

    # Test 1: TDD
    print("\n[1/5] Testing TextureDiscontinuityDetector...")
    try:
        from models.texture_discontinuity import TextureDiscontinuityDetector
        tdd = TextureDiscontinuityDetector(64)
        x = torch.randn(2, 64, 104, 104)
        disc, desc = tdd(x)
        assert disc.shape == (2, 1, 104, 104), f"Wrong shape: {disc.shape}"
        disc.mean().backward()
        print("  ✓ TDD OK")
        print(f"    - Output shape: {disc.shape}")
        print(f"    - Descriptor shape: {desc.shape}")
        print(f"    - Parameters: {sum(p.numel() for p in tdd.parameters())/1e6:.2f}M")
    except Exception as e:
        errors.append(f"TDD: {e}")
        print(f"  ✗ TDD FAILED: {e}")

    # Test 2: GAD
    print("\n[2/5] Testing GradientAnomalyDetector...")
    try:
        from models.gradient_anomaly import GradientAnomalyDetector
        gad = GradientAnomalyDetector(64)
        x = torch.randn(2, 64, 104, 104)
        anomaly, feat = gad(x)
        assert anomaly.shape == (2, 1, 104, 104), f"Wrong shape: {anomaly.shape}"
        anomaly.mean().backward()
        print("  ✓ GAD OK")
        print(f"    - Output shape: {anomaly.shape}")
        print(f"    - Feature shape: {feat.shape}")
        print(f"    - Parameters: {sum(p.numel() for p in gad.parameters())/1e6:.2f}M")
    except Exception as e:
        errors.append(f"GAD: {e}")
        print(f"  ✗ GAD FAILED: {e}")

    # Test 3: BPN
    print("\n[3/5] Testing BoundaryPriorNetwork...")
    try:
        from models.boundary_prior import BoundaryPriorNetwork
        bpn = BoundaryPriorNetwork([64, 128, 320, 512])
        features = [
            torch.randn(2, 64, 104, 104),
            torch.randn(2, 128, 52, 52),
            torch.randn(2, 320, 26, 26),
            torch.randn(2, 512, 13, 13),
        ]
        td = torch.randn(2, 1, 104, 104)
        ga = torch.randn(2, 1, 104, 104)
        boundary, scales = bpn(features, td, ga)
        assert boundary.shape == (2, 1, 104, 104), f"Wrong shape: {boundary.shape}"
        assert len(scales) == 4, f"Wrong number of scales: {len(scales)}"
        boundary.mean().backward()
        print("  ✓ BPN OK")
        print(f"    - Output shape: {boundary.shape}")
        print(f"    - Number of scales: {len(scales)}")
        print(f"    - Parameters: {sum(p.numel() for p in bpn.parameters())/1e6:.2f}M")
    except Exception as e:
        errors.append(f"BPN: {e}")
        print(f"  ✗ BPN FAILED: {e}")

    # Test 4: Full Model
    print("\n[4/5] Testing Full Enhanced MoE...")
    try:
        from models.model_level_moe import ModelLevelMoE
        model = ModelLevelMoE(
            backbone_name='pvt_v2_b2',
            num_experts=3,
            top_k=2,
            pretrained=False
        )
        model.train()
        x = torch.randn(2, 3, 416, 416)
        pred, info = model(x, return_routing_info=True)

        assert pred.shape == (2, 1, 416, 416), f"Wrong pred shape: {pred.shape}"
        assert 'texture_disc' in info, "Missing texture_disc"
        assert 'gradient_anomaly' in info, "Missing gradient_anomaly"
        assert 'boundary' in info, "Missing boundary"
        assert 'individual_expert_preds' in info, "Missing individual_expert_preds"
        assert info['individual_expert_preds'] is not None, "individual_expert_preds is None"
        assert len(info['individual_expert_preds']) == 3, f"Wrong number of expert preds: {len(info['individual_expert_preds'])}"

        pred.mean().backward()
        print("  ✓ Full Model OK")
        print(f"    - Prediction shape: {pred.shape}")
        print(f"    - Texture disc shape: {info['texture_disc'].shape}")
        print(f"    - Gradient anomaly shape: {info['gradient_anomaly'].shape}")
        print(f"    - Boundary shape: {info['boundary'].shape}")
        print(f"    - Expert predictions: {len(info['individual_expert_preds'])}")
        print(f"    - Expert probs: {info['expert_probs'][0].detach().numpy()}")

        # Test router freezing
        model.freeze_router()
        frozen_count = sum(1 for p in model.router.parameters() if not p.requires_grad)
        print(f"    - Router freeze: {frozen_count} params frozen")

        model.unfreeze_router()
        unfrozen_count = sum(1 for p in model.router.parameters() if p.requires_grad)
        print(f"    - Router unfreeze: {unfrozen_count} params active")

    except Exception as e:
        errors.append(f"Full Model: {e}")
        print(f"  ✗ Full Model FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Loss Function
    print("\n[5/5] Testing CombinedEnhancedLoss...")
    try:
        from losses.boundary_aware_loss import CombinedEnhancedLoss
        criterion = CombinedEnhancedLoss()

        pred = torch.sigmoid(torch.randn(2, 1, 416, 416))
        target = (torch.rand(2, 1, 416, 416) > 0.7).float()
        aux = {
            'boundary': torch.sigmoid(torch.randn(2, 1, 104, 104)),
            'texture_disc': torch.sigmoid(torch.randn(2, 1, 104, 104)),
            'gradient_anomaly': torch.sigmoid(torch.randn(2, 1, 104, 104)),
            'individual_expert_preds': [torch.sigmoid(torch.randn(2, 1, 416, 416)) for _ in range(3)],
            'load_balance_loss': torch.tensor(0.1),
        }

        loss, loss_dict = criterion(pred, target, aux)
        loss.backward()
        print("  ✓ Loss OK")
        print(f"    - Total loss: {loss.item():.4f}")
        print(f"    - Components: {list(loss_dict.keys())}")

        # Print component values
        for key in ['seg_bce', 'seg_dice', 'boundary', 'tdd', 'gad', 'expert', 'hard_mining', 'load_balance']:
            if key in loss_dict:
                val = loss_dict[key]
                val_float = val.item() if isinstance(val, torch.Tensor) else val
                print(f"      - {key}: {val_float:.4f}")

    except Exception as e:
        errors.append(f"Loss: {e}")
        print(f"  ✗ Loss FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    if errors:
        print(f"❌ VERIFICATION FAILED: {len(errors)} errors")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✅ VERIFICATION PASSED: All modules working!")
        return True


def print_training_info():
    """Print training information"""
    print("\n" + "="*70)
    print("TRAINING INFORMATION")
    print("="*70)

    print("\n📋 Model Architecture:")
    print("  - Backbone: PVT-v2-B2 (pretrained)")
    print("  - Experts: 3 (SINet, PraNet, ZoomNet)")
    print("  - Router: Sophisticated router with load balancing")
    print("  - Discontinuity Detection:")
    print("    * TDD: Texture Discontinuity Detector (~0.2M params)")
    print("    * GAD: Gradient Anomaly Detector (~0.15M params)")
    print("    * BPN: Boundary Prior Network (~0.3M params)")
    print("  - Total: ~85.65M params, ~48.65M active per forward")

    print("\n📋 Training Strategy:")
    print("  - Phase 1 (Epochs 1-20): Router FROZEN, experts learn")
    print("  - Phase 2 (Epochs 21-150): Router UNFROZEN, joint learning")
    print("  - EMA: Enabled (decay=0.999)")
    print("  - Mixup: Enabled (alpha=0.2)")
    print("  - Batch size: 8 per GPU")
    print("  - Learning rate: 1e-4 with warmup + cosine annealing")

    print("\n📋 Loss Components:")
    print("  - Segmentation: 1.0 (boundary-aware BCE + Dice)")
    print("  - Boundary Prediction: 2.0 (BPN supervision)")
    print("  - Discontinuity: 0.3 (TDD + GAD supervision)")
    print("  - Per-Expert: 0.3 (individual expert supervision)")
    print("  - Hard Mining: 0.5 (focus on difficult pixels)")
    print("  - Load Balance: 0.1 (router regularization)")

    print("\n📋 Expected Metrics:")
    print("  - Target IoU: 0.78+ (Training Val)")
    print("  - Target S-measure: 0.93+")
    print("  - Target F-measure: 0.88+")


def print_training_commands():
    """Print training commands"""
    print("\n" + "="*70)
    print("TRAINING COMMANDS")
    print("="*70)

    print("\n🚀 Single GPU Training:")
    print("-"*70)
    print("python train.py \\")
    print("    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \\")
    print("    --epochs 150 \\")
    print("    --batch-size 8 \\")
    print("    --img-size 416 \\")
    print("    --lr 1e-4 \\")
    print("    --enable-router-warmup \\")
    print("    --router-warmup-epochs 20 \\")
    print("    --val-freq 5 \\")
    print("    --save-interval 10")
    print("-"*70)

    print("\n🚀 Multi-GPU Training (DDP):")
    print("-"*70)
    print("torchrun --nproc_per_node=2 train.py \\")
    print("    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \\")
    print("    --epochs 150 \\")
    print("    --batch-size 8 \\")
    print("    --img-size 416 \\")
    print("    --lr 1e-4 \\")
    print("    --use-ddp \\")
    print("    --enable-router-warmup \\")
    print("    --router-warmup-epochs 20 \\")
    print("    --val-freq 5 \\")
    print("    --save-interval 10")
    print("-"*70)

    print("\n🔧 Custom Loss Weights:")
    print("-"*70)
    print("python train.py \\")
    print("    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \\")
    print("    --epochs 150 \\")
    print("    --batch-size 8 \\")
    print("    --seg-weight 1.0 \\")
    print("    --boundary-weight 2.0 \\")
    print("    --discontinuity-weight 0.3 \\")
    print("    --expert-weight 0.3 \\")
    print("    --hard-mining-weight 0.5 \\")
    print("    --load-balance-weight 0.1 \\")
    print("    --enable-router-warmup \\")
    print("    --router-warmup-epochs 20")
    print("-"*70)


def main():
    print("\n" + "="*70)
    print("ENHANCED MOE VERIFICATION & TRAINING")
    print("="*70)

    # Run verification
    if not verify_modules():
        print("\n⚠️  Fix errors before training!")
        sys.exit(1)

    # Print training info
    print_training_info()

    # Print training commands
    print_training_commands()

    print("\n" + "="*70)
    print("✅ All checks passed! Ready to train.")
    print("="*70)
    print("\n💡 Tip: Copy one of the commands above to start training!")
    print()


if __name__ == '__main__':
    main()
