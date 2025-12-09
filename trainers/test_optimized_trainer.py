"""
Comprehensive test suite for OptimizedTrainer and its components.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.optimized_trainer import (
    CosineAnnealingWithWarmup,
    ExpertCollapseDetector,
    GlobalBatchLoadBalancer,
    CODProgressiveAugmentation,
    OptimizedTrainer
)


def test_cosine_annealing_warmup():
    """Test cosine annealing scheduler with warmup."""
    print("\n" + "="*60)
    print("Testing CosineAnnealingWithWarmup Scheduler")
    print("="*60)

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Create scheduler
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_epochs=5,
        total_epochs=50,
        min_lr=1e-6,
        max_lr=1e-4
    )

    # Track learning rates
    lr_history = []

    for epoch in range(50):
        lr = optimizer.param_groups[0]['lr']
        lr_history.append(lr)
        scheduler.step()

        if epoch < 5 or epoch in [10, 20, 30, 40, 49]:
            print(f"Epoch {epoch:3d}: LR = {lr:.6f}")

    # Verify warmup
    assert lr_history[0] == 1e-6, "Should start at min_lr"
    assert abs(lr_history[4] - 1e-4) < 1e-7, "Should reach max_lr after warmup"

    # Verify cosine decay
    assert lr_history[-1] < lr_history[5], "Should decay after warmup"
    assert lr_history[-1] >= 1e-6, "Should not go below min_lr"

    print(f"\n✓ Warmup phase: {lr_history[0]:.6f} -> {lr_history[4]:.6f}")
    print(f"✓ Decay phase: {lr_history[5]:.6f} -> {lr_history[-1]:.6f}")
    print("✓ CosineAnnealingWithWarmup test passed!")

    return lr_history


def test_expert_collapse_detector():
    """Test expert collapse detection."""
    print("\n" + "="*60)
    print("Testing ExpertCollapseDetector")
    print("="*60)

    num_experts = 6
    detector = ExpertCollapseDetector(
        num_experts=num_experts,
        collapse_threshold=0.05,
        window_size=20
    )

    # Scenario 1: Balanced routing (no collapse)
    print("\nScenario 1: Balanced Routing")
    for _ in range(25):
        # Uniform routing probabilities
        routing_probs = torch.softmax(torch.randn(32, num_experts), dim=1)
        # Simulate balanced expert assignments
        expert_assignments = torch.randint(0, num_experts, (32, 2))

        detector.update(routing_probs, expert_assignments)

    collapsed, reasons = detector.check_collapse()
    stats = detector.get_statistics()

    print(f"Collapsed: {collapsed}")
    print(f"Expert usage: {[f'{u:.3f}' for u in stats['expert_usage']]}")
    print(f"Min/Max usage: {stats['min_usage']:.4f} / {stats['max_usage']:.4f}")
    print(f"Load imbalance CV: {stats['load_imbalance_cv']:.4f}")
    print(f"Avg confidence: {stats['avg_confidence']:.4f}")

    assert not collapsed, "Balanced routing should not trigger collapse"
    print("✓ Balanced routing detected correctly")

    # Scenario 2: Expert collapse (some experts unused)
    print("\nScenario 2: Expert Collapse")
    detector2 = ExpertCollapseDetector(num_experts=num_experts, window_size=20)

    for _ in range(25):
        # Imbalanced routing - only use first 2 experts
        routing_probs = torch.zeros(32, num_experts)
        routing_probs[:, 0] = 0.6
        routing_probs[:, 1] = 0.4

        # Only assign to first 2 experts
        expert_assignments = torch.randint(0, 2, (32, 2))

        detector2.update(routing_probs, expert_assignments)

    collapsed, reasons = detector2.check_collapse()
    stats = detector2.get_statistics()

    print(f"Collapsed: {collapsed}")
    print(f"Reasons: {reasons}")
    print(f"Expert usage: {[f'{u:.3f}' for u in stats['expert_usage']]}")

    assert collapsed, "Imbalanced routing should trigger collapse"
    assert stats['min_usage'] < 0.05, "Some experts should be underutilized"
    print("✓ Expert collapse detected correctly")

    print("✓ ExpertCollapseDetector test passed!")


def test_global_batch_load_balancer():
    """Test global-batch load balancing."""
    print("\n" + "="*60)
    print("Testing GlobalBatchLoadBalancer")
    print("="*60)

    num_experts = 6
    balancer = GlobalBatchLoadBalancer(
        num_experts=num_experts,
        alpha_l2=0.01,
        alpha_cv=0.01
    )

    # Simulate multiple batches
    print("\nSimulating balanced routing...")
    for batch in range(10):
        # Balanced routing
        routing_probs = torch.softmax(torch.randn(32, num_experts), dim=1)
        expert_assignments = torch.randint(0, num_experts, (32, 2))

        balancer.update(routing_probs, expert_assignments)

        # Compute load balance loss
        lb_loss = balancer.compute_load_balance_loss(routing_probs, expert_assignments)

        if batch % 3 == 0:
            print(f"Batch {batch}: Load balance loss = {lb_loss.item():.6f}")

    # Get global statistics
    stats = balancer.get_global_statistics()

    print(f"\nGlobal Statistics:")
    print(f"Expert usage: {[f'{u:.3f}' for u in stats['global_expert_usage']]}")
    print(f"Min/Max: {stats['min_global_usage']:.4f} / {stats['max_global_usage']:.4f}")
    print(f"Usage std: {stats['global_usage_std']:.4f}")
    print(f"Total tokens: {stats['total_tokens_processed']}")

    # Verify balanced usage
    usage = np.array(stats['global_expert_usage'])
    assert usage.min() > 0.10, "All experts should be used"
    assert usage.max() < 0.25, "No expert should dominate"

    print("✓ GlobalBatchLoadBalancer test passed!")


def test_progressive_augmentation():
    """Test progressive augmentation components."""
    print("\n" + "="*60)
    print("Testing CODProgressiveAugmentation")
    print("="*60)

    augmentation = CODProgressiveAugmentation(
        initial_strength=0.3,
        max_strength=0.8,
        transition_epoch=20,
        transition_duration=10
    )

    # Test strength progression
    print("\nAugmentation strength over epochs:")
    strengths = []
    for epoch in range(40):
        augmentation.update_epoch(epoch)
        strengths.append(augmentation.current_strength)

        if epoch in [0, 10, 19, 20, 25, 30, 35, 39]:
            print(f"Epoch {epoch:3d}: Strength = {augmentation.current_strength:.3f}")

    assert strengths[0] == 0.3, "Should start at initial strength"
    assert strengths[19] == 0.3, "Should maintain initial until transition"
    assert 0.3 < strengths[25] < 0.8, "Should be ramping up during transition"
    assert strengths[35] == 0.8, "Should reach max after transition"

    print("✓ Strength progression working correctly")

    # Test Fourier-based mixing
    print("\nTesting Fourier-based mixing...")
    B, C, H, W = 4, 3, 128, 128
    images1 = torch.rand(B, C, H, W)
    images2 = torch.rand(B, C, H, W)
    masks1 = torch.rand(B, 1, H, W)
    masks2 = torch.rand(B, 1, H, W)

    mixed_images, mixed_masks = augmentation.fourier_based_mixing(
        images1, images2, masks1, masks2, alpha=0.5
    )

    assert mixed_images.shape == images1.shape, "Shape should be preserved"
    assert mixed_masks.shape == masks1.shape, "Mask shape should be preserved"
    assert not torch.allclose(mixed_images, images1), "Images should be mixed"

    print(f"✓ Fourier mixing: {images1.shape} -> {mixed_images.shape}")

    # Test contrastive augmentation
    print("\nTesting contrastive augmentation...")
    images = torch.rand(B, C, H, W)
    masks = torch.rand(B, 1, H, W)

    anchor, positive, anchor_masks, pos_masks = augmentation.contrastive_augmentation(
        images, masks
    )

    assert anchor.shape == positive.shape, "Anchor and positive should have same shape"
    assert not torch.allclose(anchor, positive), "Positive should be augmented"

    print(f"✓ Contrastive aug: Generated anchor and positive pairs")

    # Test mirror disruption
    print("\nTesting mirror disruption...")
    augmentation.current_strength = 1.0  # Force high probability

    aug_images, aug_masks = augmentation.mirror_disruption(images, masks)

    assert aug_images.shape == images.shape, "Shape should be preserved"

    print(f"✓ Mirror disruption: {images.shape} -> {aug_images.shape}")

    print("✓ CODProgressiveAugmentation test passed!")


def test_optimized_trainer_integration():
    """Test full OptimizedTrainer integration."""
    print("\n" + "="*60)
    print("Testing OptimizedTrainer Integration")
    print("="*60)

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 1, 3, padding=1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.conv2(x)
            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        accumulation_steps=2,
        use_amp=True,
        total_epochs=30,
        warmup_epochs=5,
        min_lr=1e-6,
        max_lr=1e-4,
        enable_progressive_aug=True,
        aug_transition_epoch=20
    )

    print(f"\n✓ Trainer created successfully")
    print(f"  Device: {device}")
    print(f"  Accumulation steps: {trainer.accumulation_steps}")
    print(f"  AMP enabled: {trainer.use_amp}")
    print(f"  Progressive aug: {trainer.enable_progressive_aug}")

    # Create dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=50):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            image = torch.rand(3, 64, 64)
            mask = torch.randint(0, 2, (1, 64, 64)).float()
            return image, mask

    dataset = DummyDataset(size=20)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True
    )

    # Test training for a few epochs
    print("\nTraining for 3 epochs...")
    for epoch in range(3):
        metrics = trainer.train_epoch(train_loader, epoch, log_interval=5)

        print(f"\nEpoch {epoch} Summary:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")

    print("\n✓ Training loop completed successfully")

    # Test checkpoint saving/loading
    print("\nTesting checkpoint save/load...")
    checkpoint_path = "/tmp/test_checkpoint.pth"

    trainer.save_checkpoint(checkpoint_path, epoch=2, metrics=metrics)
    print(f"✓ Checkpoint saved")

    # Create new trainer and load
    trainer2 = OptimizedTrainer(
        model=SimpleModel().to(device),
        optimizer=optim.Adam(model.parameters(), lr=1e-4),
        criterion=criterion,
        device=device
    )

    loaded_epoch = trainer2.load_checkpoint(checkpoint_path)
    assert loaded_epoch == 2, "Should load correct epoch"
    print(f"✓ Checkpoint loaded (epoch {loaded_epoch})")

    # Get training summary
    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        elif isinstance(value, bool):
            print(f"  {key}: {value}")

    print("\n✓ OptimizedTrainer integration test passed!")


def test_moe_integration():
    """Test MoE-specific features."""
    print("\n" + "="*60)
    print("Testing MoE Integration Features")
    print("="*60)

    # Create model with MoE-like outputs
    class MoEModel(nn.Module):
        def __init__(self, num_experts=6):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, padding=1)
            self.num_experts = num_experts

        def forward(self, x):
            pred = self.conv(x)

            # Simulate routing information
            B = x.size(0)
            routing_probs = torch.softmax(torch.randn(B, self.num_experts, device=x.device), dim=1)
            expert_assignments = torch.randint(0, self.num_experts, (B, 2), device=x.device)

            return {
                'predictions': pred,
                'routing_info': {
                    'routing_probs': routing_probs,
                    'expert_assignments': expert_assignments
                }
            }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoEModel(num_experts=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Create trainer with MoE features enabled
    trainer = OptimizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_experts=6,
        enable_load_balancing=True,
        enable_collapse_detection=True
    )

    print(f"\n✓ MoE trainer created")
    print(f"  Load balancing: {trainer.enable_load_balancing}")
    print(f"  Collapse detection: {trainer.enable_collapse_detection}")

    # Create dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, idx):
            return torch.rand(3, 64, 64), torch.rand(1, 64, 64)

    dataset = DummyDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    # Train for 2 epochs
    print("\nTraining MoE model...")
    for epoch in range(2):
        metrics = trainer.train_epoch(train_loader, epoch, log_interval=10)

        print(f"\nEpoch {epoch} MoE Metrics:")
        for key, value in metrics.items():
            if 'collapse' in key or 'global' in key:
                print(f"  {key}: {value}")

    summary = trainer.get_training_summary()

    assert 'expert_collapse_detected' in summary, "Should track collapse status"
    assert 'global_load_stats' in summary, "Should track global load stats"

    print("\n✓ MoE integration test passed!")


def visualize_scheduler():
    """Create visualization of learning rate schedule."""
    print("\n" + "="*60)
    print("Visualizing Learning Rate Schedule")
    print("="*60)

    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_epochs=5,
        total_epochs=100,
        min_lr=1e-6,
        max_lr=1e-4
    )

    lr_history = []
    for epoch in range(100):
        lr = optimizer.param_groups[0]['lr']
        lr_history.append(lr)
        scheduler.step()

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(lr_history, linewidth=2)
    plt.axvline(x=5, color='r', linestyle='--', label='Warmup End')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Cosine Annealing with 5-Epoch Warmup (1e-6 to 1e-4)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = '/home/user/camoXpert_v2/lr_schedule_visualization.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Visualization saved to {output_path}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print(" " * 20 + "OPTIMIZED TRAINER TEST SUITE")
    print("="*80)

    try:
        # Test individual components
        lr_history = test_cosine_annealing_warmup()
        test_expert_collapse_detector()
        test_global_batch_load_balancer()
        test_progressive_augmentation()

        # Test integration
        test_optimized_trainer_integration()
        test_moe_integration()

        # Visualization
        visualize_scheduler()

        print("\n" + "="*80)
        print(" " * 25 + "ALL TESTS PASSED! ✓")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_all_tests()
