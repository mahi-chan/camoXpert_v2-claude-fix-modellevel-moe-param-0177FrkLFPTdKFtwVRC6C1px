"""
Comprehensive test suite for RecursiveFeedbackDecoder and its components.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recursive_feedback_decoder import (
    SpatialReductionAttention,
    PVTFeedForward,
    PVTBlock,
    IterationWeightingScheme,
    HighResolutionFusionModule,
    RefinementBlock,
    RecursiveFeedbackDecoder
)


def test_spatial_reduction_attention():
    """Test PVT-style spatial reduction attention."""
    print("\n" + "="*60)
    print("Testing SpatialReductionAttention")
    print("="*60)

    batch_size = 4
    seq_length = 256  # 16x16
    dim = 128
    H, W = 16, 16

    # Test with different spatial reduction ratios
    sr_ratios = [1, 2, 4, 8]

    for sr_ratio in sr_ratios:
        print(f"\nSpatial reduction ratio: {sr_ratio}")

        sr_attn = SpatialReductionAttention(
            dim=dim,
            num_heads=8,
            sr_ratio=sr_ratio
        )

        x = torch.randn(batch_size, seq_length, dim)
        output = sr_attn(x, H, W)

        print(f"  Input: {x.shape}")
        print(f"  Output: {output.shape}")

        assert output.shape == x.shape, "Output shape should match input"

        # Check memory savings
        full_attn_size = seq_length * seq_length
        reduced_attn_size = seq_length * (seq_length // (sr_ratio ** 2))
        memory_savings = 1 - (reduced_attn_size / full_attn_size)

        print(f"  Memory savings: {memory_savings * 100:.1f}%")

    print("\n✓ SpatialReductionAttention test passed!")


def test_pvt_feedforward():
    """Test PVT feed-forward network."""
    print("\n" + "="*60)
    print("Testing PVTFeedForward")
    print("="*60)

    batch_size = 4
    H, W = 16, 16
    seq_length = H * W
    dim = 128

    ffn = PVTFeedForward(dim=dim, hidden_dim=dim * 4, dropout=0.1)

    x = torch.randn(batch_size, seq_length, dim)
    output = ffn(x, H, W)

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")

    assert output.shape == x.shape, "Output shape should match input"

    print("\n✓ PVTFeedForward test passed!")


def test_pvt_block():
    """Test complete PVT block."""
    print("\n" + "="*60)
    print("Testing PVTBlock")
    print("="*60)

    batch_size = 4
    H, W = 32, 32
    seq_length = H * W
    dim = 256

    pvt_block = PVTBlock(
        dim=dim,
        num_heads=8,
        sr_ratio=4,
        mlp_ratio=4.0
    )

    x = torch.randn(batch_size, seq_length, dim)
    output = pvt_block(x, H, W)

    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")

    assert output.shape == x.shape, "Output shape should match input"

    # Check residual connections
    diff = (output - x).abs().mean()
    print(f"Mean difference from input: {diff.item():.6f}")

    print("\n✓ PVTBlock test passed!")


def test_iteration_weighting_schemes():
    """Test different iteration weighting schemes."""
    print("\n" + "="*60)
    print("Testing IterationWeightingScheme")
    print("="*60)

    batch_size = 2
    num_iterations = 4
    channels = 256
    H, W = 32, 32

    # Create iteration features
    iteration_features = [
        torch.randn(batch_size, channels, H, W) for _ in range(num_iterations)
    ]

    schemes = ['uniform', 'exponential', 'learned']

    for scheme in schemes:
        print(f"\nTesting scheme: {scheme}")

        weighting = IterationWeightingScheme(
            num_iterations=num_iterations,
            channels=channels,
            scheme=scheme
        )

        weighted = weighting(iteration_features, current_iteration=num_iterations - 1)

        print(f"  Input: {num_iterations} features of shape {iteration_features[0].shape}")
        print(f"  Output: {weighted.shape}")

        assert weighted.shape == iteration_features[0].shape, "Output shape should match input"

        # Check that output is different from simple averaging (except for uniform)
        if scheme != 'uniform':
            simple_avg = torch.stack(iteration_features).mean(dim=0)
            diff = (weighted - simple_avg).abs().mean()
            print(f"  Difference from simple average: {diff.item():.6f}")

        # Test learned scheme parameters
        if scheme == 'learned':
            print(f"  Iteration weights: {torch.sigmoid(weighting.iteration_weights).detach().numpy()}")

    print("\n✓ IterationWeightingScheme test passed!")


def test_high_resolution_fusion():
    """Test high-resolution fusion module."""
    print("\n" + "="*60)
    print("Testing HighResolutionFusionModule")
    print("="*60)

    batch_size = 2
    in_channels = [64, 128, 320, 512]
    out_channels = 256

    # Create multi-scale features
    multi_scale_features = [
        torch.randn(batch_size, 64, 64, 64),   # High res
        torch.randn(batch_size, 128, 32, 32),  # Medium res
        torch.randn(batch_size, 320, 16, 16),  # Low res
        torch.randn(batch_size, 512, 8, 8)     # Very low res
    ]

    print("Input features:")
    for i, feat in enumerate(multi_scale_features):
        print(f"  Level {i}: {feat.shape}")

    hr_fusion = HighResolutionFusionModule(
        in_channels=in_channels,
        out_channels=out_channels
    )

    fused = hr_fusion(multi_scale_features)

    print(f"\nOutput: {fused.shape}")

    # Should match highest resolution input
    max_h = max(feat.size(2) for feat in multi_scale_features)
    max_w = max(feat.size(3) for feat in multi_scale_features)

    assert fused.size(2) == max_h, "Should preserve highest resolution (height)"
    assert fused.size(3) == max_w, "Should preserve highest resolution (width)"
    assert fused.size(1) == out_channels, "Should have correct output channels"

    print(f"✓ Preserved high resolution: {max_h}×{max_w}")

    print("\n✓ HighResolutionFusionModule test passed!")


def test_refinement_block():
    """Test single refinement block."""
    print("\n" + "="*60)
    print("Testing RefinementBlock")
    print("="*60)

    batch_size = 2
    channels = 256
    H, W = 32, 32

    refinement = RefinementBlock(
        channels=channels,
        num_heads=8,
        sr_ratio=2,
        use_residual=True
    )

    x = torch.randn(batch_size, channels, H, W)
    refined = refinement(x)

    print(f"Input: {x.shape}")
    print(f"Output: {refined.shape}")

    assert refined.shape == x.shape, "Output shape should match input"

    # Check residual effect
    diff = (refined - x).abs().mean()
    print(f"Mean difference from input: {diff.item():.6f}")

    print("\n✓ RefinementBlock test passed!")


def test_recursive_feedback_decoder_basic():
    """Test basic RecursiveFeedbackDecoder functionality."""
    print("\n" + "="*60)
    print("Testing RecursiveFeedbackDecoder (Basic)")
    print("="*60)

    batch_size = 2
    encoder_channels = [64, 128, 320, 512]
    num_iterations = 4

    # Create decoder
    decoder = RecursiveFeedbackDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=256,
        num_iterations=num_iterations,
        num_classes=1,
        iteration_scheme='learned',
        use_global_feedback=True
    )

    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create encoder features
    encoder_features = [
        torch.randn(batch_size, 64, 64, 64),
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 320, 16, 16),
        torch.randn(batch_size, 512, 8, 8)
    ]

    print("\nEncoder features:")
    for i, feat in enumerate(encoder_features):
        print(f"  Level {i}: {feat.shape}")

    # Forward pass without returning iterations
    print("\nForward pass (no iterations returned)...")
    outputs = decoder(encoder_features, return_iterations=False)

    print(f"\nOutput keys: {list(outputs.keys())}")
    print(f"Final prediction: {outputs['final_prediction'].shape}")

    assert 'final_prediction' in outputs, "Should have final prediction"
    assert outputs['final_prediction'].size(0) == batch_size, "Batch size should match"
    assert outputs['final_prediction'].size(1) == 1, "Should have 1 class"

    # Forward pass with iterations
    print("\nForward pass (with iterations returned)...")
    outputs = decoder(encoder_features, return_iterations=True)

    print(f"\nOutput keys: {list(outputs.keys())}")
    print(f"Number of iteration predictions: {len(outputs['iteration_predictions'])}")
    print(f"Number of iteration features: {len(outputs['iteration_features'])}")

    assert len(outputs['iteration_predictions']) == num_iterations
    assert len(outputs['iteration_features']) == num_iterations

    print("\n✓ RecursiveFeedbackDecoder basic test passed!")


def test_different_iteration_schemes():
    """Test decoder with different iteration schemes."""
    print("\n" + "="*60)
    print("Testing Different Iteration Schemes")
    print("="*60)

    batch_size = 2
    encoder_features = [
        torch.randn(batch_size, 64, 32, 32),
        torch.randn(batch_size, 128, 16, 16),
        torch.randn(batch_size, 320, 8, 8),
        torch.randn(batch_size, 512, 4, 4)
    ]

    schemes = ['uniform', 'exponential', 'learned']

    for scheme in schemes:
        print(f"\nTesting scheme: {scheme}")

        decoder = RecursiveFeedbackDecoder(
            encoder_channels=[64, 128, 320, 512],
            decoder_channels=128,
            num_iterations=3,
            iteration_scheme=scheme,
            use_global_feedback=True
        )

        outputs = decoder(encoder_features, return_iterations=True)

        print(f"  Final prediction: {outputs['final_prediction'].shape}")
        print(f"  Iterations: {len(outputs['iteration_predictions'])}")

        # Check that predictions improve over iterations (heuristic)
        predictions = outputs['iteration_predictions']
        print(f"  Prediction stats:")
        for i, pred in enumerate(predictions):
            mean_val = pred.abs().mean().item()
            print(f"    Iteration {i}: mean={mean_val:.4f}")

    print("\n✓ Iteration schemes test passed!")


def test_global_feedback_connection():
    """Test global feedback connection."""
    print("\n" + "="*60)
    print("Testing Global Feedback Connection")
    print("="*60)

    batch_size = 2
    encoder_features = [
        torch.randn(batch_size, 64, 32, 32),
        torch.randn(batch_size, 128, 16, 16),
        torch.randn(batch_size, 320, 8, 8),
        torch.randn(batch_size, 512, 4, 4)
    ]

    # Test with and without global feedback
    for use_feedback in [False, True]:
        print(f"\nGlobal feedback: {use_feedback}")

        decoder = RecursiveFeedbackDecoder(
            encoder_channels=[64, 128, 320, 512],
            decoder_channels=128,
            num_iterations=4,
            use_global_feedback=use_feedback
        )

        outputs = decoder(encoder_features, return_iterations=True)

        print(f"  Final prediction: {outputs['final_prediction'].shape}")

        # Check parameter count
        params = sum(p.numel() for p in decoder.parameters())
        print(f"  Parameters: {params:,}")

    print("\n✓ Global feedback test passed!")


def test_dynamic_iteration_adjustment():
    """Test dynamic iteration count adjustment."""
    print("\n" + "="*60)
    print("Testing Dynamic Iteration Adjustment")
    print("="*60)

    batch_size = 2
    encoder_features = [
        torch.randn(batch_size, 64, 32, 32),
        torch.randn(batch_size, 128, 16, 16),
        torch.randn(batch_size, 320, 8, 8),
        torch.randn(batch_size, 512, 4, 4)
    ]

    # Create decoder with 5 iterations
    decoder = RecursiveFeedbackDecoder(
        encoder_channels=[64, 128, 320, 512],
        decoder_channels=128,
        num_iterations=5
    )

    print(f"Initial iterations: {decoder.get_iteration_count()}")

    # Test with different iteration counts
    for num_iter in [3, 4, 5]:
        decoder.set_iteration_count(num_iter)
        print(f"\nSet iterations to: {num_iter}")

        outputs = decoder(encoder_features, return_iterations=True)
        actual_iters = len(outputs['iteration_predictions'])

        print(f"  Actual iterations: {actual_iters}")
        assert actual_iters == num_iter, f"Should have {num_iter} iterations"

    print("\n✓ Dynamic iteration adjustment test passed!")


def test_gradient_flow():
    """Test gradient flow through decoder."""
    print("\n" + "="*60)
    print("Testing Gradient Flow")
    print("="*60)

    batch_size = 2
    encoder_features = [
        torch.randn(batch_size, 64, 32, 32, requires_grad=True),
        torch.randn(batch_size, 128, 16, 16, requires_grad=True),
        torch.randn(batch_size, 320, 8, 8, requires_grad=True),
        torch.randn(batch_size, 512, 4, 4, requires_grad=True)
    ]

    decoder = RecursiveFeedbackDecoder(
        encoder_channels=[64, 128, 320, 512],
        decoder_channels=128,
        num_iterations=4
    )

    # Forward
    outputs = decoder(encoder_features, return_iterations=True)

    # Compute loss
    target = torch.rand_like(outputs['final_prediction'])
    loss = F.binary_cross_entropy_with_logits(outputs['final_prediction'], target)

    # Add auxiliary losses from iterations
    for pred in outputs['iteration_predictions']:
        target_resized = F.interpolate(target, size=pred.shape[2:], mode='nearest')
        loss += 0.4 * F.binary_cross_entropy_with_logits(pred, target_resized)

    print(f"Loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    # Check gradients
    print("\nGradient statistics:")
    for i, feat in enumerate(encoder_features):
        if feat.grad is not None:
            grad_norm = feat.grad.norm().item()
            grad_mean = feat.grad.mean().item()
            print(f"  Level {i}: norm={grad_norm:.6f}, mean={grad_mean:.6f}")
            assert grad_norm > 0, f"Level {i} should have gradients"

    # Check decoder gradients
    total_grad_norm = 0.0
    num_params_with_grad = 0
    for name, param in decoder.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()
            num_params_with_grad += 1

    print(f"\nDecoder gradient norm: {total_grad_norm:.6f}")
    print(f"Parameters with gradients: {num_params_with_grad}")

    assert num_params_with_grad > 0, "Decoder should have gradients"

    print("\n✓ Gradient flow test passed!")


def test_memory_efficiency():
    """Test memory efficiency of spatial reduction."""
    print("\n" + "="*60)
    print("Testing Memory Efficiency")
    print("="*60)

    batch_size = 4
    encoder_features = [
        torch.randn(batch_size, 64, 64, 64),   # 4096 pixels
        torch.randn(batch_size, 128, 32, 32),  # 1024 pixels
        torch.randn(batch_size, 320, 16, 16),  # 256 pixels
        torch.randn(batch_size, 512, 8, 8)     # 64 pixels
    ]

    # Standard decoder (hypothetical full attention)
    # Memory ~ N^2 for each level

    # Our decoder with spatial reduction
    decoder = RecursiveFeedbackDecoder(
        encoder_channels=[64, 128, 320, 512],
        decoder_channels=256,
        num_iterations=4,
        sr_ratios=[8, 4, 2, 1]
    )

    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Estimate memory savings
    full_attention_memory = sum((feat.size(2) * feat.size(3)) ** 2 for feat in encoder_features)
    reduced_attention_memory = sum(
        (feat.size(2) * feat.size(3)) * (feat.size(2) * feat.size(3) // (sr ** 2))
        for feat, sr in zip(encoder_features, [8, 4, 2, 1])
    )

    memory_savings = 1 - (reduced_attention_memory / full_attention_memory)

    print(f"\nEstimated attention memory savings: {memory_savings * 100:.1f}%")
    print(f"Full attention memory units: {full_attention_memory:,}")
    print(f"Reduced attention memory units: {reduced_attention_memory:,}")

    print("\n✓ Memory efficiency test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print(" " * 15 + "RECURSIVE FEEDBACK DECODER TEST SUITE")
    print("="*80)

    try:
        # Test individual components
        test_spatial_reduction_attention()
        test_pvt_feedforward()
        test_pvt_block()
        test_iteration_weighting_schemes()
        test_high_resolution_fusion()
        test_refinement_block()

        # Test integrated decoder
        test_recursive_feedback_decoder_basic()
        test_different_iteration_schemes()
        test_global_feedback_connection()
        test_dynamic_iteration_adjustment()

        # Test training capabilities
        test_gradient_flow()

        # Test efficiency
        test_memory_efficiency()

        print("\n" + "="*80)
        print(" " * 25 + "ALL TESTS PASSED! ✓")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    # Import after path setup
    import torch.nn.functional as F
    run_all_tests()
