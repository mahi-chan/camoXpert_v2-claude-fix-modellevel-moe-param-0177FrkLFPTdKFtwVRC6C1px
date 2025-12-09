"""
Comprehensive test suite for HighOrderAttention and its components.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.high_order_attention import (
    TuckerDecomposition,
    PolynomialAttention,
    ChannelInteractionEnhancementModule,
    MultiGranularityFusion,
    CrossKnowledgePropagation,
    HighOrderAttention
)


def test_tucker_decomposition():
    """Test Tucker decomposition module."""
    print("\n" + "="*60)
    print("Testing TuckerDecomposition")
    print("="*60)

    batch_size = 4
    channels = 128
    height, width = 32, 32

    # Create module
    tucker = TuckerDecomposition(
        in_channels=channels,
        ranks=[32, 32, 32],
        spatial_size=8
    )

    print(f"Input: [{batch_size}, {channels}, {height}, {width}]")
    print(f"Decomposition ranks: {tucker.ranks}")
    print(f"Parameters:")
    print(f"  U1 (channel): {tucker.U1.shape}")
    print(f"  U2 (height): {tucker.U2.shape}")
    print(f"  U3 (width): {tucker.U3.shape}")
    print(f"  Core tensor: {tucker.core.shape}")

    # Forward pass
    x = torch.randn(batch_size, channels, height, width)
    output = tucker(x)

    print(f"\nOutput shape: {output.shape}")
    assert output.shape == x.shape, "Output shape should match input"

    # Check that output is different from input (decomposition applied)
    assert not torch.allclose(output, x), "Output should be different from input"

    # Check value range
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("✓ TuckerDecomposition test passed!")


def test_polynomial_attention():
    """Test polynomial attention module."""
    print("\n" + "="*60)
    print("Testing PolynomialAttention")
    print("="*60)

    batch_size = 4
    seq_length = 256  # H * W
    dim = 128
    num_heads = 8
    max_order = 4

    # Create module
    poly_attn = PolynomialAttention(
        dim=dim,
        num_heads=num_heads,
        max_order=max_order,
        qkv_bias=True,
        dropout=0.1
    )

    print(f"Input: [{batch_size}, {seq_length}, {dim}]")
    print(f"Num heads: {num_heads}")
    print(f"Head dim: {poly_attn.head_dim}")
    print(f"Max polynomial order: {max_order}")
    print(f"Order weights: {poly_attn.order_weights.shape}")

    # Forward pass
    x = torch.randn(batch_size, seq_length, dim)
    output, attn_map = poly_attn(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention map shape: {attn_map.shape}")

    assert output.shape == x.shape, "Output shape should match input"
    assert attn_map.shape == (batch_size, num_heads, seq_length, seq_length)

    # Check attention map properties
    # Sum over last dimension should be 1 (softmax)
    attn_sum = attn_map.sum(dim=-1)
    print(f"Attention sum (should be ~1): {attn_sum.mean().item():.6f}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    # Check order weights
    print(f"\nPolynomial order weights (softmax normalized):")
    weights_normalized = torch.softmax(poly_attn.order_weights, dim=0)
    for i, w in enumerate(weights_normalized):
        print(f"  Order {i+2}: {w.item():.4f}")

    print("✓ PolynomialAttention test passed!")


def test_ciem():
    """Test Channel Interaction and Enhancement Module."""
    print("\n" + "="*60)
    print("Testing ChannelInteractionEnhancementModule (CIEM)")
    print("="*60)

    batch_size = 4
    channels = 256
    height, width = 16, 16
    num_groups = 4

    # Create module
    ciem = ChannelInteractionEnhancementModule(
        channels=channels,
        reduction=16,
        num_groups=num_groups
    )

    print(f"Input: [{batch_size}, {channels}, {height}, {width}]")
    print(f"Channel reduction ratio: 16")
    print(f"Number of channel groups: {num_groups}")
    print(f"Channels per group: {channels // num_groups}")

    # Forward pass
    x = torch.randn(batch_size, channels, height, width)
    output = ciem(x)

    print(f"\nOutput shape: {output.shape}")
    assert output.shape == x.shape, "Output shape should match input"

    # Check that enhancement is applied
    assert not torch.allclose(output, x), "Output should be enhanced"

    # Check residual connection (output should be close to input range)
    input_mean = x.mean().item()
    output_mean = output.mean().item()
    print(f"Input mean: {input_mean:.4f}")
    print(f"Output mean: {output_mean:.4f}")

    print("✓ CIEM test passed!")


def test_multi_granularity_fusion():
    """Test multi-granularity fusion module."""
    print("\n" + "="*60)
    print("Testing MultiGranularityFusion")
    print("="*60)

    batch_size = 4
    channels = 128
    height, width = 32, 32
    num_levels = 3

    # Test all fusion modes
    fusion_modes = ['concat', 'add', 'attention']

    for mode in fusion_modes:
        print(f"\nTesting fusion mode: {mode}")

        mgf = MultiGranularityFusion(
            channels=channels,
            num_levels=num_levels,
            fusion_mode=mode
        )

        x = torch.randn(batch_size, channels, height, width)
        output = mgf(x)

        print(f"  Input: {x.shape}")
        print(f"  Output: {output.shape}")

        assert output.shape == x.shape, f"Output shape should match input for mode {mode}"

        # Check residual connection
        diff = (output - x).abs().mean()
        print(f"  Mean difference from input: {diff.item():.6f}")

    print("\n✓ MultiGranularityFusion test passed!")


def test_cross_knowledge_propagation():
    """Test cross-knowledge propagation module."""
    print("\n" + "="*60)
    print("Testing CrossKnowledgePropagation")
    print("="*60)

    batch_size = 2
    channels = 128
    num_levels = 4

    # Create features at different resolutions
    # (from fine to coarse)
    level_features = []
    base_size = 64
    for i in range(num_levels):
        size = base_size // (2 ** i)
        feat = torch.randn(batch_size, channels, size, size)
        level_features.append(feat)
        print(f"Level {i}: {feat.shape}")

    # Test all propagation modes
    propagation_modes = ['bidirectional', 'bottom_up', 'top_down']

    for mode in propagation_modes:
        print(f"\nTesting propagation mode: {mode}")

        ckp = CrossKnowledgePropagation(
            channels=channels,
            num_levels=num_levels,
            propagation_mode=mode
        )

        output_features = ckp(level_features)

        assert len(output_features) == num_levels, "Should output same number of levels"

        for i, (input_feat, output_feat) in enumerate(zip(level_features, output_features)):
            assert input_feat.shape == output_feat.shape, f"Level {i} shape should be preserved"
            print(f"  Level {i} output: {output_feat.shape}")

    print("\n✓ CrossKnowledgePropagation test passed!")


def test_high_order_attention_single_level():
    """Test HighOrderAttention with single level."""
    print("\n" + "="*60)
    print("Testing HighOrderAttention (Single Level)")
    print("="*60)

    batch_size = 2
    channels = [128]
    height, width = 32, 32

    features = [torch.randn(batch_size, channels[0], height, width)]

    print(f"Input: {features[0].shape}")

    # Create module
    hoa = HighOrderAttention(
        channels=channels,
        num_heads=8,
        max_order=4,
        num_granularity_levels=3,
        propagation_mode='bidirectional'
    )

    total_params = sum(p.numel() for p in hoa.parameters())
    print(f"Total parameters: {total_params:,}")

    # Forward pass
    enhanced_features, attention_info = hoa(features)

    print(f"\nOutput: {enhanced_features[0].shape}")
    assert len(enhanced_features) == 1, "Should output same number of levels"
    assert enhanced_features[0].shape == features[0].shape, "Shape should be preserved"

    # Check attention info
    print(f"\nAttention info components:")
    for key in attention_info.keys():
        if isinstance(attention_info[key], list):
            print(f"  {key}: {len(attention_info[key])} items")
        else:
            print(f"  {key}: {type(attention_info[key])}")

    print("✓ Single-level HighOrderAttention test passed!")


def test_high_order_attention_multi_level():
    """Test HighOrderAttention with multiple levels."""
    print("\n" + "="*60)
    print("Testing HighOrderAttention (Multi-Level)")
    print("="*60)

    batch_size = 2
    channels = [64, 128, 320, 512]  # Typical backbone channels
    heights = [64, 32, 16, 8]
    widths = [64, 32, 16, 8]

    # Create multi-level features
    features = []
    for i in range(len(channels)):
        feat = torch.randn(batch_size, channels[i], heights[i], widths[i])
        features.append(feat)
        print(f"Level {i}: {feat.shape}")

    # Create module
    hoa = HighOrderAttention(
        channels=channels,
        num_heads=8,
        max_order=4,
        num_granularity_levels=3,
        propagation_mode='bidirectional'
    )

    total_params = sum(p.numel() for p in hoa.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Forward pass
    print("\nRunning forward pass...")
    enhanced_features, attention_info = hoa(features)

    print("\nOutput shapes:")
    for i, feat in enumerate(enhanced_features):
        print(f"Level {i}: {feat.shape}")
        assert feat.shape == features[i].shape, f"Level {i} shape should be preserved"

    # Check that features are enhanced
    for i in range(len(features)):
        diff = (enhanced_features[i] - features[i]).abs().mean()
        print(f"Level {i} mean difference: {diff.item():.6f}")
        assert diff.item() > 0, f"Level {i} should be enhanced"

    # Check attention info
    print(f"\nAttention info:")
    print(f"  Tucker features: {len(attention_info['tucker_features'])}")
    print(f"  Polynomial features: {len(attention_info['polynomial_features'])}")
    print(f"  CIEM features: {len(attention_info['ciem_features'])}")
    print(f"  Granularity features: {len(attention_info['granularity_features'])}")
    print(f"  Attention maps: {len(attention_info['attention_maps'])}")

    # Check attention map shapes
    for i, attn_map in enumerate(attention_info['attention_maps']):
        expected_seq_len = heights[i] * widths[i]
        print(f"  Attention map {i}: {attn_map.shape} (seq_len={expected_seq_len})")

    print("\n✓ Multi-level HighOrderAttention test passed!")


def test_gradient_flow():
    """Test gradient flow through HighOrderAttention."""
    print("\n" + "="*60)
    print("Testing Gradient Flow")
    print("="*60)

    batch_size = 2
    channels = [64, 128]
    heights = [16, 8]
    widths = [16, 8]

    # Create features with gradient tracking
    features = []
    for i in range(len(channels)):
        feat = torch.randn(batch_size, channels[i], heights[i], widths[i], requires_grad=True)
        features.append(feat)

    # Create module
    hoa = HighOrderAttention(
        channels=channels,
        num_heads=8,
        max_order=3
    )

    # Forward pass
    enhanced_features, _ = hoa(features)

    # Compute dummy loss
    loss = sum(feat.mean() for feat in enhanced_features)

    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients
    print("\nGradient statistics:")
    for i, feat in enumerate(features):
        if feat.grad is not None:
            grad_norm = feat.grad.norm().item()
            grad_mean = feat.grad.mean().item()
            print(f"  Level {i} grad norm: {grad_norm:.6f}, mean: {grad_mean:.6f}")
            assert grad_norm > 0, f"Level {i} should have gradients"
        else:
            print(f"  Level {i}: No gradient")

    # Check module gradients
    total_grad_norm = 0.0
    num_params_with_grad = 0
    for name, param in hoa.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item()
            num_params_with_grad += 1

    print(f"\nTotal gradient norm: {total_grad_norm:.6f}")
    print(f"Parameters with gradients: {num_params_with_grad}")

    assert num_params_with_grad > 0, "Should have gradients"

    print("\n✓ Gradient flow test passed!")


def test_performance():
    """Test computational performance."""
    print("\n" + "="*60)
    print("Testing Performance")
    print("="*60)

    import time

    batch_size = 4
    channels = [64, 128, 320, 512]
    heights = [64, 32, 16, 8]
    widths = [64, 32, 16, 8]

    # Create features
    features = []
    total_elements = 0
    for i in range(len(channels)):
        feat = torch.randn(batch_size, channels[i], heights[i], widths[i])
        features.append(feat)
        total_elements += feat.numel()

    print(f"Total input elements: {total_elements:,}")

    # Create module
    hoa = HighOrderAttention(
        channels=channels,
        num_heads=8,
        max_order=4
    )

    total_params = sum(p.numel() for p in hoa.parameters())
    print(f"Total parameters: {total_params:,}")

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = hoa(features)

    # Timing
    num_runs = 10
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            enhanced_features, _ = hoa(features)

    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    print(f"\nAverage forward pass time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {batch_size / avg_time:.2f} samples/sec")

    print("\n✓ Performance test completed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print(" " * 20 + "HIGH-ORDER ATTENTION TEST SUITE")
    print("="*80)

    try:
        # Test individual components
        test_tucker_decomposition()
        test_polynomial_attention()
        test_ciem()
        test_multi_granularity_fusion()
        test_cross_knowledge_propagation()

        # Test integrated module
        test_high_order_attention_single_level()
        test_high_order_attention_multi_level()

        # Test training capabilities
        test_gradient_flow()

        # Test performance
        test_performance()

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
