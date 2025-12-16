"""
Quick debug script to test metrics calculation with different scenarios.
This tests IoU and F-measure to verify they produce different values.
"""
import torch
import numpy as np

def test_metrics_calculation():
    """Test IoU and F-measure calculations with various scenarios."""
    print("=" * 60)
    print("METRICS CALCULATION DEBUG TEST")
    print("=" * 60)
    print()
    
    # Test scenarios
    scenarios = [
        # (name, pred_ratio, target_ratio, overlap_ratio)
        ("Model predicting nothing (p=0)", 0.0, 0.3, 0.0),
        ("Model predicting everything (p=1)", 1.0, 0.3, 1.0),
        ("Perfect prediction", 0.3, 0.3, 1.0),
        ("Slight overlap", 0.1, 0.3, 0.3),
        ("Typical early training (low pred)", 0.05, 0.2, 0.2),
        ("50% overlap", 0.25, 0.25, 0.5),
    ]
    
    H, W = 448, 448  # Typical image size
    
    for name, pred_ratio, target_ratio, overlap_ratio in scenarios:
        print(f"\n--- Scenario: {name} ---")
        print(f"  pred_ratio={pred_ratio}, target_ratio={target_ratio}, overlap={overlap_ratio}")
        
        # Create synthetic masks
        t = torch.zeros(H, W)
        target_pixels = int(H * W * target_ratio)
        if target_pixels > 0:
            t.view(-1)[:target_pixels] = 1.0
        
        p_bin = torch.zeros(H, W)
        if pred_ratio > 0:
            pred_pixels = int(H * W * pred_ratio)
            overlap_pixels = int(target_pixels * overlap_ratio)
            
            # Put overlap in target region
            p_bin.view(-1)[:min(overlap_pixels, target_pixels)] = 1.0
            
            # Put remaining predictions outside target
            remaining = pred_pixels - overlap_pixels
            if remaining > 0:
                p_bin.view(-1)[target_pixels:target_pixels + remaining] = 1.0
        
        # IoU calculation (exact code from optimized_trainer.py)
        inter_bin = (p_bin * t).sum()
        union_bin = p_bin.sum() + t.sum() - inter_bin
        iou = ((inter_bin + 1e-6) / (union_bin + 1e-6)).item()
        
        # F-measure calculation (exact code from optimized_trainer.py)
        tp = (p_bin * t).sum()
        fp = (p_bin * (1 - t)).sum()
        fn = ((1 - p_bin) * t).sum()
        prec = (tp + 1e-6) / (tp + fp + 1e-6)
        rec = (tp + 1e-6) / (tp + fn + 1e-6)
        beta = 0.3
        f_measure = (((1 + beta**2) * prec * rec) / (beta**2 * prec + rec + 1e-6)).item()
        
        print(f"  TP={tp.item():.0f}, FP={fp.item():.0f}, FN={fn.item():.0f}")
        print(f"  Precision={prec.item():.6f}, Recall={rec.item():.6f}")
        print(f"  Intersection={inter_bin.item():.0f}, Union={union_bin.item():.0f}")
        print()
        print(f"  >>> IoU       = {iou:.6f}")
        print(f"  >>> F-measure = {f_measure:.6f}")
        print(f"  >>> SAME? {abs(iou - f_measure) < 0.0001}")
        
    # Special case: what if model outputs are VERY close to 0.5?
    print("\n\n" + "=" * 60)
    print("EDGE CASE: What if ALL predictions hover around threshold?")
    print("=" * 60)
    
    # Simulate model outputting values around 0.49-0.51
    p_raw = torch.ones(1, 1, 448, 448) * 0.5 + (torch.rand(1, 1, 448, 448) - 0.5) * 0.02
    t = torch.zeros(1, 1, 448, 448)
    t[..., 100:200, 100:200] = 1.0  # 100x100 target region
    
    p_bin = (p_raw[0, 0] > 0.5).float()
    
    # IoU
    inter_bin = (p_bin * t[0, 0]).sum()
    union_bin = p_bin.sum() + t[0, 0].sum() - inter_bin
    iou = ((inter_bin + 1e-6) / (union_bin + 1e-6)).item()
    
    # F-measure
    tp = (p_bin * t[0, 0]).sum()
    fp = (p_bin * (1 - t[0, 0])).sum()
    fn = ((1 - p_bin) * t[0, 0]).sum()
    prec = (tp + 1e-6) / (tp + fp + 1e-6)
    rec = (tp + 1e-6) / (tp + fn + 1e-6)
    beta = 0.3
    f_measure = (((1 + beta**2) * prec * rec) / (beta**2 * prec + rec + 1e-6)).item()
    
    print(f"  pred_sum={p_bin.sum().item():.0f} (out of {448*448})")
    print(f"  target_sum={t.sum().item():.0f}")
    print(f"  >>> IoU       = {iou:.6f}")
    print(f"  >>> F-measure = {f_measure:.6f}")
    print(f"  >>> SAME? {abs(iou - f_measure) < 0.0001}")
    print()
    
    # But what if values are VERY low?
    print("\n\n" + "=" * 60)
    print("VERY LOW VALUES (typical in early training)")
    print("=" * 60)
    
    # Simulate very low IoU and F-measure
    p_bin = torch.zeros(448, 448)  # No predictions
    t = torch.zeros(448, 448)
    t[100:200, 100:200] = 1.0  # 100x100 target
    
    inter_bin = (p_bin * t).sum()
    union_bin = p_bin.sum() + t.sum() - inter_bin
    iou = ((inter_bin + 1e-6) / (union_bin + 1e-6)).item()
    
    tp = (p_bin * t).sum()
    fp = (p_bin * (1 - t)).sum()
    fn = ((1 - p_bin) * t).sum()
    prec = (tp + 1e-6) / (tp + fp + 1e-6)
    rec = (tp + 1e-6) / (tp + fn + 1e-6)
    beta = 0.3
    f_measure = (((1 + beta**2) * prec * rec) / (beta**2 * prec + rec + 1e-6)).item()
    
    print(f"  pred_sum={p_bin.sum().item():.0f}")
    print(f"  target_sum={t.sum().item():.0f}")
    print(f"  tp={tp.item():.0f}, fp={fp.item():.0f}, fn={fn.item():.0f}")
    print(f"  prec={prec.item():.10f}, rec={rec.item():.10f}")
    print(f"  >>> IoU       = {iou:.10f}")
    print(f"  >>> F-measure = {f_measure:.10f}")
    print(f"  >>> When displayed as 4 decimal places:")
    print(f"      IoU:       {iou:.4f}")
    print(f"      F-measure: {f_measure:.4f}")

if __name__ == "__main__":
    test_metrics_calculation()
