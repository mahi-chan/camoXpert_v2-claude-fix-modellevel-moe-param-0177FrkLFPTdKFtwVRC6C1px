#!/usr/bin/env python3
"""Quick test to verify F-measure formula is correct"""

import numpy as np
import sys
sys.path.insert(0, '/kaggle/working/camoXpert_v2')

from test import CODMetrics

# Create test data
pred = np.array([[0.8, 0.9, 0.2],
                  [0.7, 0.6, 0.1],
                  [0.3, 0.4, 0.5]])

gt = np.array([[1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]])

# Compute F-measure
metrics = CODMetrics()
f_score = metrics.f_measure(pred, gt, threshold=0.5, beta2=0.09)

print(f"F-measure result: {f_score:.4f}")

# Expected: with correct formula, should be around 0.7-0.8
# With buggy formula (no parentheses), would be very different

# Manual calculation to verify
pred_bin = (pred > 0.5).astype(np.float32)
gt_bin = (gt > 0.5).astype(np.float32)

tp = np.sum(pred_bin * gt_bin)
fp = np.sum(pred_bin * (1 - gt_bin))
fn = np.sum((1 - pred_bin) * gt_bin)

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)

# CORRECT formula
correct_f = ((1 + 0.09) * precision * recall) / (0.09 * precision + recall + 1e-8)

# BUGGY formula (no inner parentheses)
buggy_f = (1 + 0.09) * precision * recall / (0.09 * precision + recall + 1e-8)

print(f"\nManual calculations:")
print(f"  TP={tp}, FP={fp}, FN={fn}")
print(f"  Precision={precision:.4f}, Recall={recall:.4f}")
print(f"  CORRECT formula result: {correct_f:.4f}")
print(f"  BUGGY formula result: {buggy_f:.4f}")

print(f"\nVerification:")
if abs(f_score - correct_f) < 0.0001:
    print("✓ F-measure is using CORRECT formula!")
elif abs(f_score - buggy_f) < 0.0001:
    print("✗ F-measure is using BUGGY formula (no parentheses)!")
else:
    print(f"? F-measure result {f_score:.4f} doesn't match either formula")
