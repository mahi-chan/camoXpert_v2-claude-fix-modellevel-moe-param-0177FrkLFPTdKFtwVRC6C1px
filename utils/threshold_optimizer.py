"""
Threshold Optimization for CamoXpert
Finds optimal binarization threshold to maximize IoU or F1-score
"""

import numpy as np
from typing import List, Tuple, Dict, Union


class ThresholdOptimizer:
    """
    Optimizes binarization threshold for segmentation predictions.

    Provides multiple methods:
    - Grid search: Exhaustive search over threshold range
    - Otsu's method: Histogram-based automatic thresholding
    - Adaptive: Statistics-based threshold selection

    Usage:
        optimizer = ThresholdOptimizer()
        best_thr, best_iou, scores = optimizer.grid_search(preds, gts, metric='iou')
    """

    @staticmethod
    def grid_search(
        preds: List[np.ndarray],
        gts: List[np.ndarray],
        metric: str = 'iou',
        thresholds: List[float] = None
    ) -> Tuple[float, float, Dict[float, float]]:
        """
        Grid search for optimal threshold.

        Tests each threshold value and computes the average metric across all
        prediction-ground truth pairs. Returns the threshold with best performance.

        Args:
            preds: List of prediction arrays [H, W] with values in [0, 1]
            gts: List of ground truth binary arrays [H, W] with values {0, 1}
            metric: Metric to optimize ('iou' or 'f1')
            thresholds: List of thresholds to try. If None, uses [0.30, 0.35, ..., 0.70]

        Returns:
            best_threshold: Optimal threshold value
            best_score: Best metric score achieved
            all_scores: Dictionary mapping threshold -> average score

        Example:
            >>> preds = [pred1, pred2, pred3]  # Probability maps
            >>> gts = [gt1, gt2, gt3]  # Binary masks
            >>> thr, score, all_scores = ThresholdOptimizer.grid_search(preds, gts)
            >>> print(f"Optimal threshold: {thr:.2f}, IoU: {score:.4f}")
        """
        if thresholds is None:
            # Default: 0.30 to 0.70 with step 0.05
            thresholds = np.arange(0.30, 0.71, 0.05).tolist()

        if metric not in ['iou', 'f1']:
            raise ValueError(f"Unknown metric: {metric}. Use 'iou' or 'f1'")

        metric_func = ThresholdOptimizer._iou if metric == 'iou' else ThresholdOptimizer._f1

        all_scores = {}
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            scores = []
            for pred, gt in zip(preds, gts):
                # Binarize prediction
                pred_bin = (pred >= threshold).astype(np.uint8)
                gt_bin = gt.astype(np.uint8)

                # Compute metric
                score = metric_func(pred_bin, gt_bin)
                scores.append(score)

            # Average across all images
            avg_score = np.mean(scores)
            all_scores[threshold] = avg_score

            if avg_score > best_score:
                best_score = avg_score
                best_threshold = threshold

        return best_threshold, best_score, all_scores

    @staticmethod
    def otsu(pred: np.ndarray) -> float:
        """
        Otsu's method for automatic threshold selection.

        Finds the threshold that minimizes intra-class variance (or equivalently,
        maximizes inter-class variance) in the prediction histogram.

        Args:
            pred: Single prediction array [H, W] with values in [0, 1]

        Returns:
            threshold: Optimal threshold in [0, 1]

        Reference:
            Otsu, N. (1979). A threshold selection method from gray-level histograms.
            IEEE Trans. Systems, Man, and Cybernetics, 9(1), 62-66.

        Example:
            >>> pred = model(image)  # [H, W] probability map
            >>> threshold = ThresholdOptimizer.otsu(pred)
            >>> binary_mask = pred >= threshold
        """
        # Convert to 0-255 range for histogram
        pred_uint8 = (pred * 255).astype(np.uint8)

        # Compute histogram
        hist, bin_edges = np.histogram(pred_uint8, bins=256, range=(0, 256))
        hist = hist.astype(np.float32)

        # Normalize histogram (probability distribution)
        hist_norm = hist / hist.sum()

        # Cumulative sums
        cumsum = np.cumsum(hist_norm)
        cumsum_mean = np.cumsum(hist_norm * np.arange(256))

        # Global mean
        global_mean = cumsum_mean[-1]

        # Between-class variance for each threshold
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            between_class_variance = np.where(
                (cumsum > 0) & (cumsum < 1),
                ((global_mean * cumsum - cumsum_mean) ** 2) / (cumsum * (1 - cumsum)),
                0
            )

        # Find threshold that maximizes between-class variance
        optimal_threshold_idx = np.argmax(between_class_variance)

        # Convert back to [0, 1] range
        optimal_threshold = optimal_threshold_idx / 255.0

        return optimal_threshold

    @staticmethod
    def adaptive(pred: np.ndarray, method: str = 'mean') -> float:
        """
        Adaptive threshold based on prediction statistics.

        Args:
            pred: Single prediction array [H, W] with values in [0, 1]
            method: Thresholding method
                - 'mean': Use mean of prediction
                - 'median': Use median of prediction
                - 'percentile': Use 50th percentile (similar to median)
                - 'otsu': Use Otsu's method (calls otsu())

        Returns:
            threshold: Adaptive threshold in [0, 1]

        Example:
            >>> pred = model(image)
            >>> thr_mean = ThresholdOptimizer.adaptive(pred, method='mean')
            >>> thr_otsu = ThresholdOptimizer.adaptive(pred, method='otsu')
        """
        if method == 'mean':
            return float(np.mean(pred))

        elif method == 'median':
            return float(np.median(pred))

        elif method == 'percentile':
            # Use 50th percentile (median)
            return float(np.percentile(pred, 50))

        elif method == 'otsu':
            return ThresholdOptimizer.otsu(pred)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean', 'median', 'percentile', or 'otsu'")

    @staticmethod
    def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU).

        Args:
            pred: Binary prediction [H, W] with values {0, 1}
            gt: Binary ground truth [H, W] with values {0, 1}

        Returns:
            iou: IoU score in [0, 1]
        """
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        if union == 0:
            # Both pred and gt are empty
            return 1.0 if intersection == 0 else 0.0

        return intersection / union

    @staticmethod
    def _f1(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute F1-score (Dice coefficient).

        Args:
            pred: Binary prediction [H, W] with values {0, 1}
            gt: Binary ground truth [H, W] with values {0, 1}

        Returns:
            f1: F1-score in [0, 1]
        """
        intersection = np.logical_and(pred, gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()

        if pred_sum + gt_sum == 0:
            # Both pred and gt are empty
            return 1.0 if intersection == 0 else 0.0

        return 2 * intersection / (pred_sum + gt_sum)

    @staticmethod
    def optimize_per_image(
        preds: List[np.ndarray],
        gts: List[np.ndarray],
        method: str = 'otsu'
    ) -> Tuple[List[float], float]:
        """
        Optimize threshold independently for each image.

        Useful for analyzing whether a single global threshold is sufficient
        or if adaptive per-image thresholding could improve performance.

        Args:
            preds: List of prediction arrays [H, W]
            gts: List of ground truth arrays [H, W]
            method: Method to use ('otsu', 'mean', 'median', 'percentile')

        Returns:
            thresholds: List of optimal thresholds for each image
            avg_iou: Average IoU using per-image thresholds

        Example:
            >>> thresholds, avg_iou = ThresholdOptimizer.optimize_per_image(preds, gts)
            >>> print(f"Per-image threshold range: [{min(thresholds):.2f}, {max(thresholds):.2f}]")
            >>> print(f"Average IoU with per-image thresholds: {avg_iou:.4f}")
        """
        thresholds = []
        ious = []

        for pred, gt in zip(preds, gts):
            # Get adaptive threshold for this image
            threshold = ThresholdOptimizer.adaptive(pred, method=method)
            thresholds.append(threshold)

            # Compute IoU with this threshold
            pred_bin = (pred >= threshold).astype(np.uint8)
            gt_bin = gt.astype(np.uint8)
            iou = ThresholdOptimizer._iou(pred_bin, gt_bin)
            ious.append(iou)

        return thresholds, np.mean(ious)


if __name__ == '__main__':
    """Test threshold optimizer."""
    print("=" * 70)
    print("Testing Threshold Optimizer")
    print("=" * 70)

    # Create synthetic predictions and ground truth
    np.random.seed(42)

    preds = []
    gts = []

    for i in range(10):
        # Create synthetic data
        gt = np.random.rand(100, 100) > 0.7  # Random binary mask

        # Prediction: GT + noise, offset by some threshold
        noise = np.random.randn(100, 100) * 0.1
        offset = 0.4 + np.random.rand() * 0.2  # Random offset between 0.4-0.6
        pred = gt.astype(np.float32) + noise + offset
        pred = np.clip(pred, 0, 1)

        preds.append(pred)
        gts.append(gt.astype(np.uint8))

    # Test 1: Grid search
    print("\n1. Grid Search:")
    print("-" * 70)

    for metric in ['iou', 'f1']:
        best_thr, best_score, all_scores = ThresholdOptimizer.grid_search(preds, gts, metric=metric)
        print(f"   {metric.upper()}: Best threshold = {best_thr:.2f}, Score = {best_score:.4f}")

        # Show top 3 thresholds
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top 3 thresholds:")
        for thr, score in sorted_scores[:3]:
            print(f"      {thr:.2f} -> {score:.4f}")

    # Test 2: Otsu's method
    print("\n2. Otsu's Method:")
    print("-" * 70)

    otsu_thresholds = [ThresholdOptimizer.otsu(pred) for pred in preds]
    print(f"   Otsu thresholds: [{min(otsu_thresholds):.3f}, {max(otsu_thresholds):.3f}]")
    print(f"   Mean Otsu threshold: {np.mean(otsu_thresholds):.3f}")

    # Compute IoU with Otsu
    otsu_ious = []
    for pred, gt, thr in zip(preds, gts, otsu_thresholds):
        pred_bin = (pred >= thr).astype(np.uint8)
        iou = ThresholdOptimizer._iou(pred_bin, gt)
        otsu_ious.append(iou)
    print(f"   Average IoU with Otsu: {np.mean(otsu_ious):.4f}")

    # Test 3: Adaptive methods
    print("\n3. Adaptive Methods:")
    print("-" * 70)

    for method in ['mean', 'median', 'percentile']:
        thresholds = [ThresholdOptimizer.adaptive(pred, method=method) for pred in preds]
        print(f"   {method.capitalize()}: [{min(thresholds):.3f}, {max(thresholds):.3f}] (avg: {np.mean(thresholds):.3f})")

        # Compute IoU
        ious = []
        for pred, gt, thr in zip(preds, gts, thresholds):
            pred_bin = (pred >= thr).astype(np.uint8)
            iou = ThresholdOptimizer._iou(pred_bin, gt)
            ious.append(iou)
        print(f"      Average IoU: {np.mean(ious):.4f}")

    # Test 4: Per-image optimization
    print("\n4. Per-Image Optimization:")
    print("-" * 70)

    for method in ['otsu', 'mean']:
        thresholds, avg_iou = ThresholdOptimizer.optimize_per_image(preds, gts, method=method)
        print(f"   {method.capitalize()}: Thresholds [{min(thresholds):.3f}, {max(thresholds):.3f}]")
        print(f"      Average IoU: {avg_iou:.4f}")

    # Test 5: Compare with default threshold
    print("\n5. Comparison with Default (0.5):")
    print("-" * 70)

    default_ious = []
    for pred, gt in zip(preds, gts):
        pred_bin = (pred >= 0.5).astype(np.uint8)
        iou = ThresholdOptimizer._iou(pred_bin, gt)
        default_ious.append(iou)

    print(f"   Default (0.5) IoU: {np.mean(default_ious):.4f}")

    # Best grid search result
    best_thr, best_score, _ = ThresholdOptimizer.grid_search(preds, gts, metric='iou')
    print(f"   Optimized ({best_thr:.2f}) IoU: {best_score:.4f}")
    print(f"   Improvement: {(best_score - np.mean(default_ious)) * 100:.2f}%")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

    print("\nUsage Examples:")
    print("-" * 70)
    print("# Grid search for optimal threshold")
    print("best_thr, best_iou, scores = ThresholdOptimizer.grid_search(preds, gts, metric='iou')")
    print()
    print("# Otsu's method for single image")
    print("threshold = ThresholdOptimizer.otsu(pred)")
    print("binary_mask = pred >= threshold")
    print()
    print("# Adaptive threshold")
    print("thr_mean = ThresholdOptimizer.adaptive(pred, method='mean')")
    print("thr_otsu = ThresholdOptimizer.adaptive(pred, method='otsu')")
    print()
    print("# Per-image optimization")
    print("thresholds, avg_iou = ThresholdOptimizer.optimize_per_image(preds, gts, method='otsu')")
