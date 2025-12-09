"""
CamoXpert Evaluation Script

Evaluates trained models on multiple COD benchmark datasets:
- COD10K (test set)
- CHAMELEON
- CAMO (test set)
- NC4K (test set)

Computes comprehensive metrics:
- S-measure (Structure Measure)
- F-measure (Weighted F-score)
- E-measure (Enhanced-alignment Measure)
- MAE (Mean Absolute Error)
- IoU (Intersection over Union)

Supports Test-Time Augmentation (TTA) for improved performance.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import cv2
from scipy.ndimage import distance_transform_edt

from models.model_level_moe import ModelLevelMoE
from utils.threshold_optimizer import ThresholdOptimizer
from utils.crf_refiner import CRFRefiner


class CODMetrics:
    """
    Camouflaged Object Detection Metrics.

    All methods accept numpy arrays [H, W] with values in [0, 1].
    Implements standard COD evaluation metrics from literature.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.total_metrics = {}
        self.num_samples = 0

    def s_measure(self, pred, gt, alpha=0.5):
        """
        Structure Measure (S-measure).

        Evaluates structural similarity between prediction and ground truth.
        Combines object-level and region-level assessments.

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]
            alpha: Balance between object and region scores (default: 0.5)

        Returns:
            S-measure score (higher is better, range [0, 1])
        """
        y = np.mean(gt)

        if y == 0:  # No object in GT
            return 1.0 - np.mean(pred)
        elif y == 1:  # Entire image is object
            return np.mean(pred)
        else:
            # Object-level score
            So = self._s_object(pred, gt)
            # Region-level score
            Sr = self._s_region(pred, gt)
            # Combined score
            return alpha * So + (1 - alpha) * Sr

    def _s_object(self, pred, gt):
        """Compute object-level structure similarity."""
        # Foreground
        pred_fg = pred * gt
        O_fg = self._object_score(pred_fg, gt)

        # Background
        pred_bg = (1 - pred) * (1 - gt)
        O_bg = self._object_score(pred_bg, 1 - gt)

        # Weighted combination
        u = np.mean(gt)
        return u * O_fg + (1 - u) * O_bg

    def _object_score(self, pred, gt):
        """Compute object score."""
        gt_sum = np.sum(gt)
        if gt_sum == 0:
            return 0.0

        pred_mean = np.sum(pred) / (gt_sum + 1e-8)
        sigma = np.sum((pred - pred_mean) ** 2) / (gt_sum + 1e-8)

        return 2.0 * pred_mean / (pred_mean ** 2 + 1.0 + sigma + 1e-8)

    def _s_region(self, pred, gt):
        """Compute region-level structure similarity."""
        # Find centroid
        X, Y = self._centroid(gt)

        # Divide into 4 regions
        pred1 = pred[:Y, :X]
        pred2 = pred[:Y, X:]
        pred3 = pred[Y:, :X]
        pred4 = pred[Y:, X:]

        gt1 = gt[:Y, :X]
        gt2 = gt[:Y, X:]
        gt3 = gt[Y:, :X]
        gt4 = gt[Y:, X:]

        # Compute SSIM for each region
        Q1 = self._ssim(pred1, gt1)
        Q2 = self._ssim(pred2, gt2)
        Q3 = self._ssim(pred3, gt3)
        Q4 = self._ssim(pred4, gt4)

        # Compute weights
        H, W = gt.shape
        w1 = X * Y / (H * W + 1e-8)
        w2 = (W - X) * Y / (H * W + 1e-8)
        w3 = X * (H - Y) / (H * W + 1e-8)
        w4 = (W - X) * (H - Y) / (H * W + 1e-8)

        # Weighted combination
        return w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    def _centroid(self, gt):
        """Compute centroid of ground truth mask."""
        H, W = gt.shape
        rows = np.arange(H)
        cols = np.arange(W)

        total = np.sum(gt) + 1e-8

        # Column centroid
        X = int(np.sum(np.sum(gt, axis=0) * cols) / total)
        # Row centroid
        Y = int(np.sum(np.sum(gt, axis=1) * rows) / total)

        # Clamp to valid range
        X = max(1, min(X, W - 1))
        Y = max(1, min(Y, H - 1))

        return X, Y

    def _ssim(self, pred, gt):
        """Compute structural similarity (SSIM)."""
        H, W = pred.shape

        if H < 2 or W < 2:
            return 0.0

        N = H * W

        # Means
        x = np.mean(pred)
        y = np.mean(gt)

        # Variances and covariance
        sigma_x2 = np.sum((pred - x) ** 2) / (N - 1 + 1e-8)
        sigma_y2 = np.sum((gt - y) ** 2) / (N - 1 + 1e-8)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1 + 1e-8)

        # SSIM formula
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

        if alpha != 0:
            return alpha / (beta + 1e-8)
        elif beta == 0:
            return 1.0
        else:
            return 0.0

    def f_measure(self, pred, gt, threshold=0.5, beta2=0.09):
        """
        F-measure (weighted F-score).

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]
            threshold: Binary threshold (default: 0.5)
            beta2: Beta squared for F-beta score (default: 0.09, matching training)

        Returns:
            F-measure score (higher is better, range [0, 1])
        """
        # Binarize
        pred_bin = (pred > threshold).astype(np.float32)
        gt_bin = (gt > threshold).astype(np.float32)

        # True positives, false positives, false negatives
        tp = np.sum(pred_bin * gt_bin)
        fp = np.sum(pred_bin * (1 - gt_bin))
        fn = np.sum((1 - pred_bin) * gt_bin)

        # Precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # F-measure (with proper parentheses to fix operator precedence)
        f_score = ((1 + beta2) * precision * recall) / (beta2 * precision + recall + 1e-8)

        return f_score

    def e_measure(self, pred, gt):
        """
        Enhanced-alignment Measure (E-measure).

        Evaluates pixel-level and image-level alignment.

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]

        Returns:
            E-measure score (higher is better, range [0, 1])
        """
        if np.sum(gt) == 0:  # No object
            return 1.0 - np.mean(pred)

        # Enhanced alignment matrix
        enhanced = self._enhanced_alignment_matrix(pred, gt)

        return np.mean(enhanced)

    def _enhanced_alignment_matrix(self, pred, gt):
        """Compute enhanced alignment matrix."""
        # Binarize GT for foreground/background
        gt_fg = (gt > 0.5).astype(np.float32)
        gt_bg = 1 - gt_fg

        # Compute alignment
        pred_mean = np.mean(pred)

        alignment = np.zeros_like(pred)

        # Foreground alignment
        if np.sum(gt_fg) > 0:
            alignment += ((pred - pred_mean) ** 2) * gt_fg / (np.sum(gt_fg) + 1e-8)

        # Background alignment
        if np.sum(gt_bg) > 0:
            alignment += ((pred - pred_mean) ** 2) * gt_bg / (np.sum(gt_bg) + 1e-8)

        # Enhanced matrix
        enhanced = 2 * alignment

        # Normalize to [0, 1]
        enhanced = 1.0 / (1.0 + enhanced)

        return enhanced

    def mae(self, pred, gt):
        """
        Mean Absolute Error.

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]

        Returns:
            MAE score (lower is better, range [0, 1])
        """
        return np.mean(np.abs(pred - gt))

    def iou(self, pred, gt, threshold=0.5):
        """
        Intersection over Union (IoU).

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]
            threshold: Binary threshold (default: 0.5)

        Returns:
            IoU score (higher is better, range [0, 1])
        """
        # Binarize
        pred_bin = (pred > threshold).astype(np.float32)
        gt_bin = (gt > threshold).astype(np.float32)

        # Intersection and union
        intersection = np.sum(pred_bin * gt_bin)
        union = np.sum(pred_bin) + np.sum(gt_bin) - intersection

        return intersection / (union + 1e-8)

    def weighted_f_measure(self, pred, gt, threshold=0.5):
        """
        Distance-weighted F-measure.

        Weights errors based on distance to object boundary.
        Errors near boundaries are penalized more heavily.

        Args:
            pred: Prediction array [H, W] with values in [0, 1]
            gt: Ground truth array [H, W] with values in [0, 1]
            threshold: Binary threshold (default: 0.5)

        Returns:
            Weighted F-measure score (higher is better, range [0, 1])
        """
        from scipy.ndimage import distance_transform_edt

        # Binarize
        pred_bin = (pred > threshold).astype(np.uint8)
        gt_bin = (gt > threshold).astype(np.uint8)

        # Compute distance transforms
        # Distance to nearest boundary
        dt_gt = distance_transform_edt(gt_bin) + distance_transform_edt(1 - gt_bin)

        # Inverse distance weighting (closer to boundary = higher weight)
        weights = 1.0 / (dt_gt + 1.0)
        weights = weights / np.max(weights)  # Normalize

        # Weighted true positives, false positives, false negatives
        tp = np.sum(weights * pred_bin * gt_bin)
        fp = np.sum(weights * pred_bin * (1 - gt_bin))
        fn = np.sum(weights * (1 - pred_bin) * gt_bin)

        # Weighted precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # Weighted F-measure
        f_score = 2 * precision * recall / (precision + recall + 1e-8)

        return f_score

    def update(self, pred, gt, threshold=0.5):
        """
        Update metrics with a new sample.

        Args:
            pred: Prediction tensor or array [B, 1, H, W] or [H, W]
            gt: Ground truth tensor or array [B, 1, H, W] or [H, W]
            threshold: Binary threshold
        """
        # Convert to numpy if tensor
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()

        # Squeeze to [H, W]
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)

        # Compute all metrics
        metrics = {
            'S-measure': self.s_measure(pred, gt),
            'F-measure': self.f_measure(pred, gt, threshold),
            'E-measure': self.e_measure(pred, gt),
            'MAE': self.mae(pred, gt),
            'IoU': self.iou(pred, gt, threshold),
            'Weighted-F': self.weighted_f_measure(pred, gt, threshold)
        }

        # Accumulate
        if self.num_samples == 0:
            self.total_metrics = metrics
        else:
            for k, v in metrics.items():
                self.total_metrics[k] += v

        self.num_samples += 1

    def compute(self):
        """
        Compute average metrics.

        Returns:
            Dictionary of averaged metrics
        """
        if self.num_samples == 0:
            return {}

        return {k: v / self.num_samples for k, v in self.total_metrics.items()}


class TTAPredictor:
    """
    Test-Time Augmentation Predictor.

    Applies multi-scale and flip augmentations during inference to improve predictions.
    Averages predictions across all augmentations for robust results.
    """

    def __init__(self, model, scales=[0.75, 1.0, 1.25], flip=True):
        """
        Initialize TTA predictor.

        Args:
            model: Trained model for inference
            scales: List of scale factors for multi-scale testing (default: [0.75, 1.0, 1.25])
            flip: Whether to use horizontal flip augmentation (default: True)
        """
        self.model = model
        self.scales = scales
        self.flip = flip
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        """
        Perform TTA prediction on input image.

        Applies multi-scale testing and optional horizontal flipping,
        then averages all predictions for final result.

        Args:
            image: Input tensor [B, C, H, W]

        Returns:
            Averaged prediction tensor [B, 1, H, W]
        """
        B, C, H, W = image.shape
        predictions = []

        for scale in self.scales:
            # Multi-scale augmentation
            if scale != 1.0:
                new_h, new_w = int(H * scale), int(W * scale)
                scaled_img = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                scaled_img = image

            # Forward pass
            output = self.model(scaled_img)
            pred = output['pred'] if isinstance(output, dict) else (output[0] if isinstance(output, tuple) else output)

            # Resize back to original size
            if scale != 1.0:
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

            predictions.append(pred)

            # Horizontal flip augmentation
            if self.flip:
                # Flip image horizontally
                flipped_img = torch.flip(scaled_img, dims=[3])

                # Forward pass on flipped image
                output_flip = self.model(flipped_img)
                pred_flip = output_flip['pred'] if isinstance(output_flip, dict) else (output_flip[0] if isinstance(output_flip, tuple) else output_flip)

                # Resize back if needed
                if scale != 1.0:
                    pred_flip = F.interpolate(pred_flip, size=(H, W), mode='bilinear', align_corners=False)

                # Un-flip prediction to match original orientation
                pred_flip = torch.flip(pred_flip, dims=[3])

                predictions.append(pred_flip)

        # Average all predictions
        final_pred = torch.stack(predictions, dim=0).mean(dim=0)

        return final_pred

    def __call__(self, image):
        """Allow class instance to be called like a function."""
        return self.predict(image)


class CODTestDataset(Dataset):
    """
    Dataset for testing on COD benchmarks.

    Expected directory structure:
    dataset_root/
        Images/
            img1.jpg
            img2.png
            ...
        GT/
            img1.png
            img2.png
            ...

    Alternatively, can specify image_dir and gt_dir directly.
    """

    def __init__(self, root=None, image_size=352, image_dir=None, gt_dir=None):
        self.image_size = image_size

        # Support two modes: root directory OR explicit image_dir/gt_dir
        if image_dir is not None and gt_dir is not None:
            # Explicit paths provided
            self.image_dir = Path(image_dir)
            self.gt_dir = Path(gt_dir)
            self.root = self.image_dir.parent  # For display purposes
        elif root is not None:
            # Root directory provided - use subdirectories
            self.root = Path(root)
            self.image_dir = self.root / 'Images' if (self.root / 'Images').exists() else self.root / 'Imgs'
            self.gt_dir = self.root / 'GT'
        else:
            raise ValueError("Must provide either root OR both image_dir and gt_dir")

        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.gt_dir.exists():
            raise ValueError(f"GT directory not found: {self.gt_dir}")

        # Get image files
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

        # Match GT files
        self.gt_files = []
        for img_file in self.image_files:
            gt_file = self.gt_dir / f"{img_file.stem}.png"
            if not gt_file.exists():
                # Try .jpg extension for GT
                gt_file = self.gt_dir / f"{img_file.stem}.jpg"
            if gt_file.exists():
                self.gt_files.append(gt_file)
            else:
                raise ValueError(f"GT file not found for {img_file.name}")

        print(f"  ✓ Loaded {len(self.image_files)} images from {self.root.name}")

        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_files[idx]).convert('RGB')
        original_size = img.size

        # Load GT
        gt = Image.open(self.gt_files[idx]).convert('L')

        # Apply transforms
        img_tensor = self.img_transform(img)
        gt_tensor = self.gt_transform(gt)

        return {
            'image': img_tensor,
            'gt': gt_tensor,
            'name': self.image_files[idx].stem,
            'original_size': original_size
        }


class Visualizer:
    """
    Visualization utilities for COD predictions.

    Creates various visualization outputs:
    - Overlay with TP/FP/FN (Green/Red/Blue)
    - Boundary comparison
    - Error heatmap
    - Segmented output (object on white background)
    - Cutout with transparent background (RGBA)
    - Comprehensive comparison figure
    """

    def __init__(self):
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    def denormalize_image(self, img_tensor):
        """
        Denormalize image tensor back to [0, 255] RGB.

        Args:
            img_tensor: [C, H, W] normalized tensor

        Returns:
            numpy array [H, W, 3] in uint8 format
        """
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        img = img_tensor.cpu().numpy()
        img = img * std + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

        return img

    def create_overlay(self, image, pred, gt, threshold=0.5):
        """
        Create TP/FP/FN overlay visualization.

        Green: True Positives
        Red: False Positives
        Blue: False Negatives

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)
            threshold: Binary threshold

        Returns:
            PIL Image with overlay
        """
        # Denormalize image
        img = self.denormalize_image(image)

        # Binarize predictions and GT
        pred_bin = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)
        gt_bin = (gt.squeeze().cpu().numpy() > threshold).astype(np.uint8)

        # Compute TP, FP, FN
        tp = (pred_bin == 1) & (gt_bin == 1)
        fp = (pred_bin == 1) & (gt_bin == 0)
        fn = (pred_bin == 0) & (gt_bin == 1)

        # Create overlay
        overlay = img.copy()

        # Green for TP
        overlay[tp] = (overlay[tp] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        # Red for FP
        overlay[fp] = (overlay[fp] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        # Blue for FN
        overlay[fn] = (overlay[fn] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

        return Image.fromarray(overlay)

    def create_boundary_overlay(self, image, pred, gt, threshold=0.5):
        """
        Create boundary comparison overlay.

        GT boundary: Green
        Pred boundary: Red

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)
            threshold: Binary threshold

        Returns:
            PIL Image with boundary overlay
        """
        # Denormalize image
        img = self.denormalize_image(image)

        # Binarize
        pred_bin = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
        gt_bin = (gt.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

        # Extract boundaries using Canny edge detection
        pred_boundary = cv2.Canny(pred_bin, 50, 150)
        gt_boundary = cv2.Canny(gt_bin, 50, 150)

        # Create overlay
        overlay = img.copy()

        # Green for GT boundary
        overlay[gt_boundary > 0] = [0, 255, 0]

        # Red for Pred boundary
        overlay[pred_boundary > 0] = [255, 0, 0]

        return Image.fromarray(overlay)

    def create_error_map(self, pred, gt):
        """
        Create error heatmap showing prediction errors.

        Args:
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)

        Returns:
            PIL Image with error heatmap
        """
        # Compute absolute error
        error = torch.abs(pred - gt).squeeze().cpu().numpy()

        # Create heatmap using matplotlib colormap
        cmap = plt.cm.jet
        error_colored = (cmap(error)[:, :, :3] * 255).astype(np.uint8)

        return Image.fromarray(error_colored)

    def create_segmented_output(self, image, pred, threshold=0.5):
        """
        Create segmented output: predicted object on white background.

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            threshold: Binary threshold

        Returns:
            PIL Image with object on white background
        """
        # Denormalize image
        img = self.denormalize_image(image)

        # Binarize prediction
        mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)

        # Create white background
        output = np.ones_like(img) * 255

        # Paste object
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        output = np.where(mask_3ch, img, output).astype(np.uint8)

        return Image.fromarray(output)

    def create_cutout(self, image, pred, threshold=0.5):
        """
        Create cutout with transparent background (RGBA).

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            threshold: Binary threshold

        Returns:
            PIL Image in RGBA mode with transparent background
        """
        # Denormalize image
        img = self.denormalize_image(image)

        # Binarize prediction
        mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

        # Create RGBA image
        rgba = np.dstack([img, mask])

        return Image.fromarray(rgba, mode='RGBA')

    def create_comparison_figure(self, image, pred, gt, metrics, name, threshold=0.5):
        """
        Create comprehensive 12-panel comparison figure.

        Layout:
        Row 1: Original | GT | Prediction | Overlay (TP/FP/FN)
        Row 2: Boundary Overlay | Error Map | Segmented | Cutout
        Row 3: Pred Heatmap | GT Binary | Pred Binary | Metrics

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)
            metrics: Dictionary of computed metrics
            name: Sample name
            threshold: Binary threshold

        Returns:
            matplotlib Figure
        """
        # Create figure with GridSpec for better layout
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Denormalize image
        img = self.denormalize_image(image)
        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()

        # Row 1: Original | GT | Prediction | Overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gt_np, cmap='gray')
        ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(pred_np, cmap='gray')
        ax3.set_title('Prediction', fontsize=14, fontweight='bold')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        overlay = np.array(self.create_overlay(image, pred, gt, threshold))
        ax4.imshow(overlay)
        ax4.set_title('TP/FP/FN Overlay', fontsize=14, fontweight='bold')
        ax4.axis('off')
        # Add legend
        green_patch = mpatches.Patch(color='green', label='True Positive')
        red_patch = mpatches.Patch(color='red', label='False Positive')
        blue_patch = mpatches.Patch(color='blue', label='False Negative')
        ax4.legend(handles=[green_patch, red_patch, blue_patch], loc='upper right', fontsize=10)

        # Row 2: Boundary | Error Map | Segmented | Cutout
        ax5 = fig.add_subplot(gs[1, 0])
        boundary = np.array(self.create_boundary_overlay(image, pred, gt, threshold))
        ax5.imshow(boundary)
        ax5.set_title('Boundary Comparison', fontsize=14, fontweight='bold')
        ax5.axis('off')
        # Add legend
        green_line = mpatches.Patch(color='green', label='GT Boundary')
        red_line = mpatches.Patch(color='red', label='Pred Boundary')
        ax5.legend(handles=[green_line, red_line], loc='upper right', fontsize=10)

        ax6 = fig.add_subplot(gs[1, 1])
        error_map = np.array(self.create_error_map(pred, gt))
        im = ax6.imshow(error_map)
        ax6.set_title('Error Heatmap', fontsize=14, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

        ax7 = fig.add_subplot(gs[1, 2])
        segmented = np.array(self.create_segmented_output(image, pred, threshold))
        ax7.imshow(segmented)
        ax7.set_title('Segmented Output', fontsize=14, fontweight='bold')
        ax7.axis('off')

        ax8 = fig.add_subplot(gs[1, 3])
        cutout = np.array(self.create_cutout(image, pred, threshold))
        # Create checkered background for transparency visualization
        checker = np.indices((cutout.shape[0], cutout.shape[1])).sum(axis=0) % 20 < 10
        checker_bg = np.ones((cutout.shape[0], cutout.shape[1], 3)) * 200
        checker_bg[checker] = 150
        # Blend with alpha
        alpha = cutout[:, :, 3:4] / 255.0
        blended = cutout[:, :, :3] * alpha + checker_bg * (1 - alpha)
        ax8.imshow(blended.astype(np.uint8))
        ax8.set_title('Cutout (RGBA)', fontsize=14, fontweight='bold')
        ax8.axis('off')

        # Row 3: Heatmaps and Metrics
        ax9 = fig.add_subplot(gs[2, 0])
        im = ax9.imshow(pred_np, cmap='hot')
        ax9.set_title('Prediction Heatmap', fontsize=14, fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)

        ax10 = fig.add_subplot(gs[2, 1])
        pred_bin = (pred_np > threshold).astype(float)
        ax10.imshow(pred_bin, cmap='gray')
        ax10.set_title('Prediction Binary', fontsize=14, fontweight='bold')
        ax10.axis('off')

        ax11 = fig.add_subplot(gs[2, 2])
        gt_bin = (gt_np > threshold).astype(float)
        ax11.imshow(gt_bin, cmap='gray')
        ax11.set_title('GT Binary', fontsize=14, fontweight='bold')
        ax11.axis('off')

        # Metrics panel
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis('off')

        metrics_text = f"""
        Sample: {name}

        COD Metrics:
        ━━━━━━━━━━━━━━━━━━━━
        S-measure:   {metrics['S-measure']:.4f} ⭐
        F-measure:   {metrics['F-measure']:.4f}
        E-measure:   {metrics['E-measure']:.4f}
        Weighted-F:  {metrics['Weighted-F']:.4f}
        MAE:         {metrics['MAE']:.4f}
        IoU:         {metrics['IoU']:.4f}
        """

        ax12.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                 verticalalignment='center', transform=ax12.transAxes)

        # Overall title
        fig.suptitle(f'CamoXpert Evaluation: {name}', fontsize=18, fontweight='bold', y=0.98)

        plt.tight_layout()

        return fig

    def save_individual_outputs(self, image, pred, gt, name, save_dir, threshold=0.5):
        """
        Save all individual visualizations to files.

        Args:
            image: [C, H, W] normalized image tensor
            pred: [1, H, W] prediction tensor (0-1)
            gt: [1, H, W] ground truth tensor (0-1)
            name: Sample name
            save_dir: Directory to save outputs
            threshold: Binary threshold

        Returns:
            Dictionary mapping output type to file path
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Original image
        img = self.denormalize_image(image)
        img_pil = Image.fromarray(img)
        img_path = save_dir / f"{name}_original.png"
        img_pil.save(img_path)
        outputs['original'] = img_path

        # Ground truth
        gt_np = (gt.squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt_pil = Image.fromarray(gt_np, mode='L')
        gt_path = save_dir / f"{name}_gt.png"
        gt_pil.save(gt_path)
        outputs['gt'] = gt_path

        # Prediction
        pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_np, mode='L')
        pred_path = save_dir / f"{name}_pred.png"
        pred_pil.save(pred_path)
        outputs['pred'] = pred_path

        # Overlay
        overlay_path = save_dir / f"{name}_overlay.png"
        self.create_overlay(image, pred, gt, threshold).save(overlay_path)
        outputs['overlay'] = overlay_path

        # Boundary overlay
        boundary_path = save_dir / f"{name}_boundary.png"
        self.create_boundary_overlay(image, pred, gt, threshold).save(boundary_path)
        outputs['boundary'] = boundary_path

        # Error map
        error_path = save_dir / f"{name}_error.png"
        self.create_error_map(pred, gt).save(error_path)
        outputs['error'] = error_path

        # Segmented output
        segmented_path = save_dir / f"{name}_segmented.png"
        self.create_segmented_output(image, pred, threshold).save(segmented_path)
        outputs['segmented'] = segmented_path

        # Cutout (RGBA)
        cutout_path = save_dir / f"{name}_cutout.png"
        self.create_cutout(image, pred, threshold).save(cutout_path)
        outputs['cutout'] = cutout_path

        return outputs


def load_checkpoint(checkpoint_path, num_experts=None, device='cuda', use_dataparallel=False):
    """
    Load model from checkpoint, handling DDP 'module.' prefix.

    Args:
        checkpoint_path: Path to checkpoint file
        num_experts: Number of experts in the model (auto-detected if None)
        device: Device to load model on
        use_dataparallel: Whether to use DataParallel for multi-GPU inference

    Returns:
        Loaded model in eval mode
    """
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Checkpoint from epoch: {epoch}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    # Auto-detect num_experts from checkpoint if not specified
    if num_experts is None:
        # Look for router decision network output layer to detect num_experts
        for key in new_state_dict.keys():
            if 'router.decision_network' in key and 'weight' in key:
                # The last layer outputs num_experts probabilities
                if new_state_dict[key].dim() == 2:  # FC layer
                    detected_experts = new_state_dict[key].shape[0]
                    num_experts = detected_experts
                    print(f"✓ Auto-detected {num_experts} experts from checkpoint")
                    break

        if num_experts is None:
            raise ValueError("Could not auto-detect num_experts from checkpoint. Please specify --num-experts")

    # Create model
    model = ModelLevelMoE(
        backbone_name='pvt_v2_b2',
        num_experts=num_experts,
        top_k=2,
        pretrained=False
    )

    # Load state dict
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    # Use DataParallel for multi-GPU inference if available
    if use_dataparallel and torch.cuda.device_count() > 1:
        print(f"✓ Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    print(f"✓ Model loaded successfully")
    print(f"{'='*70}\n")

    return model


def test_time_augmentation(model, image, scales=[0.75, 1.0, 1.25], use_flip=True):
    """
    Apply Test-Time Augmentation (TTA) for improved predictions.

    Args:
        model: Trained model
        image: Input image tensor [1, 3, H, W]
        scales: List of scale factors for multi-scale testing
        use_flip: Whether to use horizontal flip augmentation

    Returns:
        Averaged prediction [1, 1, H, W]
    """
    B, C, H, W = image.shape
    predictions = []

    for scale in scales:
        # Resize image
        if scale != 1.0:
            new_h, new_w = int(H * scale), int(W * scale)
            scaled_img = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            scaled_img = image

        # Forward pass
        with torch.no_grad():
            output = model(scaled_img)
            pred = output['pred'] if isinstance(output, dict) else (output[0] if isinstance(output, tuple) else output)

            # Resize back to original size
            if scale != 1.0:
                pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

            predictions.append(pred)

        # Flip augmentation
        if use_flip:
            flipped_img = torch.flip(scaled_img, dims=[3])  # Horizontal flip

            with torch.no_grad():
                output_flip = model(flipped_img)
                pred_flip = output_flip['pred'] if isinstance(output_flip, dict) else (output_flip[0] if isinstance(output_flip, tuple) else output_flip)

                # Resize back
                if scale != 1.0:
                    pred_flip = F.interpolate(pred_flip, size=(H, W), mode='bilinear', align_corners=False)

                # Un-flip prediction
                pred_flip = torch.flip(pred_flip, dims=[3])
                predictions.append(pred_flip)

    # Average all predictions
    final_pred = torch.stack(predictions, dim=0).mean(dim=0)

    return final_pred


def evaluate_dataset(model, dataset, device, use_tta=False, use_crf=False, threshold=0.5,
                     threshold_method='fixed', output_dir=None,
                     save_visualizations=False, num_vis_samples=None, collect_for_optimization=False,
                     batch_size=1):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataset: Test dataset
        device: Device to run evaluation on
        use_tta: Whether to use Test-Time Augmentation
        use_crf: Whether to use CRF post-processing
        threshold: Threshold for binary prediction (used if threshold_method='fixed')
        threshold_method: Method for threshold selection ('fixed', 'otsu', 'adaptive-mean', 'adaptive-median')
        output_dir: Optional directory to save prediction masks
        save_visualizations: Whether to save visualizations
        num_vis_samples: Number of samples to visualize (None = all)
        collect_for_optimization: If True, collect all predictions and GTs for threshold optimization
        batch_size: Batch size for evaluation (only effective when TTA/CRF disabled)

    Returns:
        Dictionary of computed metrics (and optionally lists of predictions and GTs)
    """
    # Force batch_size=1 if using TTA or CRF (they require per-image processing)
    if use_tta or use_crf:
        actual_batch_size = 1
    else:
        actual_batch_size = batch_size

    # Optimized DataLoader for maximum GPU utilization
    dataloader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=8,  # Increased for faster data loading
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=4,  # Prefetch more batches
        persistent_workers=True  # Keep workers alive between iterations
    )
    metrics = CODMetrics()

    # Create output directory if saving predictions
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizer if needed
    visualizer = Visualizer() if save_visualizations else None
    vis_count = 0

    # Create CRF refiner if needed
    crf_refiner = CRFRefiner() if use_crf else None

    # Collect predictions and GTs for threshold optimization
    all_preds = [] if collect_for_optimization else None
    all_gts = [] if collect_for_optimization else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Evaluating {dataset.root.name}"):
            images = batch['image'].to(device)
            gts = batch['gt'].to(device)
            names = batch['name']

            current_batch_size = images.size(0)

            # Forward pass (batch processing when TTA/CRF disabled)
            if use_tta or use_crf:
                # Process one image at a time for TTA/CRF
                assert current_batch_size == 1, "TTA/CRF requires batch_size=1"
                image = images
                gt = gts
                name = names[0]

                if use_tta:
                    pred = test_time_augmentation(model, image)
                else:
                    output = model(image)
                    pred = output['pred'] if isinstance(output, dict) else (output[0] if isinstance(output, tuple) else output)

                pred = torch.sigmoid(pred)

                if use_crf:
                    img_for_crf = image.squeeze(0)
                    pred_refined = crf_refiner.refine(img_for_crf, pred)
                    pred = torch.from_numpy(pred_refined).unsqueeze(0).unsqueeze(0).to(device)

                # Process as single item
                preds = [pred]
                gt_list = [gt]
                name_list = [name]
            else:
                # Batch forward pass
                output_batch = model(images)
                preds_batch = output_batch['pred'] if isinstance(output_batch, dict) else (output_batch[0] if isinstance(output_batch, tuple) else output_batch)
                preds_batch = torch.sigmoid(preds_batch)

                # Split batch into individual predictions
                preds = [preds_batch[i:i+1] for i in range(current_batch_size)]
                gt_list = [gts[i:i+1] for i in range(current_batch_size)]
                name_list = names

            # Process each prediction in the batch
            for idx_in_batch, (pred, gt, name) in enumerate(zip(preds, gt_list, name_list)):
                # Compute adaptive threshold per image if requested
                sample_threshold = threshold
                if threshold_method != 'fixed':
                    pred_np_for_thresh = pred.squeeze().cpu().numpy()
                    if threshold_method == 'otsu':
                        sample_threshold = ThresholdOptimizer.otsu(pred_np_for_thresh)
                    elif threshold_method == 'adaptive-mean':
                        sample_threshold = ThresholdOptimizer.adaptive(pred_np_for_thresh, method='mean')
                    elif threshold_method == 'adaptive-median':
                        sample_threshold = ThresholdOptimizer.adaptive(pred_np_for_thresh, method='median')

                # Collect for threshold optimization if requested
                if collect_for_optimization:
                    pred_np = pred.squeeze().cpu().numpy()
                    gt_np = gt.squeeze().cpu().numpy()
                    all_preds.append(pred_np)
                    all_gts.append(gt_np)

                # Compute metrics for this sample
                sample_metrics = CODMetrics()
                sample_metrics.update(pred, gt, threshold=sample_threshold)
                sample_metrics_dict = sample_metrics.compute()

                # Update overall metrics
                metrics.update(pred, gt, threshold=sample_threshold)

            # DEBUG: Print first 3 samples
            if vis_count < 3:
                pred_np_debug = pred.squeeze().cpu().numpy()
                gt_np_debug = gt.squeeze().cpu().numpy()
                
                # Compute IoU manually for verification
                pred_bin = (pred_np_debug > 0.5).astype(np.float32)
                gt_bin = (gt_np_debug > 0.5).astype(np.float32)
                intersection = (pred_bin * gt_bin).sum()
                union = pred_bin.sum() + gt_bin.sum() - intersection
                manual_iou = intersection / (union + 1e-8)
                
                print(f"\n[DEBUG] Sample {name}:")
                print(f"  Pred range: [{pred_np_debug.min():.4f}, {pred_np_debug.max():.4f}]")
                print(f"  GT range: [{gt_np_debug.min():.4f}, {gt_np_debug.max():.4f}]")
                print(f"  Pred positive: {pred_bin.sum():.0f}, GT positive: {gt_bin.sum():.0f}")
                print(f"  Manual IoU: {manual_iou:.4f}")
                print(f"  Sample metrics: {sample_metrics_dict}")
                vis_count += 1

                # Save prediction if requested
                if output_dir is not None:
                    pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    pred_img = Image.fromarray(pred_np)
                    pred_img.save(output_dir / f"{name}.png")

                # Save visualizations if requested
                if save_visualizations and (num_vis_samples is None or vis_count < num_vis_samples):
                    vis_dir = output_dir.parent / 'visualizations' / dataset.root.name if output_dir else Path('visualizations') / dataset.root.name

                    # Get the original image for this sample
                    if use_tta or use_crf:
                        img_tensor = images.squeeze(0)
                    else:
                        # Find the index in the original batch
                        idx = name_list.index(name)
                        img_tensor = images[idx]

                    # Save comparison figure
                    fig = visualizer.create_comparison_figure(
                        img_tensor, pred, gt, sample_metrics_dict, name, sample_threshold
                    )
                    fig_path = vis_dir / 'figures'
                    fig_path.mkdir(parents=True, exist_ok=True)
                    fig.savefig(fig_path / f"{name}_comparison.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)

                    # Save individual outputs
                    individual_dir = vis_dir / 'individual' / name
                    visualizer.save_individual_outputs(
                        img_tensor, pred, gt, name, individual_dir, sample_threshold
                    )

                    vis_count += 1

    # Compute final metrics
    final_metrics = metrics.compute()

    if collect_for_optimization:
        return final_metrics, all_preds, all_gts
    else:
        return final_metrics


def save_results_json(results, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Results saved to: {output_path}")


def save_results_markdown(results, output_path, checkpoint_path, use_tta, optimize_threshold=False):
    """Save results to Markdown file with formatted tables."""
    with open(output_path, 'w') as f:
        f.write("# CamoXpert Evaluation Results\n\n")
        f.write(f"**Checkpoint**: `{checkpoint_path}`\n\n")
        f.write(f"**Test-Time Augmentation**: {'Enabled' if use_tta else 'Disabled'}\n\n")

        # Check if threshold optimization was used
        if optimize_threshold and 'average' in results and 'optimal_metrics' in results['average']:
            # Threshold optimization results
            f.write("## Threshold Optimization Results\n\n")

            # Default threshold results
            f.write(f"### Default Threshold Results\n\n")
            f.write("| Dataset | Threshold | S-measure | F-measure | E-measure | MAE ↓ | IoU |\n")
            f.write("|---------|-----------|-----------|-----------|-----------|-------|-----|\n")

            for dataset_name, data in results.items():
                if dataset_name != 'average':
                    metrics = data['default_metrics']
                    thr = data['default_threshold']
                    f.write(f"| {dataset_name:11} | {thr:.2f} | {metrics['S-measure']:.4f} | "
                           f"{metrics['F-measure']:.4f} | {metrics['E-measure']:.4f} | "
                           f"{metrics['MAE']:.4f} | {metrics['IoU']:.4f} |\n")

            if 'average' in results:
                avg = results['average']['default_metrics']
                thr = results['average']['default_threshold']
                f.write(f"| **Average** | {thr:.2f} | **{avg['S-measure']:.4f}** | "
                       f"**{avg['F-measure']:.4f}** | **{avg['E-measure']:.4f}** | "
                       f"**{avg['MAE']:.4f}** | **{avg['IoU']:.4f}** |\n")

            # Optimal threshold results
            f.write(f"\n### Optimal Threshold Results\n\n")
            f.write("| Dataset | Threshold (IoU) | Threshold (F1) | S-measure ⭐ | F-measure | E-measure | MAE ↓ | IoU | Improvement |\n")
            f.write("|---------|-----------------|----------------|-------------|-----------|-----------|-------|-----|-------------|\n")

            for dataset_name, data in results.items():
                if dataset_name != 'average':
                    metrics = data['optimal_metrics']
                    thr_iou = data['optimal_threshold_iou']
                    thr_f1 = data['optimal_threshold_f1']
                    imp_iou = data['improvement_iou']
                    imp_f1 = data['improvement_f1']
                    f.write(f"| {dataset_name:11} | {thr_iou:.2f} | {thr_f1:.2f} | "
                           f"{metrics['S-measure']:.4f} | {metrics['F-measure']:.4f} | "
                           f"{metrics['E-measure']:.4f} | {metrics['MAE']:.4f} | "
                           f"{metrics['IoU']:.4f} | IoU: +{imp_iou:.2f}%, F1: +{imp_f1:.2f}% |\n")

            if 'average' in results:
                avg = results['average']['optimal_metrics']
                thr_iou = results['average']['optimal_threshold_iou']
                thr_f1 = results['average']['optimal_threshold_f1']
                imp_iou = results['average']['improvement_iou']
                imp_f1 = results['average']['improvement_f1']
                f.write(f"| **Average** | {thr_iou:.2f} | {thr_f1:.2f} | "
                       f"**{avg['S-measure']:.4f}** | **{avg['F-measure']:.4f}** | "
                       f"**{avg['E-measure']:.4f}** | **{avg['MAE']:.4f}** | "
                       f"**{avg['IoU']:.4f}** | **IoU: +{imp_iou:.2f}%, F1: +{imp_f1:.2f}%** |\n")

        else:
            # Standard results (no threshold optimization)
            # Main metrics table
            f.write("## Primary Metrics\n\n")
            f.write("| Dataset | S-measure ⭐ | F-measure | E-measure | MAE ↓ | IoU |\n")
            f.write("|---------|-------------|-----------|-----------|-------|-----|\n")

            for dataset_name, metrics in results.items():
                if dataset_name != 'average':
                    f.write(f"| {dataset_name:11} | {metrics['S-measure']:.4f} | "
                           f"{metrics['F-measure']:.4f} | {metrics['E-measure']:.4f} | "
                           f"{metrics['MAE']:.4f} | {metrics['IoU']:.4f} |\n")

            # Average row
            if 'average' in results:
                avg = results['average']
                f.write(f"| **Average** | **{avg['S-measure']:.4f}** | "
                       f"**{avg['F-measure']:.4f}** | **{avg['E-measure']:.4f}** | "
                       f"**{avg['MAE']:.4f}** | **{avg['IoU']:.4f}** |\n")

        # Notes
        f.write("\n## Notes\n\n")
        f.write("- ⭐ S-measure is the primary metric for COD evaluation\n")
        f.write("- ↓ indicates lower is better\n")
        f.write("- All other metrics: higher is better\n")

    print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CamoXpert Evaluation')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-experts', type=int, default=None,
                       help='Number of experts in the model (auto-detected from checkpoint if not specified)')

    # Data arguments
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory containing test datasets (optional if using individual dataset paths)')
    parser.add_argument('--datasets', nargs='+',
                       default=['COD10K', 'CHAMELEON', 'CAMO', 'NC4K'],
                       help='Datasets to evaluate on (default: all)')
    parser.add_argument('--image-size', type=int, default=416,
                       help='Input image size (default: 352)')

    # Individual dataset paths (alternative to --data-root)
    parser.add_argument('--cod10k-img', type=str, default=None,
                       help='Path to COD10K test images directory')
    parser.add_argument('--cod10k-gt', type=str, default=None,
                       help='Path to COD10K test ground truth directory')
    parser.add_argument('--chameleon-img', type=str, default=None,
                       help='Path to CHAMELEON images directory')
    parser.add_argument('--chameleon-gt', type=str, default=None,
                       help='Path to CHAMELEON ground truth directory')
    parser.add_argument('--camo-img', type=str, default=None,
                       help='Path to CAMO test images directory')
    parser.add_argument('--camo-gt', type=str, default=None,
                       help='Path to CAMO test ground truth directory')
    parser.add_argument('--nc4k-img', type=str, default=None,
                       help='Path to NC4K images directory')
    parser.add_argument('--nc4k-gt', type=str, default=None,
                       help='Path to NC4K ground truth directory')

    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation (default: 1, use 8-16 for faster testing)')
    parser.add_argument('--use-dataparallel', action='store_true',
                       help='Enable DataParallel for multi-GPU inference (experimental, may cause errors)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: disable TTA, CRF, and visualizations for maximum speed')
    parser.add_argument('--tta', action='store_true',
                       help='Enable Test-Time Augmentation (multi-scale + flip)')
    parser.add_argument('--use-crf', action='store_true',
                       help='Enable CRF post-processing for boundary refinement')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary prediction (default: 0.5)')
    parser.add_argument('--threshold-method', type=str, default='fixed',
                       choices=['fixed', 'otsu', 'adaptive-mean', 'adaptive-median'],
                       help='Threshold selection method: fixed (default), otsu (per-image), adaptive-mean, adaptive-median')
    parser.add_argument('--optimize-threshold', action='store_true',
                       help='Find optimal threshold per dataset using grid search (overrides threshold-method)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save prediction masks to output directory')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save comprehensive visualizations (overlays, boundaries, error maps, etc.)')
    parser.add_argument('--num-vis-samples', type=int, default=None,
                       help='Number of samples to visualize per dataset (default: all)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./test_results',
                       help='Output directory for results (default: ./test_results)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Handle fast mode
    if args.fast:
        print("\n⚡ FAST MODE ENABLED - Disabling TTA, CRF, and visualizations for maximum speed")
        args.tta = False
        args.use_crf = False
        args.save_visualizations = False

    # Build dataset_paths dict from individual arguments
    dataset_paths = {}

    # Debug: Print all dataset-related arguments
    print("\n[DEBUG] Dataset path arguments:")
    print(f"  COD10K:    img={getattr(args, 'cod10k_img', 'NOT FOUND')}, gt={getattr(args, 'cod10k_gt', 'NOT FOUND')}")
    print(f"  CHAMELEON: img={getattr(args, 'chameleon_img', 'NOT FOUND')}, gt={getattr(args, 'chameleon_gt', 'NOT FOUND')}")
    print(f"  CAMO:      img={getattr(args, 'camo_img', 'NOT FOUND')}, gt={getattr(args, 'camo_gt', 'NOT FOUND')}")
    print(f"  NC4K:      img={getattr(args, 'nc4k_img', 'NOT FOUND')}, gt={getattr(args, 'nc4k_gt', 'NOT FOUND')}")
    print()

    dataset_config = {
        'COD10K': (args.cod10k_img, args.cod10k_gt),
        'CHAMELEON': (args.chameleon_img, args.chameleon_gt),
        'CAMO': (args.camo_img, args.camo_gt),
        'NC4K': (args.nc4k_img, args.nc4k_gt),
    }

    for dataset_name, (img_path, gt_path) in dataset_config.items():
        if img_path is not None and gt_path is not None:
            # Check if paths exist
            img_path_obj = Path(img_path)
            gt_path_obj = Path(gt_path)
            if img_path_obj.exists() and gt_path_obj.exists():
                dataset_paths[dataset_name] = {
                    'image_dir': img_path,
                    'gt_dir': gt_path
                }

    # Validate configuration
    if args.data_root is None and len(dataset_paths) == 0:
        raise ValueError("Must provide either --data-root OR individual dataset paths (e.g., --cod10k-img and --cod10k-gt)")

    # Filter datasets to only those available
    if len(dataset_paths) > 0:
        # Using individual paths - only evaluate datasets with provided paths
        available_datasets = [d for d in args.datasets if d in dataset_paths]
        if len(available_datasets) == 0:
            raise ValueError(f"No datasets available. Requested: {args.datasets}, Available: {list(dataset_paths.keys())}")
        args.datasets = available_datasets

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("CAMOEXPERT EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Batch size: {args.batch_size}")
    if torch.cuda.device_count() > 1 and args.use_dataparallel:
        print(f"GPUs: {torch.cuda.device_count()} (DataParallel enabled)")
    elif torch.cuda.device_count() > 1:
        print(f"GPUs: {torch.cuda.device_count()} (single GPU mode, use --use-dataparallel for multi-GPU)")
    else:
        print(f"GPUs: {torch.cuda.device_count()}")
    print(f"TTA: {'Enabled' if args.tta else 'Disabled'}")
    print(f"CRF Refinement: {'Enabled' if args.use_crf else 'Disabled'}")
    if args.use_crf:
        try:
            from utils.crf_refiner import HAS_CRF
            if HAS_CRF:
                print("  Using Dense CRF (pydensecrf)")
            else:
                print("  Using morphological refinement (pydensecrf not available)")
        except:
            print("  Using morphological refinement")
    print(f"Threshold Optimization: {'Enabled' if args.optimize_threshold else 'Disabled'}")
    if args.optimize_threshold:
        print(f"  Will find optimal threshold via grid search")
    else:
        print(f"  Threshold method: {args.threshold_method}")
        if args.threshold_method == 'fixed':
            print(f"  Binary threshold: {args.threshold}")
        else:
            print(f"  Per-image adaptive thresholding")
    print(f"Visualizations: {'Enabled' if args.save_visualizations else 'Disabled'}")
    if args.save_visualizations:
        print(f"  Vis samples per dataset: {args.num_vis_samples if args.num_vis_samples else 'All'}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    # Load model
    model = load_checkpoint(args.checkpoint, num_experts=args.num_experts, device=device,
                           use_dataparallel=args.use_dataparallel)

    # Evaluate on each dataset
    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        try:
            # Create dataset using individual paths or data_root
            if dataset_name in dataset_paths:
                # Use individual paths
                paths = dataset_paths[dataset_name]
                print(f"  Using custom paths:")
                print(f"    Images: {paths['image_dir']}")
                print(f"    GT:     {paths['gt_dir']}")
                dataset = CODTestDataset(
                    image_size=args.image_size,
                    image_dir=paths['image_dir'],
                    gt_dir=paths['gt_dir']
                )
            elif args.data_root is not None:
                # Use data_root directory structure
                dataset_path = Path(args.data_root) / dataset_name

                # Handle different naming conventions
                if not dataset_path.exists():
                    # Try with -v3 suffix (e.g., COD10K-v3)
                    dataset_path = Path(args.data_root) / f"{dataset_name}-v3"

                if not dataset_path.exists():
                    print(f"⚠️  Dataset not found: {dataset_path}")
                    print(f"    Skipping {dataset_name}...")
                    continue

                # For COD10K, CAMO, NC4K, use Test subdirectory
                if dataset_name in ['COD10K', 'CAMO', 'NC4K']:
                    test_path = dataset_path / 'Test'
                    if test_path.exists():
                        dataset_path = test_path

                dataset = CODTestDataset(dataset_path, image_size=args.image_size)
            else:
                print(f"⚠️  No path configured for {dataset_name}")
                print(f"    Skipping...")
                continue

            # Evaluate
            pred_dir = output_dir / 'predictions' / dataset_name if args.save_predictions else None

            if args.optimize_threshold:
                # Collect predictions for threshold optimization
                # Note: When optimizing threshold globally, we use fixed threshold for initial evaluation
                metrics, all_preds, all_gts = evaluate_dataset(
                    model, dataset, device,
                    use_tta=args.tta,
                    use_crf=args.use_crf,
                    threshold=args.threshold,
                    threshold_method='fixed',  # Use fixed for initial pass
                    output_dir=pred_dir,
                    save_visualizations=args.save_visualizations,
                    num_vis_samples=args.num_vis_samples,
                    collect_for_optimization=True,
                    batch_size=args.batch_size
                )

                # Print results with default threshold
                print(f"\n  Results for {dataset_name} (threshold={args.threshold:.2f}):")
                print(f"    S-measure: {metrics['S-measure']:.4f}")
                print(f"    F-measure: {metrics['F-measure']:.4f}")
                print(f"    E-measure: {metrics['E-measure']:.4f}")
                print(f"    MAE:       {metrics['MAE']:.4f}")
                print(f"    IoU:       {metrics['IoU']:.4f}")

                # Optimize threshold
                print(f"\n  Optimizing threshold for {dataset_name}...")
                best_thr_iou, best_iou, iou_scores = ThresholdOptimizer.grid_search(
                    all_preds, all_gts, metric='iou'
                )
                best_thr_f1, best_f1, f1_scores = ThresholdOptimizer.grid_search(
                    all_preds, all_gts, metric='f1'
                )

                print(f"    Best threshold (IoU): {best_thr_iou:.2f} -> IoU={best_iou:.4f}")
                print(f"    Best threshold (F1):  {best_thr_f1:.2f} -> F1={best_f1:.4f}")

                # Re-evaluate with optimal threshold
                print(f"\n  Re-evaluating with optimal threshold ({best_thr_iou:.2f})...")
                metrics_optimal = evaluate_dataset(
                    model, dataset, device,
                    use_tta=args.tta,
                    use_crf=args.use_crf,
                    threshold=best_thr_iou,
                    threshold_method='fixed',  # Use optimized fixed threshold
                    output_dir=None,  # Don't save predictions again
                    save_visualizations=False,  # Don't save visualizations again
                    collect_for_optimization=False,
                    batch_size=args.batch_size
                )

                # Store both results
                all_results[dataset_name] = {
                    'default_threshold': args.threshold,
                    'default_metrics': metrics,
                    'optimal_threshold_iou': best_thr_iou,
                    'optimal_threshold_f1': best_thr_f1,
                    'optimal_metrics': metrics_optimal,
                    'improvement_iou': (best_iou - metrics['IoU']) * 100,  # % improvement
                    'improvement_f1': (best_f1 - metrics['F-measure']) * 100
                }

                # Print optimal results
                print(f"\n  Results with optimal threshold ({best_thr_iou:.2f}):")
                print(f"    S-measure: {metrics_optimal['S-measure']:.4f} ⭐")
                print(f"    F-measure: {metrics_optimal['F-measure']:.4f}")
                print(f"    E-measure: {metrics_optimal['E-measure']:.4f}")
                print(f"    MAE:       {metrics_optimal['MAE']:.4f}")
                print(f"    IoU:       {metrics_optimal['IoU']:.4f}")
                print(f"\n  Improvement:")
                print(f"    IoU: +{all_results[dataset_name]['improvement_iou']:.2f}%")
                print(f"    F1:  +{all_results[dataset_name]['improvement_f1']:.2f}%")

            else:
                # Standard evaluation
                metrics = evaluate_dataset(
                    model, dataset, device,
                    use_tta=args.tta,
                    use_crf=args.use_crf,
                    threshold=args.threshold,
                    threshold_method=args.threshold_method,
                    output_dir=pred_dir,
                    save_visualizations=args.save_visualizations,
                    num_vis_samples=args.num_vis_samples,
                    batch_size=args.batch_size
                )

                # Store results
                all_results[dataset_name] = metrics

                # Print results
                print(f"\n  Results for {dataset_name}:")
                print(f"    S-measure: {metrics['S-measure']:.4f} ⭐")
                print(f"    F-measure: {metrics['F-measure']:.4f}")
                print(f"    E-measure: {metrics['E-measure']:.4f}")
                print(f"    MAE:       {metrics['MAE']:.4f}")
                print(f"    IoU:       {metrics['IoU']:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            continue

    # Compute average metrics
    if len(all_results) > 0:
        if args.optimize_threshold:
            # Average both default and optimal metrics
            avg_default_metrics = {}
            avg_optimal_metrics = {}
            first_result = all_results[list(all_results.keys())[0]]

            for metric_name in first_result['default_metrics'].keys():
                avg_default_metrics[metric_name] = np.mean([
                    results['default_metrics'][metric_name] for results in all_results.values()
                ])
                avg_optimal_metrics[metric_name] = np.mean([
                    results['optimal_metrics'][metric_name] for results in all_results.values()
                ])

            avg_thr_iou = np.mean([r['optimal_threshold_iou'] for r in all_results.values()])
            avg_thr_f1 = np.mean([r['optimal_threshold_f1'] for r in all_results.values()])
            avg_improvement_iou = np.mean([r['improvement_iou'] for r in all_results.values()])
            avg_improvement_f1 = np.mean([r['improvement_f1'] for r in all_results.values()])

            all_results['average'] = {
                'default_threshold': args.threshold,
                'default_metrics': avg_default_metrics,
                'optimal_threshold_iou': avg_thr_iou,
                'optimal_threshold_f1': avg_thr_f1,
                'optimal_metrics': avg_optimal_metrics,
                'improvement_iou': avg_improvement_iou,
                'improvement_f1': avg_improvement_f1
            }

            print(f"\n{'='*70}")
            print("AVERAGE METRICS")
            print(f"{'='*70}")
            print(f"\nDefault threshold ({args.threshold:.2f}):")
            print(f"  S-measure: {avg_default_metrics['S-measure']:.4f}")
            print(f"  F-measure: {avg_default_metrics['F-measure']:.4f}")
            print(f"  E-measure: {avg_default_metrics['E-measure']:.4f}")
            print(f"  MAE:       {avg_default_metrics['MAE']:.4f}")
            print(f"  IoU:       {avg_default_metrics['IoU']:.4f}")

            print(f"\nOptimal threshold (avg: {avg_thr_iou:.2f} for IoU, {avg_thr_f1:.2f} for F1):")
            print(f"  S-measure: {avg_optimal_metrics['S-measure']:.4f} ⭐")
            print(f"  F-measure: {avg_optimal_metrics['F-measure']:.4f}")
            print(f"  E-measure: {avg_optimal_metrics['E-measure']:.4f}")
            print(f"  MAE:       {avg_optimal_metrics['MAE']:.4f}")
            print(f"  IoU:       {avg_optimal_metrics['IoU']:.4f}")

            print(f"\nAverage improvement:")
            print(f"  IoU: +{avg_improvement_iou:.2f}%")
            print(f"  F1:  +{avg_improvement_f1:.2f}%")
            print(f"{'='*70}\n")

        else:
            # Standard average
            avg_metrics = {}
            for metric_name in all_results[list(all_results.keys())[0]].keys():
                avg_metrics[metric_name] = np.mean([
                    results[metric_name] for results in all_results.values()
                ])
            all_results['average'] = avg_metrics

            print(f"\n{'='*70}")
            print("AVERAGE METRICS")
            print(f"{'='*70}")
            print(f"  S-measure: {avg_metrics['S-measure']:.4f} ⭐")
            print(f"  F-measure: {avg_metrics['F-measure']:.4f}")
            print(f"  E-measure: {avg_metrics['E-measure']:.4f}")
            print(f"  MAE:       {avg_metrics['MAE']:.4f}")
            print(f"  IoU:       {avg_metrics['IoU']:.4f}")
            print(f"{'='*70}\n")

    # Save results
    save_results_json(all_results, output_dir / 'results.json')
    save_results_markdown(all_results, output_dir / 'results.md', args.checkpoint, args.tta, args.optimize_threshold)

    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}\n")


if __name__ == '__main__':
    main()
