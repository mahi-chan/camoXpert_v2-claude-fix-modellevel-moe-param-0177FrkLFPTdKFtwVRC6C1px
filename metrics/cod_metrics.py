import torch
import numpy as np


class CODMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics"""
        self.total_metrics = {}
        self.num_batches = 0

    def update(self, pred, target, threshold=0.5):
        """Accumulate metrics from a batch"""
        batch_metrics = self.compute_all(pred, target, threshold)

        if self.num_batches == 0:
            self.total_metrics = {k: v for k, v in batch_metrics.items()}
        else:
            for k, v in batch_metrics.items():
                self.total_metrics[k] += v

        self.num_batches += 1

    def compute(self):
        """Compute average metrics across all batches"""
        if self.num_batches == 0:
            return {}
        return {k: v / self.num_batches for k, v in self.total_metrics.items()}

    @torch.no_grad()
    def pixel_accuracy(self, pred, target, threshold=0.5):
        """Calculate pixel-wise accuracy"""
        pred = (pred > threshold).float().detach()
        target = target.detach()
        correct = (pred == target).float().sum()
        total = target.numel()
        return (correct / total).item()

    @torch.no_grad()
    def precision(self, pred, target, threshold=0.5):
        """Calculate precision (positive predictive value)"""
        pred = (pred > threshold).float().detach()
        target = target.detach()
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        return (tp / (tp + fp + 1e-6)).item()

    @torch.no_grad()
    def recall(self, pred, target, threshold=0.5):
        """Calculate recall (sensitivity)"""
        pred = (pred > threshold).float().detach()
        target = target.detach()
        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        return (tp / (tp + fn + 1e-6)).item()

    @torch.no_grad()
    def dice_score(self, pred, target, threshold=0.5):
        """Calculate Dice Score (F1 Score)"""
        pred = (pred > threshold).float().detach()
        target = target.detach()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return ((2.0 * intersection + 1e-6) / (union + 1e-6)).item()

    @torch.no_grad()
    def specificity(self, pred, target, threshold=0.5):
        """Calculate specificity (true negative rate)"""
        pred = (pred > threshold).float().detach()
        target = target.detach()
        tn = ((1 - pred) * (1 - target)).sum()
        fp = (pred * (1 - target)).sum()
        return (tn / (tn + fp + 1e-6)).item()

    @torch.no_grad()
    def mae(self, pred, target):
        """Mean Absolute Error"""
        return torch.abs(pred.detach() - target.detach()).mean().item()

    @torch.no_grad()
    def iou(self, pred, target, threshold=0.5):
        """Intersection over Union"""
        pred = (pred > threshold).float().detach()
        target = target.detach()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return ((intersection + 1e-6) / (union + 1e-6)).item()

    @torch.no_grad()
    def f_measure(self, pred, target, threshold=0.5, beta=0.3):
        """F-measure (weighted F-score)"""
        pred = (pred > threshold).float().detach()
        target = target.detach()
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        return (((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall + 1e-6)).item()

    @torch.no_grad()
    def s_measure(self, pred, target, alpha=0.5):
        """Structure Measure"""
        pred = pred.detach()
        target = target.detach()
        y = target.mean()
        if y == 0:
            return (1.0 - pred.mean()).item()
        elif y == 1:
            return pred.mean().item()
        else:
            Q = alpha * self._s_object(pred, target) + (1 - alpha) * self._s_region(pred, target)
            return Q.item() if isinstance(Q, torch.Tensor) else Q

    def _s_object(self, pred, target):
        pred_fg = pred * target
        pred_bg = (1 - pred) * (1 - target)
        O_fg = self._object_score(pred_fg, target)
        O_bg = self._object_score(pred_bg, 1 - target)
        u = target.mean()
        return u * O_fg + (1 - u) * O_bg

    def _object_score(self, pred, target):
        pred_mean = pred.sum() / (target.sum() + 1e-8)
        sigma = ((pred - pred_mean) ** 2).sum() / (target.sum() + 1e-8)
        return 2.0 * pred_mean / (pred_mean ** 2 + 1.0 + sigma + 1e-8)

    def _s_region(self, pred, target):
        X, Y = self._centroid(target)
        pred1, pred2, pred3, pred4, w1, w2, w3, w4 = self._divide_with_xy(pred, target, X, Y)
        target1, target2, target3, target4, _, _, _, _ = self._divide_with_xy(target, target, X, Y)
        Q1, Q2, Q3, Q4 = self._ssim(pred1, target1), self._ssim(pred2, target2), self._ssim(pred3, target3), self._ssim(
            pred4, target4)
        return w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    def _centroid(self, target):
        rows = torch.arange(target.shape[2], dtype=torch.float32, device=target.device)
        cols = torch.arange(target.shape[3], dtype=torch.float32, device=target.device)
        total = target.sum() + 1e-8
        X = (target.sum(dim=2) * cols.view(1, 1, -1)).sum() / total
        Y = (target.sum(dim=3) * rows.view(1, 1, -1)).sum() / total
        return max(1, min(int(X.item()), target.shape[3] - 1)), max(1, min(int(Y.item()), target.shape[2] - 1))

    def _divide_with_xy(self, pred, target, X, Y):
        h, w = pred.shape[2], pred.shape[3]
        pred1, pred2, pred3, pred4 = pred[:, :, :Y, :X], pred[:, :, :Y, X:], pred[:, :, Y:, :X], pred[:, :, Y:, X:]
        target1, target2, target3, target4 = target[:, :, :Y, :X], target[:, :, :Y, X:], target[:, :, Y:, :X], target[:,
                                                                                                               :, Y:,
                                                                                                               X:]
        w1 = X * Y / (h * w + 1e-8)
        w2 = (w - X) * Y / (h * w + 1e-8)
        w3 = X * (h - Y) / (h * w + 1e-8)
        w4 = (w - X) * (h - Y) / (h * w + 1e-8)
        return pred1, pred2, pred3, pred4, w1, w2, w3, w4

    def _ssim(self, pred, target):
        h, w = pred.shape[2], pred.shape[3]
        if h < 2 or w < 2:
            return torch.tensor(0.0, device=pred.device)
        N = h * w
        x, y = pred.mean(), target.mean()
        sigma_x2 = ((pred - x) ** 2).sum() / (N - 1 + 1e-8)
        sigma_y2 = ((target - y) ** 2).sum() / (N - 1 + 1e-8)
        sigma_xy = ((pred - x) * (target - y)).sum() / (N - 1 + 1e-8)
        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)
        return alpha / (beta + 1e-8) if alpha != 0 else (1.0 if beta == 0 else 0.0)

    @torch.no_grad()
    def e_measure(self, pred, target):
        """Enhanced-alignment Measure"""
        pred, target = pred.detach(), target.detach()
        if target.sum() == 0:
            return (1 - pred).mean().item()
        enhanced_matrix = self._cal_enhanced_matrix(pred, target)
        return enhanced_matrix.mean().item()

    def _cal_enhanced_matrix(self, pred, target):
        enhanced_matrix = torch.zeros_like(pred)
        target_fg, target_bg = (target == 1).float(), (target == 0).float()
        if target_fg.sum() > 0:
            enhanced_matrix += ((pred - pred.mean()) ** 2) * target_fg / (target_fg.sum() + 1e-8)
        if target_bg.sum() > 0:
            enhanced_matrix += ((pred - pred.mean()) ** 2) * target_bg / (target_bg.sum() + 1e-8)
        return 2 * enhanced_matrix

    def compute_all(self, pred, target, threshold=0.5):
        """Compute all metrics including accuracy metrics"""
        return {
            # Standard metrics
            'MAE': self.mae(pred, target),
            'IoU': self.iou(pred, target, threshold),
            'F-measure': self.f_measure(pred, target, threshold),
            'S-measure': self.s_measure(pred, target),
            'E-measure': self.e_measure(pred, target),

            # Accuracy metrics
            'Pixel_Accuracy': self.pixel_accuracy(pred, target, threshold),
            'Precision': self.precision(pred, target, threshold),
            'Recall': self.recall(pred, target, threshold),
            'Dice_Score': self.dice_score(pred, target, threshold),
            'Specificity': self.specificity(pred, target, threshold)
        }