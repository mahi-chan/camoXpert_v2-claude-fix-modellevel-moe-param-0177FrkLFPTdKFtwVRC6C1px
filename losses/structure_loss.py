import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    """
    Structure-aware loss focusing on boundary preservation.

    Combines region similarity and edge matching for better
    camouflage boundary detection.
    """

    def __init__(self):
        super().__init__()

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_edges(self, x):
        """Compute edge map using Sobel filters"""
        edges_x = F.conv2d(x, self.sobel_x, padding=1)
        edges_y = F.conv2d(x, self.sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-8)
        return edges

    def forward(self, pred, target):
        # Compute edges
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)

        # Edge loss
        edge_loss = F.mse_loss(pred_edges, target_edges)

        # Region similarity (SSIM-like)
        pred_mean = pred.mean(dim=(2, 3), keepdim=True)
        target_mean = target.mean(dim=(2, 3), keepdim=True)

        pred_std = pred.std(dim=(2, 3), keepdim=True)
        target_std = target.std(dim=(2, 3), keepdim=True)

        covariance = ((pred - pred_mean) * (target - target_mean)).mean(dim=(2, 3))

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * pred_mean * target_mean + c1) * (2 * covariance + c2)) / \
               ((pred_mean ** 2 + target_mean ** 2 + c1) * (pred_std ** 2 + target_std ** 2 + c2))

        region_loss = 1 - ssim.mean()

        return edge_loss + region_loss