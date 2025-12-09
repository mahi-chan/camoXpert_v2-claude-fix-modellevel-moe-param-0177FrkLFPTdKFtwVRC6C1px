import torch
import torch.nn as nn
import torch.nn.functional as F


class CamoXpertLoss(nn.Module):
    def __init__(self, bce_weight=1.0, iou_weight=1.0, ssim_weight=0.5, aux_weight=0.01):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.ssim_weight = ssim_weight
        self.aux_weight = aux_weight
        self.bce = nn.BCELoss()

    def iou_loss(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou

    def ssim_loss(self, pred, target, window_size=11):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

        sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return 1 - ssim_map.mean()

    def forward(self, pred, target, aux_loss=0):
        bce_loss = self.bce(pred, target)
        iou_loss = self.iou_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)

        total_loss = (self.bce_weight * bce_loss +
                      self.iou_weight * iou_loss +
                      self.ssim_weight * ssim_loss +
                      self.aux_weight * aux_loss)

        return total_loss, {
            'bce': bce_loss.item(),
            'iou': iou_loss.item(),
            'ssim': ssim_loss.item(),
            'aux': aux_loss if isinstance(aux_loss, float) else aux_loss.item()
        }