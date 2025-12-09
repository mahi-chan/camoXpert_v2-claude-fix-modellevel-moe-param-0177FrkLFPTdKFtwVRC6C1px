import torch
import torch.nn.functional as F


class TestTimeAugmentation:
    """Test-Time Augmentation for improved inference"""

    def __init__(self, model, scales=[0.75, 1.0, 1.25], flips=[False, True], rotations=[0]):
        self.model = model
        self.scales = scales
        self.flips = flips
        self.rotations = rotations

    @torch.no_grad()
    def predict(self, image):
        """
        Apply TTA with multi-scale, flip, and rotation

        Args:
            image: Input tensor [B, C, H, W]

        Returns:
            Averaged prediction [B, 1, H, W]
        """
        B, C, H, W = image.shape
        predictions = []

        for scale in self.scales:
            # Scale
            if scale != 1.0:
                h_s, w_s = int(H * scale), int(W * scale)
                img_scaled = F.interpolate(image, size=(h_s, w_s),
                                           mode='bilinear', align_corners=False)
            else:
                img_scaled = image

            for flip in self.flips:
                for rotation in self.rotations:
                    img_aug = img_scaled

                    # Flip
                    if flip:
                        img_aug = torch.flip(img_aug, dims=[3])

                    # Rotate
                    if rotation != 0:
                        img_aug = torch.rot90(img_aug, k=rotation // 90, dims=[2, 3])

                    # Predict
                    pred, _ = self.model(img_aug)

                    # Undo rotation
                    if rotation != 0:
                        pred = torch.rot90(pred, k=-rotation // 90, dims=[2, 3])

                    # Undo flip
                    if flip:
                        pred = torch.flip(pred, dims=[3])

                    # Undo scale
                    if scale != 1.0:
                        pred = F.interpolate(pred, size=(H, W),
                                             mode='bilinear', align_corners=False)

                    predictions.append(pred)

        # Average all predictions
        return torch.stack(predictions).mean(dim=0)