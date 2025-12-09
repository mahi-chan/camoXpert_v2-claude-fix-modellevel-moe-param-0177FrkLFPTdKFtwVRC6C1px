"""
Test-Time Augmentation for CamoXpert
Multi-scale + flip augmentation for improved predictions
"""

import torch
import torch.nn.functional as F


class TTAPredictor:
    """
    Test-Time Augmentation Predictor

    Combines predictions from:
    - Multiple scales (0.75x, 1.0x, 1.25x)
    - Horizontal flips
    - Optional vertical flips

    Usage:
        tta = TTAPredictor(model)
        pred = tta.predict(image)  # Returns averaged probability map
    """

    def __init__(self, model, scales=[0.75, 1.0, 1.25], flip_horizontal=True, flip_vertical=False):
        """
        Initialize TTA predictor.

        Args:
            model: Trained model for inference
            scales: List of scale factors (default: [0.75, 1.0, 1.25])
            flip_horizontal: Whether to use horizontal flip (default: True)
            flip_vertical: Whether to use vertical flip (default: False)
        """
        self.model = model
        self.scales = scales
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        """
        Perform TTA prediction.

        Args:
            image: [B, 3, H, W] normalized input tensor

        Returns:
            pred: [B, 1, H, W] averaged probability map (0-1)
        """
        B, C, H, W = image.shape
        device = image.device
        preds = []

        for scale in self.scales:
            # Resize
            size = (int(H * scale), int(W * scale))
            img_scaled = F.interpolate(image, size=size, mode='bilinear', align_corners=False)

            # Original
            pred = self._forward(img_scaled)
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
            preds.append(pred)

            # Horizontal flip
            if self.flip_horizontal:
                img_flip = torch.flip(img_scaled, dims=[3])
                pred_flip = self._forward(img_flip)
                pred_flip = torch.flip(pred_flip, dims=[3])
                pred_flip = F.interpolate(pred_flip, size=(H, W), mode='bilinear', align_corners=False)
                preds.append(pred_flip)

            # Vertical flip
            if self.flip_vertical:
                img_flip_v = torch.flip(img_scaled, dims=[2])
                pred_flip_v = self._forward(img_flip_v)
                pred_flip_v = torch.flip(pred_flip_v, dims=[2])
                pred_flip_v = F.interpolate(pred_flip_v, size=(H, W), mode='bilinear', align_corners=False)
                preds.append(pred_flip_v)

        # Average all predictions
        return torch.stack(preds).mean(dim=0)

    def _forward(self, x):
        """
        Forward pass with sigmoid.

        Args:
            x: Input tensor

        Returns:
            Prediction with sigmoid applied [B, 1, H, W]
        """
        output = self.model(x)
        pred = output['pred'] if isinstance(output, dict) else output
        return torch.sigmoid(pred)

    def predict_with_uncertainty(self, image):
        """
        Predict with uncertainty estimation.

        Returns mean prediction and standard deviation across augmentations
        as a measure of model uncertainty.

        Args:
            image: [B, 3, H, W] normalized input tensor

        Returns:
            mean_pred: [B, 1, H, W] averaged probability map
            std_pred: [B, 1, H, W] standard deviation across augmentations
        """
        B, C, H, W = image.shape
        preds = []

        for scale in self.scales:
            size = (int(H * scale), int(W * scale))
            img_scaled = F.interpolate(image, size=size, mode='bilinear', align_corners=False)

            # Original
            pred = self._forward(img_scaled)
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
            preds.append(pred)

            # Horizontal flip
            if self.flip_horizontal:
                img_flip = torch.flip(img_scaled, dims=[3])
                pred_flip = self._forward(img_flip)
                pred_flip = torch.flip(pred_flip, dims=[3])
                pred_flip = F.interpolate(pred_flip, size=(H, W), mode='bilinear', align_corners=False)
                preds.append(pred_flip)

            # Vertical flip
            if self.flip_vertical:
                img_flip_v = torch.flip(img_scaled, dims=[2])
                pred_flip_v = self._forward(img_flip_v)
                pred_flip_v = torch.flip(pred_flip_v, dims=[2])
                pred_flip_v = F.interpolate(pred_flip_v, size=(H, W), mode='bilinear', align_corners=False)
                preds.append(pred_flip_v)

        # Stack predictions [N, B, 1, H, W]
        preds_stacked = torch.stack(preds)

        # Compute mean and std
        mean_pred = preds_stacked.mean(dim=0)
        std_pred = preds_stacked.std(dim=0)

        return mean_pred, std_pred

    def __call__(self, image):
        """Allow class instance to be called like a function."""
        return self.predict(image)

    def get_num_augmentations(self):
        """
        Get total number of augmentations used.

        Returns:
            Number of forward passes per image
        """
        num_augs = len(self.scales)  # Base scales

        if self.flip_horizontal:
            num_augs += len(self.scales)  # Horizontal flips

        if self.flip_vertical:
            num_augs += len(self.scales)  # Vertical flips

        return num_augs


class FastTTAPredictor(TTAPredictor):
    """
    Fast TTA with reduced augmentations for speed.

    Uses only 1.0x scale with horizontal flip.
    Suitable for quick validation during training.
    """

    def __init__(self, model):
        """Initialize fast TTA with minimal augmentations."""
        super().__init__(
            model=model,
            scales=[1.0],
            flip_horizontal=True,
            flip_vertical=False
        )


class AggressiveTTAPredictor(TTAPredictor):
    """
    Aggressive TTA with maximum augmentations.

    Uses 5 scales + horizontal + vertical flips.
    Slower but more robust predictions.
    """

    def __init__(self, model):
        """Initialize aggressive TTA with maximum augmentations."""
        super().__init__(
            model=model,
            scales=[0.5, 0.75, 1.0, 1.25, 1.5],
            flip_horizontal=True,
            flip_vertical=True
        )


if __name__ == '__main__':
    """Test TTA predictor."""
    print("="*70)
    print("Testing TTA Predictor")
    print("="*70)

    # Create dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.rand(x.shape[0], 1, x.shape[2], x.shape[3])

    model = DummyModel()

    # Test standard TTA
    print("\n1. Standard TTA:")
    tta = TTAPredictor(model)
    print(f"   Number of augmentations: {tta.get_num_augmentations()}")

    image = torch.randn(2, 3, 352, 352)
    pred = tta.predict(image)
    print(f"   Input shape: {image.shape}")
    print(f"   Output shape: {pred.shape}")
    print(f"   Output range: [{pred.min():.3f}, {pred.max():.3f}]")

    # Test with uncertainty
    print("\n2. TTA with uncertainty:")
    mean_pred, std_pred = tta.predict_with_uncertainty(image)
    print(f"   Mean shape: {mean_pred.shape}")
    print(f"   Std shape: {std_pred.shape}")
    print(f"   Avg uncertainty (std): {std_pred.mean():.4f}")

    # Test fast TTA
    print("\n3. Fast TTA:")
    fast_tta = FastTTAPredictor(model)
    print(f"   Number of augmentations: {fast_tta.get_num_augmentations()}")
    pred_fast = fast_tta.predict(image)
    print(f"   Output shape: {pred_fast.shape}")

    # Test aggressive TTA
    print("\n4. Aggressive TTA:")
    agg_tta = AggressiveTTAPredictor(model)
    print(f"   Number of augmentations: {agg_tta.get_num_augmentations()}")
    pred_agg = agg_tta.predict(image)
    print(f"   Output shape: {pred_agg.shape}")

    # Callable test
    print("\n5. Callable test:")
    pred_callable = tta(image)
    print(f"   Using __call__: {pred_callable.shape}")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)

    print("\nUsage Examples:")
    print("-" * 70)
    print("# Standard TTA (default)")
    print("tta = TTAPredictor(model)")
    print("pred = tta.predict(image)")
    print()
    print("# Fast TTA (quick validation)")
    print("fast_tta = FastTTAPredictor(model)")
    print("pred = fast_tta(image)")
    print()
    print("# Aggressive TTA (maximum quality)")
    print("agg_tta = AggressiveTTAPredictor(model)")
    print("pred = agg_tta.predict(image)")
    print()
    print("# With uncertainty estimation")
    print("mean, std = tta.predict_with_uncertainty(image)")
    print("uncertainty_mask = std > 0.1  # High uncertainty regions")
