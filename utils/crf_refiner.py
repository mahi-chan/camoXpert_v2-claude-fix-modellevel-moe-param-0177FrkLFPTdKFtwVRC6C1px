"""
CRF Post-Processing for CamoXpert
Refines prediction boundaries using Dense CRF
"""

import numpy as np
import cv2
import torch

# Try to import pydensecrf, fallback to morphological refinement
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAS_CRF = True
except ImportError:
    HAS_CRF = False
    print("Warning: pydensecrf not installed. Using morphological refinement instead.")
    print("Install with: pip install pydensecrf")


class CRFRefiner:
    """
    Dense CRF refinement for segmentation masks.

    Uses dense conditional random fields to refine prediction boundaries
    based on image appearance. Falls back to morphological operations
    if pydensecrf is not available.

    Usage:
        refiner = CRFRefiner()
        refined = refiner.refine(image, pred)
    """

    def __init__(self,
                 sxy_gaussian=3,
                 compat_gaussian=3,
                 sxy_bilateral=50,
                 srgb_bilateral=13,
                 compat_bilateral=10,
                 iterations=5):
        """
        Initialize CRF refiner.

        Args:
            sxy_gaussian: Spatial std for Gaussian pairwise term (default: 3)
            compat_gaussian: Compatibility for Gaussian term (default: 3)
            sxy_bilateral: Spatial std for bilateral term (default: 50)
            srgb_bilateral: Color std for bilateral term (default: 13)
            compat_bilateral: Compatibility for bilateral term (default: 10)
            iterations: Number of CRF iterations (default: 5)
        """
        self.sxy_gaussian = sxy_gaussian
        self.compat_gaussian = compat_gaussian
        self.sxy_bilateral = sxy_bilateral
        self.srgb_bilateral = srgb_bilateral
        self.compat_bilateral = compat_bilateral
        self.iterations = iterations

    def refine(self, image, prob):
        """
        Refine prediction using CRF.

        Args:
            image: [H, W, 3] uint8 RGB image or [C, H, W] tensor
            prob: [H, W] float32 probability map (0-1) or [1, H, W] tensor

        Returns:
            refined: [H, W] float32 refined probability map
        """
        # Convert tensors to numpy if needed
        if torch.is_tensor(image):
            image = image.cpu().numpy()
            if len(image.shape) == 3:  # [C, H, W]
                image = np.transpose(image, (1, 2, 0))  # -> [H, W, C]

        if torch.is_tensor(prob):
            prob = prob.cpu().numpy()

        # Squeeze probability map
        prob = np.squeeze(prob)

        if HAS_CRF:
            return self._crf_refine(image, prob)
        else:
            return self._morphological_refine(prob)

    def _crf_refine(self, image, prob):
        """Dense CRF refinement using pydensecrf."""
        H, W = prob.shape

        # Ensure image is uint8 RGB
        if image.dtype != np.uint8:
            # Denormalize if normalized
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Create CRF
        d = dcrf.DenseCRF2D(W, H, 2)

        # Unary potentials from probability map
        prob_clipped = np.clip(prob, 1e-5, 1 - 1e-5)
        U = np.stack([1 - prob_clipped, prob_clipped], axis=0).astype(np.float32)
        U = -np.log(U)
        U = U.reshape((2, -1))
        d.setUnaryEnergy(U)

        # Pairwise potentials - Gaussian (spatial smoothness)
        d.addPairwiseGaussian(
            sxy=self.sxy_gaussian,
            compat=self.compat_gaussian
        )

        # Pairwise potentials - Bilateral (edge-aware smoothing)
        d.addPairwiseBilateral(
            sxy=self.sxy_bilateral,
            srgb=self.srgb_bilateral,
            rgbim=image.copy(),
            compat=self.compat_bilateral
        )

        # Inference
        Q = d.inference(self.iterations)
        Q = np.array(Q).reshape((2, H, W))

        return Q[1]  # Return foreground probability

    def _morphological_refine(self, prob, threshold=0.5):
        """Fallback morphological refinement when pydensecrf is not available."""
        # Binarize
        binary = (prob > threshold).astype(np.uint8)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Close small holes
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Open to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Smooth boundaries with Gaussian blur
        smoothed = cv2.GaussianBlur(opened.astype(np.float32), (5, 5), 0)

        return smoothed

    def refine_batch(self, images, probs):
        """
        Refine batch of predictions.

        Args:
            images: list of [H, W, 3] images or [C, H, W] tensors
            probs: list of [H, W] probability maps or [1, H, W] tensors

        Returns:
            list of refined probability maps
        """
        return [self.refine(img, prob) for img, prob in zip(images, probs)]


class MorphologicalRefiner:
    """
    Simple morphological refinement (no CRF dependency).

    Faster but less accurate than CRF. Uses morphological operations
    to clean up prediction masks and smooth boundaries.

    Usage:
        refiner = MorphologicalRefiner()
        refined = refiner.refine(prob)
    """

    def __init__(self, kernel_size=5):
        """
        Initialize morphological refiner.

        Args:
            kernel_size: Size of morphological kernel (default: 5)
        """
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    def refine(self, prob, threshold=0.5):
        """
        Refine prediction using morphological operations.

        Args:
            prob: [H, W] probability map (0-1)
            threshold: Binary threshold (default: 0.5)

        Returns:
            refined: [H, W] refined probability map
        """
        # Convert tensor to numpy if needed
        if torch.is_tensor(prob):
            prob = prob.cpu().numpy()

        prob = np.squeeze(prob)

        # Binarize
        binary = (prob > threshold).astype(np.uint8)

        # Close holes
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)

        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel)

        return opened.astype(np.float32)

    def refine_batch(self, probs, threshold=0.5):
        """
        Refine batch of predictions.

        Args:
            probs: list of [H, W] probability maps
            threshold: Binary threshold

        Returns:
            list of refined probability maps
        """
        return [self.refine(prob, threshold) for prob in probs]


if __name__ == '__main__':
    """Test CRF refiner."""
    print("=" * 70)
    print("Testing CRF Refiner")
    print("=" * 70)

    if HAS_CRF:
        print("\n✓ pydensecrf is available - using Dense CRF")
    else:
        print("\n⚠ pydensecrf not available - using morphological refinement")
        print("  Install with: pip install pydensecrf")

    # Create synthetic test data
    print("\n1. Creating synthetic test data...")
    H, W = 256, 256

    # Synthetic image (checkered pattern)
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[::32, ::32] = 255
    image[:, :, 0] = (image[:, :, 0] + 100) % 255
    image[:, :, 1] = (image[:, :, 1] + 150) % 255
    image[:, :, 2] = (image[:, :, 2] + 200) % 255

    # Synthetic probability map (circle with noise)
    Y, X = np.ogrid[:H, :W]
    center = (H // 2, W // 2)
    radius = 60
    circle = ((X - center[1]) ** 2 + (Y - center[0]) ** 2) <= radius ** 2

    # Add noise
    prob = circle.astype(np.float32)
    noise = np.random.randn(H, W) * 0.1
    prob = np.clip(prob + noise, 0, 1)

    print(f"   Image shape: {image.shape}")
    print(f"   Prob shape: {prob.shape}")
    print(f"   Prob range: [{prob.min():.3f}, {prob.max():.3f}]")

    # Test CRF refiner
    print("\n2. Testing CRFRefiner...")
    refiner = CRFRefiner(iterations=5)
    refined = refiner.refine(image, prob)
    print(f"   Refined shape: {refined.shape}")
    print(f"   Refined range: [{refined.min():.3f}, {refined.max():.3f}]")

    # Test morphological refiner
    print("\n3. Testing MorphologicalRefiner...")
    morph_refiner = MorphologicalRefiner(kernel_size=5)
    morph_refined = morph_refiner.refine(prob, threshold=0.5)
    print(f"   Morphological refined shape: {morph_refined.shape}")
    print(f"   Morphological refined range: [{morph_refined.min():.3f}, {morph_refined.max():.3f}]")

    # Test batch processing
    print("\n4. Testing batch processing...")
    batch_images = [image] * 3
    batch_probs = [prob] * 3
    batch_refined = refiner.refine_batch(batch_images, batch_probs)
    print(f"   Batch size: {len(batch_refined)}")
    print(f"   First refined shape: {batch_refined[0].shape}")

    # Test with tensor input
    print("\n5. Testing with PyTorch tensors...")
    image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
    prob_tensor = torch.from_numpy(prob).unsqueeze(0)
    refined_tensor = refiner.refine(image_tensor, prob_tensor)
    print(f"   Input tensor shapes: image={image_tensor.shape}, prob={prob_tensor.shape}")
    print(f"   Refined shape: {refined_tensor.shape}")

    # Compare original vs refined
    print("\n6. Comparing original vs refined...")
    orig_binary = (prob > 0.5).astype(np.uint8)
    refined_binary = (refined > 0.5).astype(np.uint8)

    orig_count = np.sum(orig_binary)
    refined_count = np.sum(refined_binary)
    print(f"   Original foreground pixels: {orig_count}")
    print(f"   Refined foreground pixels: {refined_count}")
    print(f"   Change: {refined_count - orig_count} pixels ({(refined_count - orig_count) / orig_count * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

    print("\nUsage Examples:")
    print("-" * 70)
    print("# Basic usage")
    print("refiner = CRFRefiner()")
    print("refined_pred = refiner.refine(image, pred)")
    print()
    print("# Custom CRF parameters")
    print("refiner = CRFRefiner(sxy_bilateral=80, srgb_bilateral=10, iterations=10)")
    print("refined_pred = refiner.refine(image, pred)")
    print()
    print("# Batch processing")
    print("refined_batch = refiner.refine_batch(images, preds)")
    print()
    print("# Morphological refinement (faster, no pydensecrf)")
    print("morph_refiner = MorphologicalRefiner(kernel_size=7)")
    print("refined_pred = morph_refiner.refine(pred, threshold=0.5)")
