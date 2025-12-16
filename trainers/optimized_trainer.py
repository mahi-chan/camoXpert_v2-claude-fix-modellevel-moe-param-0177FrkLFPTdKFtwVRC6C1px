"""
OptimizedTrainer: Advanced training orchestrator for camouflaged object detection.

Features:
1. Cosine annealing with 5-epoch warmup (1e-6 to 1e-4)
2. Expert collapse detection for MoE models
3. Global-batch load balancing for MoE synchronization
4. Gradient accumulation for effective batch size increase
5. Mixed precision training with automatic loss scaling
6. Progressive augmentation (delayed start at epoch 50, max strength 0.5)
7. COD-specific augmentations (Fourier mixing, contrastive, mirror disruption)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import warnings
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    Warms up linearly from base_lr to max_lr over warmup_epochs,
    then decays following cosine annealing to min_lr.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs (default: 5)
        total_epochs: Total number of training epochs
        min_lr: Minimum learning rate (default: 1e-6)
        max_lr: Maximum learning rate after warmup (default: 1e-4)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        max_lr: float = 1e-4,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup from min_lr to max_lr
            alpha = self.last_epoch / self.warmup_epochs
            return [self.min_lr + alpha * (self.max_lr - self.min_lr)
                    for _ in self.base_lrs]
        else:
            # Cosine annealing from max_lr to min_lr
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
                    for _ in self.base_lrs]


class ExpertCollapseDetector:
    """
    Monitors routing statistics to detect expert collapse in MoE models.

    Expert collapse occurs when:
    1. Some experts receive very few tokens (underutilization)
    2. Token distribution is highly imbalanced
    3. Routing confidence is very low

    Args:
        num_experts: Number of experts in the MoE model
        collapse_threshold: Threshold for detecting collapse (default: 0.05)
        window_size: Number of batches to track for moving average (default: 100)
    """
    def __init__(
        self,
        num_experts: int,
        collapse_threshold: float = 0.05,
        window_size: int = 100
    ):
        self.num_experts = num_experts
        self.collapse_threshold = collapse_threshold
        self.window_size = window_size

        # Tracking statistics
        self.expert_counts = deque(maxlen=window_size)  # Per-batch expert usage
        self.routing_confidences = deque(maxlen=window_size)  # Per-batch confidence
        self.load_imbalance_scores = deque(maxlen=window_size)  # Per-batch imbalance

        self.collapse_detected = False
        self.collapse_reasons = []

    def update(self, routing_probs: torch.Tensor, expert_assignments: torch.Tensor):
        """
        Update statistics with current batch routing information.

        Args:
            routing_probs: [batch_size, num_experts] routing probabilities
            expert_assignments: [batch_size, top_k] expert assignments
        """
        batch_size = routing_probs.size(0)

        # 1. Count tokens per expert
        expert_count = torch.zeros(self.num_experts, device=routing_probs.device)
        for i in range(self.num_experts):
            expert_count[i] = (expert_assignments == i).sum()

        # Normalize to percentage
        expert_usage = expert_count.float() / expert_assignments.numel()
        self.expert_counts.append(expert_usage.cpu().numpy())

        # 2. Track routing confidence (max probability)
        max_probs, _ = routing_probs.max(dim=1)
        avg_confidence = max_probs.mean().item()
        self.routing_confidences.append(avg_confidence)

        # 3. Compute load imbalance using coefficient of variation
        mean_usage = expert_usage.mean()
        std_usage = expert_usage.std()
        cv = (std_usage / (mean_usage + 1e-10)).item()
        self.load_imbalance_scores.append(cv)

    def check_collapse(self) -> Tuple[bool, List[str]]:
        """
        Check if expert collapse has occurred.

        Returns:
            collapsed: Boolean indicating if collapse detected
            reasons: List of reasons for collapse detection
        """
        if len(self.expert_counts) < self.window_size // 2:
            return False, []

        # Get recent statistics
        recent_counts = np.array(list(self.expert_counts))
        recent_confidences = np.array(list(self.routing_confidences))
        recent_imbalance = np.array(list(self.load_imbalance_scores))

        # Average usage per expert across batches
        avg_usage = recent_counts.mean(axis=0)

        reasons = []
        collapsed = False

        # Check 1: Underutilized experts
        underutilized = avg_usage < self.collapse_threshold
        if underutilized.any():
            num_underutilized = underutilized.sum()
            min_usage = avg_usage.min()
            reasons.append(
                f"{num_underutilized}/{self.num_experts} experts underutilized "
                f"(min usage: {min_usage:.4f} < {self.collapse_threshold})"
            )
            collapsed = True

        # Check 2: High load imbalance
        avg_imbalance = recent_imbalance.mean()
        if avg_imbalance > 2.0:  # CV > 2.0 indicates high imbalance
            reasons.append(f"High load imbalance (CV: {avg_imbalance:.4f} > 2.0)")
            collapsed = True

        # Check 3: Low routing confidence
        avg_confidence = recent_confidences.mean()
        if avg_confidence < 0.3:  # Very uncertain routing
            reasons.append(f"Low routing confidence ({avg_confidence:.4f} < 0.3)")
            collapsed = True

        self.collapse_detected = collapsed
        self.collapse_reasons = reasons

        return collapsed, reasons

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics summary."""
        if len(self.expert_counts) == 0:
            return {}

        recent_counts = np.array(list(self.expert_counts))
        recent_confidences = np.array(list(self.routing_confidences))
        recent_imbalance = np.array(list(self.load_imbalance_scores))

        avg_usage = recent_counts.mean(axis=0)

        return {
            'expert_usage': avg_usage.tolist(),
            'min_usage': float(avg_usage.min()),
            'max_usage': float(avg_usage.max()),
            'avg_confidence': float(recent_confidences.mean()),
            'load_imbalance_cv': float(recent_imbalance.mean()),
            'collapsed': self.collapse_detected,
            'collapse_reasons': self.collapse_reasons
        }


class GlobalBatchLoadBalancer:
    """
    Tracks expert usage across entire batches for load balancing in MoE models.

    Computes global statistics to encourage balanced expert utilization:
    1. Expert usage frequency (how often each expert is selected)
    2. Load balance loss (L2 + Coefficient of Variation)
    3. Expert capacity overflow tracking

    Args:
        num_experts: Number of experts
        alpha_l2: Weight for L2 load balance loss (default: 0.01)
        alpha_cv: Weight for CV load balance loss (default: 0.01)
    """
    def __init__(
        self,
        num_experts: int,
        alpha_l2: float = 0.01,
        alpha_cv: float = 0.01
    ):
        self.num_experts = num_experts
        self.alpha_l2 = alpha_l2
        self.alpha_cv = alpha_cv

        # Global statistics across training
        self.global_expert_counts = torch.zeros(num_experts)
        self.global_token_count = 0

        # Batch-level tracking
        self.batch_expert_usage = []
        self.capacity_overflow_count = 0

    def update(
        self,
        routing_probs: torch.Tensor,
        expert_assignments: torch.Tensor,
        capacity_overflow: Optional[torch.Tensor] = None
    ):
        """
        Update global statistics with current batch.

        Args:
            routing_probs: [batch_size, num_experts] routing probabilities
            expert_assignments: [batch_size, top_k] expert assignments
            capacity_overflow: [batch_size] boolean indicating overflow
        """
        # Count expert usage in this batch
        expert_count = torch.zeros(self.num_experts, device=routing_probs.device)
        for i in range(self.num_experts):
            expert_count[i] = (expert_assignments == i).sum()

        # Update global counts (ensure both tensors are on CPU)
        expert_count_cpu = expert_count.cpu()
        self.global_expert_counts = self.global_expert_counts.cpu()  # Ensure it's on CPU
        self.global_expert_counts += expert_count_cpu
        self.global_token_count += expert_assignments.numel()

        # Track batch usage
        batch_usage = expert_count.float() / expert_assignments.numel()
        self.batch_expert_usage.append(batch_usage.cpu().numpy())

        # Track capacity overflow
        if capacity_overflow is not None:
            self.capacity_overflow_count += capacity_overflow.sum().item()

    def compute_load_balance_loss(
        self,
        routing_probs: torch.Tensor,
        expert_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balance loss to encourage uniform expert usage.

        Combines two metrics:
        1. L2 loss: Penalizes deviation from uniform distribution
        2. CV loss: Penalizes high coefficient of variation

        Args:
            routing_probs: [batch_size, num_experts] routing probabilities
            expert_assignments: [batch_size, top_k] expert assignments

        Returns:
            load_balance_loss: Scalar tensor
        """
        # Count tokens per expert in this batch
        expert_count = torch.zeros(self.num_experts, device=routing_probs.device)
        for i in range(self.num_experts):
            expert_count[i] = (expert_assignments == i).sum()

        # Normalize to fraction
        total_tokens = expert_assignments.numel()
        expert_fraction = expert_count / (total_tokens + 1e-10)

        # Target: uniform distribution
        uniform_target = torch.ones_like(expert_fraction) / self.num_experts

        # L2 loss: squared deviation from uniform
        l2_loss = F.mse_loss(expert_fraction, uniform_target)

        # Coefficient of Variation loss
        mean_frac = expert_fraction.mean()
        std_frac = expert_fraction.std()
        cv = std_frac / (mean_frac + 1e-10)
        cv_loss = cv ** 2

        # Combined loss
        load_balance_loss = self.alpha_l2 * l2_loss + self.alpha_cv * cv_loss

        return load_balance_loss

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics summary."""
        if self.global_token_count == 0:
            return {}

        global_fraction = self.global_expert_counts / self.global_token_count

        return {
            'global_expert_usage': global_fraction.tolist(),
            'min_global_usage': float(global_fraction.min()),
            'max_global_usage': float(global_fraction.max()),
            'global_usage_std': float(global_fraction.std()),
            'capacity_overflow_rate': self.capacity_overflow_count / self.global_token_count
                                     if self.global_token_count > 0 else 0.0,
            'total_tokens_processed': self.global_token_count
        }

    def reset_batch_statistics(self):
        """Reset batch-level statistics."""
        self.batch_expert_usage = []


class CODProgressiveAugmentation:
    """
    Progressive augmentation for camouflaged object detection.

    Increases augmentation strength after epoch 50 to improve robustness.
    Delayed start prevents interference with early convergence.
    Includes COD-specific augmentations:
    1. Fourier-based mixing: Mix images in frequency domain
    2. Contrastive learning: Generate positive/negative pairs
    3. Mirror disruption: Break symmetry assumptions

    Args:
        initial_strength: Initial augmentation strength (default: 0.3)
        max_strength: Maximum augmentation strength (default: 0.5)
        transition_epoch: Epoch to start increasing strength (default: 50)
        transition_duration: Epochs over which to ramp up (default: 50)
    """
    def __init__(
        self,
        initial_strength: float = 0.3,
        max_strength: float = 0.5,
        transition_epoch: int = 50,
        transition_duration: int = 50
    ):
        self.initial_strength = initial_strength
        self.max_strength = max_strength
        self.transition_epoch = transition_epoch
        self.transition_duration = transition_duration

        self.current_strength = initial_strength
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """Update current epoch and augmentation strength."""
        self.current_epoch = epoch

        if epoch < self.transition_epoch:
            self.current_strength = self.initial_strength
        elif epoch < self.transition_epoch + self.transition_duration:
            # Linear ramp-up
            progress = (epoch - self.transition_epoch) / self.transition_duration
            self.current_strength = self.initial_strength + \
                                   progress * (self.max_strength - self.initial_strength)
        else:
            self.current_strength = self.max_strength

    def fourier_based_mixing(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        masks1: torch.Tensor,
        masks2: torch.Tensor,
        alpha: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix two images in the frequency domain using FFT.

        Fourier-based mixing preserves high-frequency details (important for camouflage)
        while blending low-frequency color/illumination information.

        Args:
            images1: [B, C, H, W] first batch of images
            images2: [B, C, H, W] second batch of images
            masks1: [B, 1, H, W] first batch of masks
            masks2: [B, 1, H, W] second batch of masks
            alpha: Mixing ratio (default: random based on current_strength)

        Returns:
            mixed_images: [B, C, H, W] mixed images
            mixed_masks: [B, 1, H, W] mixed masks
        """
        if alpha is None:
            # Sample mixing ratio based on current strength
            alpha = np.random.beta(self.current_strength, self.current_strength)

        B, C, H, W = images1.shape

        # Convert to frequency domain
        fft1 = torch.fft.rfft2(images1, norm='ortho')
        fft2 = torch.fft.rfft2(images2, norm='ortho')

        # Separate amplitude and phase
        amp1, phase1 = torch.abs(fft1), torch.angle(fft1)
        amp2, phase2 = torch.abs(fft2), torch.angle(fft2)

        # Create frequency mask for selective mixing
        # Mix low frequencies more, preserve high frequencies
        freq_h, freq_w = fft1.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(freq_h, device=images1.device),
            torch.arange(freq_w, device=images1.device),
            indexing='ij'
        )

        # Distance from DC component (low frequency center)
        center_h, center_w = freq_h // 2, freq_w // 2
        freq_distance = torch.sqrt(((y - center_h) / freq_h) ** 2 +
                                   (x / freq_w) ** 2)

        # Lower alpha for high frequencies (preserve details)
        adaptive_alpha = alpha * torch.exp(-2 * freq_distance)
        adaptive_alpha = adaptive_alpha[None, None, :, :]

        # Mix amplitudes adaptively
        mixed_amp = adaptive_alpha * amp1 + (1 - adaptive_alpha) * amp2

        # Mix phases (important for preserving structure)
        mixed_phase = adaptive_alpha * phase1 + (1 - adaptive_alpha) * phase2

        # Reconstruct mixed FFT
        mixed_fft = mixed_amp * torch.exp(1j * mixed_phase)

        # Convert back to spatial domain
        mixed_images = torch.fft.irfft2(mixed_fft, s=(H, W), norm='ortho')

        # Mix masks in spatial domain (simple linear blend)
        mixed_masks = alpha * masks1 + (1 - alpha) * masks2

        return mixed_images, mixed_masks

    def contrastive_augmentation(
        self,
        images: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate positive and negative pairs for contrastive learning.

        Positive pairs: Same image with different augmentations
        Negative pairs: Different images

        Args:
            images: [B, C, H, W] input images
            masks: [B, 1, H, W] input masks

        Returns:
            anchor_images: [B, C, H, W] anchor images
            positive_images: [B, C, H, W] positive augmentations
            anchor_masks: [B, 1, H, W] anchor masks
            positive_masks: [B, 1, H, W] positive masks
        """
        B, C, H, W = images.shape

        # Anchor: original images with mild augmentation
        anchor_images = images
        anchor_masks = masks

        # Positive: same images with stronger augmentation
        positive_images = images.clone()
        positive_masks = masks.clone()

        # Apply color jittering (brightness, contrast, saturation)
        if torch.rand(1).item() < self.current_strength:
            # Brightness adjustment
            brightness_factor = 1.0 + self.current_strength * (torch.rand(B, 1, 1, 1, device=images.device) - 0.5)
            positive_images = positive_images * brightness_factor

            # Contrast adjustment
            contrast_factor = 1.0 + self.current_strength * (torch.rand(B, 1, 1, 1, device=images.device) - 0.5)
            mean = positive_images.mean(dim=[2, 3], keepdim=True)
            positive_images = (positive_images - mean) * contrast_factor + mean

        # Apply Gaussian blur
        if torch.rand(1).item() < self.current_strength * 0.5:
            kernel_size = int(5 + self.current_strength * 10)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            sigma = self.current_strength * 2.0

            # Create Gaussian kernel
            coords = torch.arange(kernel_size, dtype=torch.float32, device=images.device) - kernel_size // 2
            kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()

            # Apply 1D convolutions (separable)
            kernel_1d = kernel.view(1, 1, -1, 1)
            positive_images = F.conv2d(positive_images.reshape(-1, 1, H, W), kernel_1d, padding=(kernel_size//2, 0))
            positive_images = F.conv2d(positive_images, kernel_1d.transpose(-1, -2), padding=(0, kernel_size//2))
            positive_images = positive_images.reshape(B, C, H, W)

        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            positive_images = torch.flip(positive_images, dims=[3])
            positive_masks = torch.flip(positive_masks, dims=[3])

        # Clamp to valid range
        positive_images = torch.clamp(positive_images, 0, 1)

        return anchor_images, positive_images, anchor_masks, positive_masks

    def mirror_disruption(
        self,
        images: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Break symmetry assumptions by random mirroring and asymmetric crops.

        Camouflaged objects may have natural symmetry. This augmentation
        disrupts that to improve model robustness.

        Args:
            images: [B, C, H, W] input images
            masks: [B, 1, H, W] input masks

        Returns:
            augmented_images: [B, C, H, W] augmented images
            augmented_masks: [B, 1, H, W] augmented masks
        """
        B, C, H, W = images.shape

        augmented_images = images.clone()
        augmented_masks = masks.clone()

        for i in range(B):
            if torch.rand(1).item() < self.current_strength:
                # Random mirror mode
                mode = np.random.choice(['horizontal', 'vertical', 'both', 'diagonal'])

                if mode == 'horizontal':
                    augmented_images[i] = torch.flip(augmented_images[i], dims=[2]).contiguous()
                    augmented_masks[i] = torch.flip(augmented_masks[i], dims=[2]).contiguous()

                elif mode == 'vertical':
                    augmented_images[i] = torch.flip(augmented_images[i], dims=[1]).contiguous()
                    augmented_masks[i] = torch.flip(augmented_masks[i], dims=[1]).contiguous()

                elif mode == 'both':
                    augmented_images[i] = torch.flip(augmented_images[i], dims=[1, 2]).contiguous()
                    augmented_masks[i] = torch.flip(augmented_masks[i], dims=[1, 2]).contiguous()

                elif mode == 'diagonal':
                    # Transpose-like operation (swap H and W)
                    if H == W:
                        augmented_images[i] = augmented_images[i].transpose(1, 2).contiguous()
                        augmented_masks[i] = augmented_masks[i].transpose(1, 2).contiguous()

            # Asymmetric crop and resize (if strength is high enough)
            if torch.rand(1).item() < self.current_strength * 0.5:
                # Random crop size (retain 70-95% of image)
                crop_factor = 0.7 + 0.25 * (1 - self.current_strength)
                crop_h = int(H * crop_factor)
                crop_w = int(W * crop_factor)

                # Random crop position (asymmetric)
                top = torch.randint(0, H - crop_h + 1, (1,)).item()
                left = torch.randint(0, W - crop_w + 1, (1,)).item()

                # Crop
                cropped_img = augmented_images[i:i+1, :, top:top+crop_h, left:left+crop_w]
                cropped_mask = augmented_masks[i:i+1, :, top:top+crop_h, left:left+crop_w]

                # Resize back to original size
                augmented_images[i] = F.interpolate(cropped_img, size=(H, W), mode='bilinear', align_corners=False)[0]
                augmented_masks[i] = F.interpolate(cropped_mask, size=(H, W), mode='nearest')[0]

        return augmented_images, augmented_masks

    def apply(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        augmentation_type: str = 'random'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation based on type.

        Args:
            images: [B, C, H, W] input images
            masks: [B, 1, H, W] input masks
            augmentation_type: 'fourier', 'contrastive', 'mirror', or 'random'

        Returns:
            augmented_images: Augmented images
            augmented_masks: Augmented masks
        """
        if augmentation_type == 'random':
            aug_type = np.random.choice(['fourier', 'contrastive', 'mirror'])
        else:
            aug_type = augmentation_type

        if aug_type == 'fourier':
            # Need two batches for mixing
            B = images.size(0)
            if B > 1:
                # Shuffle and mix
                indices = torch.randperm(B, device=images.device)
                images2 = images[indices]
                masks2 = masks[indices]
                return self.fourier_based_mixing(images, images2, masks, masks2)
            else:
                return images, masks

        elif aug_type == 'contrastive':
            # Return positive pairs
            _, positive_images, _, positive_masks = self.contrastive_augmentation(images, masks)
            return positive_images, positive_masks

        elif aug_type == 'mirror':
            return self.mirror_disruption(images, masks)

        else:
            return images, masks


class OptimizedTrainer:
    """
    Advanced training orchestrator for camouflaged object detection.

    Features:
    1. Cosine annealing with 5-epoch warmup (1e-6 to 1e-4)
    2. Expert collapse detection for MoE models
    3. Global-batch load balancing for MoE synchronization
    4. Gradient accumulation for effective batch size increase
    5. Mixed precision training with automatic loss scaling
    6. Progressive augmentation (increases after epoch 50, delayed for convergence)
    7. COD-specific augmentations

    Args:
        model: PyTorch model to train
        optimizer: Optimizer instance
        criterion: Loss function
        device: Training device
        accumulation_steps: Number of gradient accumulation steps (default: 1)
        use_amp: Use automatic mixed precision (default: True)
        total_epochs: Total training epochs (default: 100)
        warmup_epochs: Warmup epochs (default: 5)
        min_lr: Minimum learning rate (default: 1e-6)
        max_lr: Maximum learning rate (default: 1e-4)
        num_experts: Number of experts in MoE (default: None)
        enable_load_balancing: Enable MoE load balancing (default: False)
        enable_collapse_detection: Enable expert collapse detection (default: False)
        enable_progressive_aug: Enable progressive augmentation (default: True)
        aug_transition_epoch: Epoch to start increasing augmentation (default: 50)
        aug_max_strength: Maximum augmentation strength (default: 0.5)
        aug_transition_duration: Epochs to ramp up augmentation (default: 50)
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        total_epochs: int = 100,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        max_lr: float = 1e-4,
        num_experts: Optional[int] = None,
        enable_load_balancing: bool = False,
        enable_collapse_detection: bool = False,
        enable_progressive_aug: bool = True,
        aug_transition_epoch: int = 50,
        aug_max_strength: float = 0.5,
        aug_transition_duration: int = 50,
        ema = None  # EMA instance for per-step updates
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.ema = ema  # Store EMA for per-step updates

        # Learning rate scheduler
        self.scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
            max_lr=max_lr
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=use_amp)

        # MoE components
        self.num_experts = num_experts
        self.enable_load_balancing = enable_load_balancing and num_experts is not None
        self.enable_collapse_detection = enable_collapse_detection and num_experts is not None

        if self.enable_load_balancing:
            self.load_balancer = GlobalBatchLoadBalancer(num_experts=num_experts)
        else:
            self.load_balancer = None

        if self.enable_collapse_detection:
            self.collapse_detector = ExpertCollapseDetector(num_experts=num_experts)
        else:
            self.collapse_detector = None

        # Progressive augmentation
        self.enable_progressive_aug = enable_progressive_aug
        if enable_progressive_aug:
            self.augmentation = CODProgressiveAugmentation(
                transition_epoch=aug_transition_epoch,
                max_strength=aug_max_strength,
                transition_duration=aug_transition_duration
            )
        else:
            self.augmentation = None

        # Training statistics
        self.current_epoch = 0
        self.global_step = 0
        self.epoch_losses = []
        self.lr_history = []

    def train_epoch(
        self,
        train_loader,
        epoch: int,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            log_interval: Logging interval in batches

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        self.current_epoch = epoch

        # Update augmentation strength
        if self.enable_progressive_aug:
            self.augmentation.update_epoch(epoch)

        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_lb_loss = 0.0
        num_batches = 0

        # Track expert selections for monitoring router collapse
        expert_selection_counts = {}

        # Reset gradient accumulation
        self.optimizer.zero_grad()

        # Wrap with tqdm if available
        if TQDM_AVAILABLE:
            train_iter = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=100, leave=True)
        else:
            train_iter = train_loader

        for batch_idx, (images, masks) in enumerate(train_iter):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Apply progressive augmentation
            if self.enable_progressive_aug and self.augmentation is not None:
                # Randomly choose augmentation type
                if torch.rand(1).item() < self.augmentation.current_strength:
                    images, masks = self.augmentation.apply(images, masks, 'random')

            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                # Forward pass - always request routing info during training for MoE
                outputs = self.model(images, return_routing_info=True)

                # Handle different output formats
                if isinstance(outputs, dict):
                    predictions = outputs['predictions']
                    aux_outputs = outputs.get('aux_outputs', None)
                    routing_info = outputs.get('routing_info', None)
                elif isinstance(outputs, tuple):
                    # Model returns (prediction, routing_info) where routing_info is a dict
                    predictions = outputs[0]
                    if len(outputs) > 1 and isinstance(outputs[1], dict):
                        routing_info = outputs[1]
                        aux_outputs = routing_info.get('aux_outputs', None)
                    else:
                        routing_info = None
                        aux_outputs = outputs[1] if len(outputs) > 1 else None
                else:
                    predictions = outputs
                    aux_outputs = None
                    routing_info = None

                # Compute main loss (pass input_image for frequency-weighted loss)
                loss = self.criterion(predictions, masks, input_image=images)

                # Add auxiliary loss if available
                aux_loss = 0.0
                if aux_outputs is not None and isinstance(aux_outputs, (list, tuple)):
                    for aux_pred in aux_outputs:
                        aux_loss += 0.4 * self.criterion(aux_pred, masks, input_image=images)
                    loss = loss + aux_loss

                # Add load balancing loss if MoE
                lb_loss = 0.0
                if self.enable_load_balancing and routing_info is not None:
                    routing_probs = routing_info.get('routing_probs')
                    expert_assignments = routing_info.get('expert_assignments')

                    if routing_probs is not None and expert_assignments is not None:
                        lb_loss = self.load_balancer.compute_load_balance_loss(
                            routing_probs, expert_assignments
                        )
                        loss = loss + lb_loss

                        # Update load balancer statistics
                        self.load_balancer.update(routing_probs, expert_assignments)

                        # Track expert selections for monitoring
                        with torch.no_grad():
                            for expert_idx in expert_assignments.flatten().cpu().numpy():
                                expert_selection_counts[expert_idx] = expert_selection_counts.get(expert_idx, 0) + 1

                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update expert collapse detector
            if self.enable_collapse_detection and routing_info is not None:
                routing_probs = routing_info.get('routing_probs')
                expert_assignments = routing_info.get('expert_assignments')

                if routing_probs is not None and expert_assignments is not None:
                    with torch.no_grad():
                        self.collapse_detector.update(routing_probs, expert_assignments)

            # Optimizer step after accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Update EMA after each optimizer step (per-step update)
                if self.ema is not None:
                    self.ema.update()

                self.global_step += 1

            # Accumulate losses
            epoch_loss += loss.item() * self.accumulation_steps
            if isinstance(aux_loss, torch.Tensor):
                epoch_aux_loss += aux_loss.item()
            if isinstance(lb_loss, torch.Tensor):
                epoch_lb_loss += lb_loss.item()
            num_batches += 1

            # Logging
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']

                # Update tqdm progress bar if available
                if TQDM_AVAILABLE and isinstance(train_iter, tqdm):
                    postfix_dict = {
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.6f}'
                    }
                    if self.enable_progressive_aug and self.augmentation is not None:
                        postfix_dict['aug'] = f'{self.augmentation.current_strength:.2f}'
                    train_iter.set_postfix(postfix_dict)
                else:
                    # Fallback to print if tqdm not available
                    print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
                    if self.enable_progressive_aug and self.augmentation is not None:
                        print(f"  Aug Strength: {self.augmentation.current_strength:.3f}")

        # Step scheduler
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)

        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        self.epoch_losses.append(avg_loss)

        metrics = {
            'loss': avg_loss,
            'lr': current_lr
        }

        if epoch_aux_loss > 0:
            metrics['aux_loss'] = epoch_aux_loss / num_batches

        if epoch_lb_loss > 0:
            metrics['load_balance_loss'] = epoch_lb_loss / num_batches

        # Check for expert collapse
        if self.enable_collapse_detection:
            collapsed, reasons = self.collapse_detector.check_collapse()
            if collapsed:
                warnings.warn(f"Expert collapse detected at epoch {epoch}:")
                for reason in reasons:
                    warnings.warn(f"  - {reason}")

            stats = self.collapse_detector.get_statistics()
            metrics.update({f'collapse_{k}': v for k, v in stats.items()})

        # Global load balancing statistics
        if self.enable_load_balancing:
            global_stats = self.load_balancer.get_global_statistics()
            metrics.update({f'global_{k}': v for k, v in global_stats.items()})

        # Add expert selection statistics
        if expert_selection_counts:
            total_selections = sum(expert_selection_counts.values())
            metrics['expert_selections'] = expert_selection_counts
            metrics['expert_selection_pcts'] = {
                f'expert_{k}': (v / total_selections * 100)
                for k, v in expert_selection_counts.items()
            }

        return metrics

    def validate(
        self,
        val_loader,
        metrics_fn=None
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data
            metrics_fn: Optional function to compute additional metrics
                        NOTE: metrics_fn should accept (pred, target) tensors for a SINGLE BATCH
                        and return a dict of metric values

        Returns:
            metrics: Dictionary of validation metrics (synchronized across DDP ranks)
        """
        import torch.distributed as dist
        import torch.nn.functional as F
        
        self.model.eval()
        
        # Simple explicit accumulators - NO DICTS, NO LISTS
        total_loss = 0.0
        total_mae = 0.0
        total_iou = 0.0
        total_f = 0.0
        total_s = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                batch_size = images.size(0)
                
                # Forward pass
                with autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        preds = outputs['predictions']
                    elif isinstance(outputs, tuple):
                        preds = outputs[0]
                    else:
                        preds = outputs
                    
                    loss = self.criterion(preds, masks, input_image=images)
                
                total_loss += loss.item() * batch_size
                
                # Compute metrics INLINE - no external function
                preds_prob = torch.sigmoid(preds.detach())
                if preds_prob.shape[2:] != masks.shape[2:]:
                    preds_prob = F.interpolate(preds_prob, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                # Create CODMetrics once per batch (not per image)
                from metrics.cod_metrics import CODMetrics
                metrics_calc = CODMetrics()
                
                for i in range(batch_size):
                    p = preds_prob[i, 0]  # [H, W]
                    t = masks[i, 0]       # [H, W]
                    
                    # MAE
                    total_mae += torch.abs(p - t).mean().item()
                    
                    # S-measure - use proper CODMetrics (created above)
                    total_s += metrics_calc.s_measure(preds_prob[i:i+1], masks[i:i+1])
                    
                    # Binary mask for IoU/F-measure (0.5 threshold)
                    p_bin = (p > 0.5).float()
                    
                    # IoU
                    inter_bin = (p_bin * t).sum()
                    union_bin = p_bin.sum() + t.sum() - inter_bin
                    total_iou += ((inter_bin + 1e-6) / (union_bin + 1e-6)).item()
                    
                    # F-measure (beta=0.3)
                    tp = (p_bin * t).sum()
                    fp = (p_bin * (1 - t)).sum()
                    fn = ((1 - p_bin) * t).sum()
                    prec = (tp + 1e-6) / (tp + fp + 1e-6)
                    rec = (tp + 1e-6) / (tp + fn + 1e-6)
                    beta = 0.3
                    total_f += (((1 + beta**2) * prec * rec) / (beta**2 * prec + rec + 1e-6)).item()
                
                total_samples += batch_size
        
        # Compute local averages
        metrics = {
            'val_loss': total_loss / max(total_samples, 1),
            'val_mae': total_mae / max(total_samples, 1),
            'val_iou': total_iou / max(total_samples, 1),
            'val_f_measure': total_f / max(total_samples, 1),
            'val_s_measure': total_s / max(total_samples, 1)
        }
        
        # WARNING: Detect when IoU and F-measure are suspiciously identical (bug indicator)
        # Note: Check AFTER DDP sync to see the actual final values
        pre_sync_iou = metrics['val_iou']
        pre_sync_f = metrics['val_f_measure']
        
        # DDP sync
        if dist.is_initialized():
            sync_tensor = torch.tensor([
                total_iou, total_f, total_mae, total_s, total_loss, float(total_samples)
            ], device=self.device, dtype=torch.float32)
            dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            
            total_all = sync_tensor[5].item()
            metrics = {
                'val_iou': sync_tensor[0].item() / max(total_all, 1),
                'val_f_measure': sync_tensor[1].item() / max(total_all, 1),
                'val_mae': sync_tensor[2].item() / max(total_all, 1),
                'val_s_measure': sync_tensor[3].item() / max(total_all, 1),
                'val_loss': sync_tensor[4].item() / max(total_all, 1)
            }
        
        # Check FINAL values (post-sync) for identical IoU/F-measure bug
        iou_val = metrics['val_iou']
        f_val = metrics['val_f_measure']
        if abs(iou_val - f_val) < 0.0001 and iou_val > 0.1:  # Same to 4 decimals AND non-trivial
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\n  ⚠️ [BUG DETECTED] IoU ({iou_val:.6f}) ≈ F-measure ({f_val:.6f})")
                print(f"     Pre-sync: IoU={pre_sync_iou:.6f}, F={pre_sync_f:.6f}")
                print(f"     Post-sync: IoU={iou_val:.6f}, F={f_val:.6f}")
                print(f"     This indicates a potential calculation bug!")
        
        return metrics

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'lr_history': self.lr_history,
            'epoch_losses': self.epoch_losses
        }

        # Save MoE statistics
        if self.enable_load_balancing:
            checkpoint['load_balancer_state'] = {
                'global_expert_counts': self.load_balancer.global_expert_counts,
                'global_token_count': self.load_balancer.global_token_count
            }

        if self.enable_collapse_detection:
            checkpoint['collapse_detector_state'] = self.collapse_detector.get_statistics()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str, weights_only: bool = False) -> int:
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file
            weights_only: If True, only load model weights (allows changing lr/wd)

        Returns:
            epoch: Epoch number to resume from
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Always load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if weights_only:
            # Only load model weights, keep fresh optimizer/scheduler
            print(f"✓ Loaded model weights only from {filepath} (epoch {checkpoint['epoch']})")
            print(f"  Optimizer/scheduler: Using NEW hyperparameters")
            epoch = 0  # Restart epoch counter for new training phase
        else:
            # Load full checkpoint (optimizer, scheduler, scaler)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            self.lr_history = checkpoint.get('lr_history', [])
            self.epoch_losses = checkpoint.get('epoch_losses', [])

            # Restore MoE statistics
            if self.enable_load_balancing and 'load_balancer_state' in checkpoint:
                lb_state = checkpoint['load_balancer_state']
                self.load_balancer.global_expert_counts = lb_state['global_expert_counts']
                self.load_balancer.global_token_count = lb_state['global_token_count']

            epoch = checkpoint['epoch']
            print(f"✓ Loaded full checkpoint from {filepath} (epoch {epoch})")

        return epoch

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        summary = {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'best_loss': min(self.epoch_losses) if self.epoch_losses else float('inf'),
            'latest_loss': self.epoch_losses[-1] if self.epoch_losses else None
        }

        if self.enable_progressive_aug:
            summary['augmentation_strength'] = self.augmentation.current_strength

        if self.enable_collapse_detection:
            summary['expert_collapse_detected'] = self.collapse_detector.collapse_detected
            summary['collapse_stats'] = self.collapse_detector.get_statistics()

        if self.enable_load_balancing:
            summary['global_load_stats'] = self.load_balancer.get_global_statistics()

        return summary
