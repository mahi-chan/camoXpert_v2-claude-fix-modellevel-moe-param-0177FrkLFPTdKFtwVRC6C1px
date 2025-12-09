"""
Sophisticated Router Network for Model-Level MoE - UPGRADED VERSION

Advanced router with comprehensive feature analysis and Expert Choice routing.

Features analyzed:
1. Texture complexity using multi-scale dilated convolutions (rates 1, 2, 4)
2. Edge density with Sobel-initialized edge detection kernels
3. Frequency content using FFT analysis (high vs low frequency dominance)
4. Context scale with pyramid pooling (1x1, 2x2, 3x3, 6x6)
5. Uncertainty estimation for routing confidence scores

Routing mechanisms:
- Traditional Token Choice: Tokens select top-k experts
- Expert Choice (NEW): Experts select top-k tokens
- Global-batch load balancing to prevent expert collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional


class MultiScaleTextureAnalyzer(nn.Module):
    """
    Texture complexity branch using multi-scale dilated convolutions.

    Uses dilation rates [1, 2, 4] to capture texture at different scales:
    - Rate 1: Fine-grained texture patterns
    - Rate 2: Medium-scale texture
    - Rate 4: Coarse texture patterns
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # Three parallel dilated convolution branches
        # Note: out_channels // 3 may not divide evenly, so we calculate explicitly
        ch1 = out_channels // 3
        ch2 = out_channels // 3
        ch3 = out_channels - ch1 - ch2  # Remainder goes to last branch

        self.dilation1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1, 3, padding=1, dilation=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True)
        )

        self.dilation2 = nn.Sequential(
            nn.Conv2d(in_channels, ch2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True)
        )

        self.dilation4 = nn.Sequential(
            nn.Conv2d(in_channels, ch3, 3, padding=4, dilation=4),
            nn.BatchNorm2d(ch3),
            nn.ReLU(inplace=True)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels]
        """
        # Extract multi-scale texture features
        feat1 = self.dilation1(x)  # Fine texture
        feat2 = self.dilation2(x)  # Medium texture
        feat4 = self.dilation4(x)  # Coarse texture

        # Concatenate and fuse
        combined = torch.cat([feat1, feat2, feat4], dim=1)
        output = self.fusion(combined)

        return output.flatten(1)


class SobelEdgeAnalyzer(nn.Module):
    """
    Edge density branch with Sobel-initialized edge detection kernels.

    Detects edges in horizontal, vertical, and diagonal directions.
    Kernels are initialized with Sobel operators but remain learnable.
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # Edge detection convolutions (initialized with Sobel)
        self.horizontal_edge = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, bias=False)
        self.vertical_edge = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, bias=False)
        self.diagonal1_edge = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, bias=False)
        self.diagonal2_edge = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, bias=False)

        # Initialize with Sobel kernels
        self._init_sobel_kernels()

        # Post-processing
        self.post_process = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def _init_sobel_kernels(self):
        """Initialize convolution kernels with Sobel edge detectors"""
        # Horizontal Sobel (detects vertical edges)
        sobel_h = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32) / 8.0

        # Vertical Sobel (detects horizontal edges)
        sobel_v = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32) / 8.0

        # Diagonal Sobel 1 (detects diagonal edges ↘)
        sobel_d1 = torch.tensor([
            [-2, -1,  0],
            [-1,  0,  1],
            [ 0,  1,  2]
        ], dtype=torch.float32) / 8.0

        # Diagonal Sobel 2 (detects diagonal edges ↙)
        sobel_d2 = torch.tensor([
            [ 0, -1, -2],
            [ 1,  0, -1],
            [ 2,  1,  0]
        ], dtype=torch.float32) / 8.0

        # Apply to all channels
        in_channels = self.horizontal_edge.in_channels
        out_channels = self.horizontal_edge.out_channels

        for i in range(out_channels):
            for j in range(in_channels):
                self.horizontal_edge.weight.data[i, j] = sobel_h
                self.vertical_edge.weight.data[i, j] = sobel_v
                self.diagonal1_edge.weight.data[i, j] = sobel_d1
                self.diagonal2_edge.weight.data[i, j] = sobel_d2

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels]
        """
        # Detect edges in all directions
        h_edges = self.horizontal_edge(x)
        v_edges = self.vertical_edge(x)
        d1_edges = self.diagonal1_edge(x)
        d2_edges = self.diagonal2_edge(x)

        # Combine all edge features
        edge_features = torch.cat([h_edges, v_edges, d1_edges, d2_edges], dim=1)

        # Post-process and pool
        output = self.post_process(edge_features)

        return output.flatten(1)


class FFTFrequencyAnalyzer(nn.Module):
    """
    Frequency content branch using FFT analysis.

    Analyzes frequency spectrum to detect high vs low frequency dominance.
    Uses 2D FFT to decompose spatial features into frequency components.
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # Dimensionality reduction before FFT (for efficiency)
        self.pre_process = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Process low-frequency components
        self.low_freq_net = nn.Sequential(
            nn.Conv2d(128, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # Process high-frequency components
        self.high_freq_net = nn.Sequential(
            nn.Conv2d(128, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # Final processing
        self.post_process = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels] - Frequency domain features
        """
        B, C, H, W = x.shape

        # Pre-process to reduce channels
        x = self.pre_process(x)  # [B, 128, H, W]

        # Disable AMP for FFT operations (cuFFT requires power-of-2 sizes in half precision)
        with torch.amp.autocast('cuda', enabled=False):
            # Convert to float32 for FFT
            x_float = x.float()

            # Apply 2D FFT (per channel)
            # FFT along spatial dimensions
            x_fft = torch.fft.rfft2(x_float, dim=(-2, -1), norm='ortho')  # [B, 128, H, W//2+1]

            # Get magnitude spectrum
            x_mag = torch.abs(x_fft)  # [B, 128, H, W//2+1]

        # Separate low and high frequency components
        # Low frequencies are in the center/low indices
        h_half = H // 2
        w_half = x_mag.shape[-1] // 2

        # Create masks for low and high frequencies
        # Low frequency: center region (DC and low frequencies)
        low_freq_mask = torch.zeros_like(x_mag)
        low_freq_mask[:, :, :h_half//2, :w_half//2] = 1.0

        # High frequency: outer region
        high_freq_mask = 1.0 - low_freq_mask

        # Apply masks
        low_freq_mag = x_mag * low_freq_mask
        high_freq_mag = x_mag * high_freq_mask

        # Convert back to spatial domain for CNN processing
        # Use magnitude as features (real-valued)
        # Pad to match original spatial size if needed
        if low_freq_mag.shape[-1] < W:
            pad_size = W - low_freq_mag.shape[-1]
            low_freq_mag = F.pad(low_freq_mag, (0, pad_size, 0, 0))
            high_freq_mag = F.pad(high_freq_mag, (0, pad_size, 0, 0))

        # Process low and high frequency features separately
        low_feat = self.low_freq_net(low_freq_mag[:, :, :H, :W])
        high_feat = self.high_freq_net(high_freq_mag[:, :, :H, :W])

        # Combine
        combined = torch.cat([low_feat, high_feat], dim=1)

        # Post-process
        output = self.post_process(combined)

        return output


class PyramidPoolingAnalyzer(nn.Module):
    """
    Context scale branch with pyramid pooling.

    Pools at multiple scales (1x1, 2x2, 3x3, 6x6) to capture
    context at different spatial resolutions.
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # Pyramid pooling levels
        self.pool_scales = [1, 2, 3, 6]

        # Convolution for each pooling level
        self.pyramid_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels // len(self.pool_scales), 1),
                nn.BatchNorm2d(out_channels // len(self.pool_scales)),
                nn.ReLU(inplace=True)
            ) for scale in self.pool_scales
        ])

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels]
        """
        B, C, H, W = x.shape

        # Apply pyramid pooling at each scale
        pyramid_features = []
        for conv in self.pyramid_convs:
            # Pool and process
            pooled = conv(x)  # [B, out_channels//4, scale, scale]

            # Upsample back to original size
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            pyramid_features.append(upsampled)

        # Concatenate all pyramid levels
        combined = torch.cat(pyramid_features, dim=1)  # [B, out_channels, H, W]

        # Fuse and pool
        output = self.fusion(combined)

        return output.flatten(1)


class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimation branch that outputs routing confidence scores.

    Uses Monte Carlo Dropout and feature variance to estimate uncertainty.
    Higher uncertainty indicates harder routing decisions.
    """
    def __init__(self, in_channels, out_channels=128):
        super().__init__()

        # Feature extraction with dropout for uncertainty
        self.feature_net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # Spatial dropout for uncertainty
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d(1)
        )

        # Uncertainty head (estimates aleatoric + epistemic uncertainty)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()  # Output confidence in [0, 1]
        )

    def forward(self, x, num_samples=5):
        """
        Args:
            x: [B, C, H, W]
            num_samples: Number of MC dropout samples
        Returns:
            confidence: [B, 1] - Routing confidence scores
            uncertainty: [B, 1] - Routing uncertainty scores
        """
        B = x.shape[0]

        # Monte Carlo Dropout: multiple forward passes
        if self.training or num_samples > 1:
            samples = []
            for _ in range(num_samples):
                feat = self.feature_net(x).flatten(1)  # [B, out_channels]
                samples.append(feat)

            # Stack samples
            samples = torch.stack(samples, dim=0)  # [num_samples, B, out_channels]

            # Compute mean and variance
            mean_feat = samples.mean(dim=0)  # [B, out_channels]
            var_feat = samples.var(dim=0)    # [B, out_channels]

            # Uncertainty is the average variance across features
            uncertainty = var_feat.mean(dim=1, keepdim=True)  # [B, 1]

            # Confidence is inverse of uncertainty
            confidence = torch.exp(-uncertainty)  # Higher variance -> lower confidence

        else:
            # Single forward pass (inference)
            mean_feat = self.feature_net(x).flatten(1)
            confidence = torch.ones(B, 1, device=x.device)
            uncertainty = torch.zeros(B, 1, device=x.device)

        # Also estimate confidence from feature head
        predicted_confidence = self.uncertainty_head(mean_feat)

        # Combine both confidence estimates
        final_confidence = (confidence + predicted_confidence) / 2.0

        return final_confidence, uncertainty


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice routing mechanism.

    Instead of each token selecting top-k experts (Token Choice),
    each expert selects top-k tokens to process (Expert Choice).

    Benefits:
    - Better load balancing (each expert processes exactly capacity tokens)
    - Prevents expert collapse (underutilized experts)
    - More efficient for large batch sizes
    """
    def __init__(self, num_experts, expert_capacity_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor

    def forward(self, router_logits, num_tokens):
        """
        Args:
            router_logits: [B*H*W, num_experts] - Router scores for each token
            num_tokens: Total number of tokens (B*H*W)

        Returns:
            expert_assignments: [num_experts, capacity] - Token indices assigned to each expert
            expert_weights: [num_experts, capacity] - Weights for assigned tokens
            expert_mask: [num_experts, capacity] - Binary mask (1 if token assigned, 0 if padding)
        """
        # Compute capacity per expert
        capacity = int(num_tokens * self.expert_capacity_factor / self.num_experts)
        capacity = max(capacity, 1)  # At least 1 token per expert

        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]

        # Transpose to get scores from expert perspective
        expert_scores = router_probs.t()  # [num_experts, num_tokens]

        # Each expert selects its top-capacity tokens
        top_k_scores, top_k_indices = torch.topk(
            expert_scores,
            k=min(capacity, num_tokens),
            dim=1
        )  # [num_experts, capacity]

        # Normalize scores for selected tokens
        expert_weights = top_k_scores / (top_k_scores.sum(dim=1, keepdim=True) + 1e-8)

        # Create mask (1 for real tokens, 0 for padding)
        expert_mask = torch.ones_like(expert_weights)

        # If capacity > available tokens, mask padding
        if capacity > num_tokens:
            expert_mask[:, num_tokens:] = 0

        return top_k_indices, expert_weights, expert_mask


class SophisticatedRouter(nn.Module):
    """
    Advanced Multi-branch Router with Expert Choice routing.

    Features:
    1. Multi-scale texture analysis (dilations 1, 2, 4)
    2. Sobel-initialized edge detection
    3. FFT-based frequency analysis
    4. Pyramid pooling for context (1x1, 2x2, 3x3, 6x6)
    5. Uncertainty estimation
    6. Expert Choice routing
    7. Global-batch load balancing

    Args:
        backbone_dims: Feature dimensions from backbone [64, 128, 320, 512]
        num_experts: Number of expert models (default: 4)
        top_k: How many experts to use in Token Choice mode (default: 2)
        routing_mode: 'token_choice' or 'expert_choice'
        expert_capacity_factor: Capacity multiplier for Expert Choice (default: 1.25)
    """
    def __init__(
        self,
        backbone_dims=[64, 128, 320, 512],
        num_experts=4,
        top_k=2,
        routing_mode='token_choice',
        expert_capacity_factor=1.25
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.routing_mode = routing_mode
        self.expert_capacity_factor = expert_capacity_factor

        # Use highest resolution features for detailed analysis
        self.high_dim = backbone_dims[-1]  # 512

        # ============================================================
        # BRANCH 1: Multi-Scale Texture Analysis
        # Dilated convolutions with rates [1, 2, 4]
        # ============================================================
        self.texture_analyzer = MultiScaleTextureAnalyzer(
            in_channels=self.high_dim,
            out_channels=256
        )

        # ============================================================
        # BRANCH 2: Sobel Edge Density Analysis
        # Edge detection with Sobel-initialized kernels
        # ============================================================
        self.edge_analyzer = SobelEdgeAnalyzer(
            in_channels=self.high_dim,
            out_channels=256
        )

        # ============================================================
        # BRANCH 3: FFT Frequency Analysis
        # Analyzes high vs low frequency dominance
        # ============================================================
        self.frequency_analyzer = FFTFrequencyAnalyzer(
            in_channels=self.high_dim,
            out_channels=256
        )

        # ============================================================
        # BRANCH 4: Pyramid Pooling Context Analysis
        # Multi-scale context with pooling [1x1, 2x2, 3x3, 6x6]
        # ============================================================
        self.context_analyzer = PyramidPoolingAnalyzer(
            in_channels=self.high_dim,
            out_channels=256
        )

        # ============================================================
        # BRANCH 5: Uncertainty Estimation
        # Routing confidence scores
        # ============================================================
        self.uncertainty_estimator = UncertaintyEstimator(
            in_channels=self.high_dim,
            out_channels=128
        )

        # ============================================================
        # Global Feature Integration
        # Considers all feature scales
        # ============================================================
        self.multi_scale_integration = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, 64, 1)
            ) for dim in backbone_dims
        ])

        # ============================================================
        # Decision Network
        # Combines all branches to make routing decision
        # ============================================================
        # Total input: 256 (texture) + 256 (edge) + 256 (freq) + 256 (context) + (64*4) = 1280
        total_features = 256 + 256 + 256 + 256 + (64 * len(backbone_dims))

        self.decision_network = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_experts)
        )

        # Temperature parameter for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

        # Expert Choice router
        self.expert_choice_router = ExpertChoiceRouter(
            num_experts=num_experts,
            expert_capacity_factor=expert_capacity_factor
        )

        # Load balancing loss coefficient (increased 10x to prevent router collapse)
        self.load_balance_coef = 0.1

    def forward(self, features, return_uncertainty=False):
        """
        Args:
            features: List of backbone features [f1, f2, f3, f4]
                     where f4 is the highest-level feature
            return_uncertainty: If True, return uncertainty estimates

        Returns:
            If routing_mode == 'token_choice':
                expert_probs: [B, num_experts] - Probability distribution over experts
                top_k_indices: [B, top_k] - Indices of selected experts
                top_k_weights: [B, top_k] - Weights for selected experts (sum to 1)
                aux_outputs: Dictionary with additional outputs

            If routing_mode == 'expert_choice':
                router_logits: [B*H*W, num_experts] - Router scores for each token
                expert_assignments: [num_experts, capacity] - Token indices per expert
                expert_weights: [num_experts, capacity] - Weights for assigned tokens
                aux_outputs: Dictionary with additional outputs
        """
        B = features[-1].shape[0]
        highest_features = features[-1]  # [B, 512, H, W]

        # ============================================================
        # Extract characteristics from different branches
        # ============================================================

        # 1. Texture complexity
        texture_feat = self.texture_analyzer(highest_features)  # [B, 256]

        # 2. Edge density
        edge_feat = self.edge_analyzer(highest_features)  # [B, 256]

        # 3. Frequency content
        freq_feat = self.frequency_analyzer(highest_features)  # [B, 256]

        # 4. Context scale
        context_feat = self.context_analyzer(highest_features)  # [B, 256]

        # 5. Uncertainty estimation
        confidence, uncertainty = self.uncertainty_estimator(highest_features)  # [B, 1]

        # Integrate multi-scale features
        multi_scale_feats = []
        for feat, integrator in zip(features, self.multi_scale_integration):
            ms_feat = integrator(feat).view(B, -1)  # [B, 64]
            multi_scale_feats.append(ms_feat)
        multi_scale_feat = torch.cat(multi_scale_feats, dim=1)  # [B, 256]

        # ============================================================
        # Combine all branches
        # ============================================================
        combined = torch.cat([
            texture_feat,      # 256
            edge_feat,         # 256
            freq_feat,         # 256
            context_feat,      # 256
            multi_scale_feat   # 256
        ], dim=1)  # [B, 1280]

        # ============================================================
        # Make routing decision
        # ============================================================
        logits = self.decision_network(combined)  # [B, num_experts]

        # Apply temperature scaling
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        logits = logits / temp

        # Weight logits by confidence (uncertain samples get smoother routing)
        logits = logits * confidence

        # Prepare auxiliary outputs
        aux_outputs = {
            'confidence': confidence,
            'uncertainty': uncertainty,
            'texture_features': texture_feat,
            'edge_features': edge_feat,
            'frequency_features': freq_feat,
            'context_features': context_feat
        }

        # ============================================================
        # Route based on selected mode
        # ============================================================

        if self.routing_mode == 'expert_choice':
            # Expert Choice routing: Experts select tokens
            H, W = highest_features.shape[2:]
            num_tokens = B * H * W

            # Reshape features to token format
            # [B, num_experts] -> [B*H*W, num_experts] by repeating
            # In practice, you'd compute per-token logits, but for image-level routing,
            # we use the same logits for all tokens in an image
            token_logits = logits.unsqueeze(1).unsqueeze(2).expand(B, H, W, -1)
            token_logits = token_logits.reshape(-1, self.num_experts)  # [B*H*W, num_experts]

            # Expert Choice routing
            expert_assignments, expert_weights, expert_mask = self.expert_choice_router(
                token_logits, num_tokens
            )

            # Compute load balancing loss
            router_probs = F.softmax(logits, dim=1)
            load_balance_loss = self._compute_load_balance_loss(router_probs)

            aux_outputs.update({
                'expert_assignments': expert_assignments,
                'expert_mask': expert_mask,
                'load_balance_loss': load_balance_loss,
                'router_logits': token_logits
            })

            return logits, expert_assignments, expert_weights, aux_outputs

        else:
            # Token Choice routing: Tokens select experts (traditional)
            expert_probs = F.softmax(logits, dim=1)  # [B, num_experts]

            # Select top-k experts
            top_k_probs, top_k_indices = torch.topk(expert_probs, self.top_k, dim=1)

            # Renormalize top-k probabilities
            top_k_weights = top_k_probs / (top_k_probs.sum(dim=1, keepdim=True) + 1e-8)

            # Compute load balancing loss
            load_balance_loss = self._compute_load_balance_loss(expert_probs)

            aux_outputs.update({
                'load_balance_loss': load_balance_loss,
                'router_logits': logits
            })

            return expert_probs, top_k_indices, top_k_weights, aux_outputs

    def compute_entropy_bonus(self, probs):
        """
        Encourage routing entropy > 1.0 for diverse expert usage.

        Higher entropy = more diverse routing = less expert collapse.

        Args:
            probs: [B, num_experts] - Probability distribution
        Returns:
            entropy_bonus: Scalar tensor (penalty when entropy < target)
        """
        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        # Target entropy for diverse routing
        # For 3 experts: log(3) ≈ 1.1, so target 1.0 means reasonably diverse
        target_entropy = 1.0

        # Penalize when entropy drops below target (encourages diversity)
        entropy_bonus = F.relu(target_entropy - entropy)

        return entropy_bonus

    def _compute_load_balance_loss(self, expert_probs):
        """
        Compute global-batch load balancing loss to prevent expert collapse.

        Encourages uniform expert usage across the batch.
        NOW INCLUDES entropy regularization for diverse routing.

        Args:
            expert_probs: [B, num_experts]
        Returns:
            load_balance_loss: Scalar tensor
        """
        # Average probability for each expert across the batch
        mean_probs = expert_probs.mean(dim=0)  # [num_experts]

        # Ideal uniform distribution
        ideal_prob = 1.0 / self.num_experts

        # L2 distance from uniform distribution
        load_balance_loss = ((mean_probs - ideal_prob) ** 2).sum()

        # Also penalize coefficient of variation (relative std)
        cv = mean_probs.std() / (mean_probs.mean() + 1e-8)

        # Add entropy bonus to encourage diverse routing
        entropy_loss = self.compute_entropy_bonus(expert_probs)

        # Combined loss with entropy regularization (weight 0.05)
        total_loss = self.load_balance_coef * (load_balance_loss + cv + 0.05 * entropy_loss)

        return total_loss

    def get_expert_usage_stats(self, expert_probs):
        """
        Analyze which experts are being used.

        Args:
            expert_probs: [B, num_experts]

        Returns:
            Dictionary with usage statistics
        """
        with torch.no_grad():
            avg_probs = expert_probs.mean(dim=0)  # [num_experts]
            max_expert = expert_probs.argmax(dim=1)  # [B]

            stats = {
                'avg_expert_probs': avg_probs.cpu().numpy(),
                'expert_selection_counts': torch.bincount(
                    max_expert,
                    minlength=self.num_experts
                ).cpu().numpy(),
                'entropy': -(expert_probs * torch.log(expert_probs + 1e-8)).sum(dim=1).mean().item(),
                'coefficient_of_variation': (avg_probs.std() / (avg_probs.mean() + 1e-8)).item(),
                'max_prob': avg_probs.max().item(),
                'min_prob': avg_probs.min().item()
            }

        return stats

    def set_routing_mode(self, mode):
        """Switch between 'token_choice' and 'expert_choice' routing"""
        assert mode in ['token_choice', 'expert_choice']
        self.routing_mode = mode


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("="*70)
    print("Testing Upgraded Sophisticated Router")
    print("="*70)

    # Test Token Choice routing
    print("\n" + "="*70)
    print("Test 1: Token Choice Routing")
    print("="*70)

    router_tc = SophisticatedRouter(
        backbone_dims=[64, 128, 320, 512],
        num_experts=4,
        top_k=2,
        routing_mode='token_choice'
    )

    print(f"Router parameters: {count_parameters(router_tc) / 1e6:.2f}M")

    # Create dummy features
    features = [
        torch.randn(2, 64, 112, 112),
        torch.randn(2, 128, 56, 56),
        torch.randn(2, 320, 28, 28),
        torch.randn(2, 512, 14, 14)
    ]

    expert_probs, top_k_indices, top_k_weights, aux = router_tc(features)

    print(f"\nOutput shapes:")
    print(f"  Expert probabilities: {expert_probs.shape}")
    print(f"  Top-k indices: {top_k_indices.shape}")
    print(f"  Top-k weights: {top_k_weights.shape}")
    print(f"\nAuxiliary outputs:")
    print(f"  Confidence: {aux['confidence'].shape}")
    print(f"  Uncertainty: {aux['uncertainty'].shape}")
    print(f"  Load balance loss: {aux['load_balance_loss'].item():.6f}")

    print(f"\nExample routing:")
    print(f"  Expert probs: {expert_probs[0]}")
    print(f"  Selected experts: {top_k_indices[0]}")
    print(f"  Expert weights: {top_k_weights[0]}")
    print(f"  Weights sum to: {top_k_weights[0].sum():.4f}")
    print(f"  Confidence: {aux['confidence'][0].item():.4f}")
    print(f"  Uncertainty: {aux['uncertainty'][0].item():.4f}")

    stats = router_tc.get_expert_usage_stats(expert_probs)
    print(f"\nUsage statistics:")
    print(f"  Average expert probs: {stats['avg_expert_probs']}")
    print(f"  Routing entropy: {stats['entropy']:.3f}")
    print(f"  Coefficient of variation: {stats['coefficient_of_variation']:.3f}")

    # Test Expert Choice routing
    print("\n" + "="*70)
    print("Test 2: Expert Choice Routing")
    print("="*70)

    router_ec = SophisticatedRouter(
        backbone_dims=[64, 128, 320, 512],
        num_experts=4,
        top_k=2,
        routing_mode='expert_choice',
        expert_capacity_factor=1.25
    )

    logits, expert_assignments, expert_weights, aux = router_ec(features)

    print(f"\nOutput shapes:")
    print(f"  Router logits: {logits.shape}")
    print(f"  Expert assignments: {expert_assignments.shape}")
    print(f"  Expert weights: {expert_weights.shape}")
    print(f"\nAuxiliary outputs:")
    print(f"  Expert mask: {aux['expert_mask'].shape}")
    print(f"  Load balance loss: {aux['load_balance_loss'].item():.6f}")

    print(f"\nExpert Choice details:")
    for i in range(4):
        num_assigned = aux['expert_mask'][i].sum().item()
        print(f"  Expert {i}: {int(num_assigned)} tokens assigned")

    # Test dynamic mode switching
    print("\n" + "="*70)
    print("Test 3: Dynamic Mode Switching")
    print("="*70)

    router = SophisticatedRouter(
        backbone_dims=[64, 128, 320, 512],
        num_experts=4,
        top_k=2
    )

    print(f"Initial mode: {router.routing_mode}")

    # Switch to expert choice
    router.set_routing_mode('expert_choice')
    print(f"Switched to: {router.routing_mode}")
    output_ec = router(features)
    print(f"  Output length (Expert Choice): {len(output_ec)}")

    # Switch back to token choice
    router.set_routing_mode('token_choice')
    print(f"Switched to: {router.routing_mode}")
    output_tc = router(features)
    print(f"  Output length (Token Choice): {len(output_tc)}")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)

    print("\nKey Features:")
    print("  ✓ Multi-scale texture analysis (dilations 1, 2, 4)")
    print("  ✓ Sobel-initialized edge detection")
    print("  ✓ FFT-based frequency analysis")
    print("  ✓ Pyramid pooling context (1x1, 2x2, 3x3, 6x6)")
    print("  ✓ Uncertainty estimation with confidence scores")
    print("  ✓ Expert Choice routing")
    print("  ✓ Global-batch load balancing")
    print("  ✓ Dynamic routing mode switching")
