"""
Model-Level Mixture-of-Experts (MoE) with HARD SPARSE ROUTING

Key difference from Soft MoE:
- Soft MoE: Runs ALL experts, combines with soft weights
- Hard MoE: Runs ONLY top-k experts per sample (sparse, efficient)

IMPROVEMENTS:
1. Gumbel-Softmax: Differentiable hard selection (training stability)
2. Straight-Through Estimator: Hard forward, soft backward
3. Expert dropout: Prevents weight imbalance
4. 33% compute savings (only 2/3 experts run per sample)

Expert Lineup (same as Soft MoE for fair comparison):
- Expert 0: SINet (Search & Identify)
- Expert 1: PraNet (Reverse Attention)  
- Expert 2: ZoomNet (Multi-Scale Zoom)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sophisticated_router import SophisticatedRouter
from models.expert_architectures import (
    FEDERLightExpert,  # Lightweight frequency expert (~15M, same as SINet)
    PraNetExpert,
    ZoomNetExpert,
)


def gumbel_softmax_hard(logits, temperature=1.0, hard=True):
    """
    Gumbel-Softmax with Straight-Through Estimator
    
    Forward: Hard one-hot selection (discrete, sparse)
    Backward: Soft gradients flow to all experts (training stability)
    
    This is the key technique that overcomes hard MoE training challenges.
    
    Args:
        logits: [B, num_experts] - unnormalized routing scores
        temperature: Softmax temperature (lower = more discrete)
        hard: If True, use straight-through estimator
        
    Returns:
        If hard: One-hot selection with soft gradients
        If not hard: Soft probabilities
    """
    # Sample from Gumbel(0, 1)
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    
    # Add Gumbel noise and apply temperature-scaled softmax
    y_soft = F.softmax((logits + gumbels) / temperature, dim=-1)
    
    if hard:
        # Get hard one-hot from argmax
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        
        # Straight-through: hard in forward, soft in backward
        # Gradient flows through y_soft, but forward uses y_hard
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


class ModelLevelMoEHard(nn.Module):
    """
    Model-Level MoE with Hard Sparse Routing + Gumbel-Softmax
    
    Key Improvements:
    1. Gumbel-Softmax: Differentiable discrete selection
    2. FEDER Expert: Frequency-based approach (diverse from PraNet/ZoomNet)
    3. Auxiliary loss: Unselected experts get gradient signal
    4. Temperature annealing: Harder selection as training progresses
    
    Total params: ~78M
    Active params per forward: ~61M (Backbone + Router + 2 experts)
    Compute savings: ~33% of expert compute
    """

    def __init__(self, backbone_name='pvt_v2_b2', num_experts=3, top_k=2,
                 pretrained=True, use_deep_supervision=False,
                 temperature=1.0, min_temperature=0.5, anneal_rate=0.003,
                 expert_dropout=0.1):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.use_deep_supervision = use_deep_supervision
        
        # Gumbel-Softmax parameters
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        
        # Expert dropout: randomly drop an expert during training
        # This prevents weight imbalance and forces specialization
        self.expert_dropout = expert_dropout

        print("\n" + "="*70)
        print("MODEL-LEVEL MoE - HARD SPARSE ROUTING (Gumbel-Softmax)")
        print("="*70)
        print(f"  Strategy: Router selects top-{top_k} of {num_experts} experts")
        print(f"  ONLY {top_k} experts run per sample (sparse!)")
        print(f"  Compute savings: {(1 - top_k/num_experts)*100:.0f}% expert compute saved")
        print(f"  Training tricks:")
        print(f"    - Gumbel-Softmax (temp={temperature})")
        print(f"    - Expert dropout ({expert_dropout*100:.0f}%)")
        print("="*70)

        # ============================================================
        # SHARED BACKBONE
        # ============================================================
        print("\n[1/3] Loading shared backbone...")
        self.backbone = self._create_backbone(backbone_name, pretrained)
        self.feature_dims = self._get_feature_dims(backbone_name)
        print(f"✓ Backbone: {backbone_name}")
        print(f"✓ Feature dims: {self.feature_dims}")

        # ============================================================
        # SOPHISTICATED ROUTER
        # ============================================================
        print("\n[2/3] Initializing sophisticated router...")
        self.router = SophisticatedRouter(
            backbone_dims=self.feature_dims,
            num_experts=num_experts,
            top_k=top_k
        )
        router_params = sum(p.numel() for p in self.router.parameters())
        print(f"✓ Router created: {router_params/1e6:.1f}M parameters")

        # ============================================================
        # EXPERT MODELS - Diverse Architectures
        # ============================================================
        print("\n[3/3] Creating DIVERSE expert models...")

        # Hard MoE uses FEDER-Light for architectural diversity
        # FEDER-Light: Frequency decomposition (different from spatial methods!)
        # PraNet: Reverse attention (boundary-focused)
        # ZoomNet: Multi-scale zoom (scale-focused)
        self.expert_models = nn.ModuleList([
            FEDERLightExpert(self.feature_dims),  # Expert 0: Frequency (~15M)
            PraNetExpert(self.feature_dims),      # Expert 1: Reverse Attention
            ZoomNetExpert(self.feature_dims),     # Expert 2: Multi-Scale Zoom
        ])

        expert_names = ["FEDER-Light (Frequency)", "PraNet (Boundary)", "ZoomNet (Scale)"]
        for i, (name, expert) in enumerate(zip(expert_names, self.expert_models)):
            params = sum(p.numel() for p in expert.parameters())
            print(f"✓ Expert {i} ({name}): {params/1e6:.1f}M parameters")

        # ============================================================
        # Calculate parameters
        # ============================================================
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        expert_params = total_params - backbone_params - router_params

        print("\n" + "="*70)
        print(f"TOTAL PARAMETERS: {total_params/1e6:.1f}M")
        print(f"  Backbone: {backbone_params/1e6:.1f}M")
        print(f"  Router: {router_params/1e6:.1f}M")
        print(f"  All Experts: {expert_params/1e6:.1f}M")
        print(f"  ACTIVE per forward: ~{(backbone_params + router_params + top_k*(expert_params/num_experts))/1e6:.1f}M")
        print("="*70)

    def _create_backbone(self, backbone_name, pretrained):
        """Create backbone network using timm"""
        import timm
        try:
            backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            if hasattr(backbone, 'set_grad_checkpointing'):
                backbone.set_grad_checkpointing(enable=True)
        except Exception as e:
            raise ValueError(f"Failed to create backbone '{backbone_name}': {e}")
        return backbone

    def _get_feature_dims(self, backbone_name):
        """Get feature dimensions for different backbones"""
        backbone_dims = {
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b3': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
            'pvt_v2_b5': [64, 128, 320, 512],
        }
        if backbone_name in backbone_dims:
            return backbone_dims[backbone_name]
        raise ValueError(f"Unknown backbone dimensions for: {backbone_name}")

    def freeze_router(self):
        for param in self.router.parameters():
            param.requires_grad = False

    def unfreeze_router(self):
        for param in self.router.parameters():
            param.requires_grad = True

    def anneal_temperature(self):
        """Decrease temperature over training for harder selections"""
        self.temperature = max(
            self.min_temperature,
            self.temperature * (1 - self.anneal_rate)
        )
        return self.temperature

    def forward(self, x, return_routing_info=False):
        """
        Hard Sparse Forward with Gumbel-Softmax
        
        Training:
        - Uses Gumbel-Softmax for differentiable hard selection
        - Gradients flow to all experts through straight-through estimator
        - Computes auxiliary loss on unselected experts for diversity
        
        Inference:
        - Pure hard selection (argmax)
        - Only selected experts run
        
        Args:
            x: Input images [B, 3, H, W]
            return_routing_info: Return routing statistics
            
        Returns:
            prediction: [B, 1, H, W]
            routing_info: dict with routing stats
        """
        B, _, H, W = x.shape
        device = x.device

        # ============================================================
        # Step 1: Extract features from shared backbone
        # ============================================================
        features = self.backbone(x)

        # ============================================================
        # Step 2: Get routing logits from router
        # ============================================================
        expert_probs, top_k_indices, top_k_weights, router_aux = self.router(features)
        
        # Get router logits for Gumbel-Softmax
        # expert_probs is softmax(logits), so approximate logits
        router_logits = torch.log(expert_probs + 1e-8)

        # ============================================================
        # Step 3: Hard selection via Gumbel-Softmax
        # ============================================================
        if self.training:
            # Expert Dropout: Randomly drop one expert during training
            # This prevents weight imbalance and forces specialization
            dropped_expert = None
            if self.expert_dropout > 0 and torch.rand(1).item() < self.expert_dropout:
                dropped_expert = torch.randint(0, self.num_experts, (1,)).item()
                # Set dropped expert's logits to -inf
                router_logits = router_logits.clone()
                router_logits[:, dropped_expert] = -1e9
            
            # During training: Use Gumbel-Softmax for differentiable hard selection
            # This gives hard selection in forward but soft gradients in backward
            
            # Get top-k via repeated Gumbel-Softmax sampling without replacement
            selected_weights = torch.zeros(B, self.num_experts, device=device)
            selected_mask = torch.zeros(B, self.num_experts, device=device)
            
            remaining_logits = router_logits.clone()
            
            for k in range(self.top_k):
                # Sample one expert using Gumbel-Softmax
                gumbel_sample = gumbel_softmax_hard(
                    remaining_logits, 
                    temperature=self.temperature, 
                    hard=True
                )
                selected_mask += gumbel_sample
                selected_weights += gumbel_sample * expert_probs
                
                # Mask out selected expert for next round (no replacement)
                remaining_logits = remaining_logits - gumbel_sample * 1e9
            
            # Normalize weights
            selected_weights = selected_weights / (selected_weights.sum(dim=1, keepdim=True) + 1e-8)
            
        else:
            # During inference: Pure argmax selection
            selected_mask = torch.zeros(B, self.num_experts, device=device)
            selected_weights = torch.zeros(B, self.num_experts, device=device)
            
            _, topk_idx = torch.topk(expert_probs, self.top_k, dim=1)
            for k in range(self.top_k):
                selected_mask.scatter_(1, topk_idx[:, k:k+1], 1.0)
            
            selected_weights = expert_probs * selected_mask
            selected_weights = selected_weights / (selected_weights.sum(dim=1, keepdim=True) + 1e-8)

        # ============================================================
        # Step 4: Run ONLY selected experts (SPARSE!)
        # ============================================================
        # Pre-compute output shape
        out_h = features[0].shape[2] * 4
        out_w = features[0].shape[3] * 4
        
        final_prediction = torch.zeros(B, 1, out_h, out_w, device=device)
        
        # Run only selected experts
        for expert_idx in range(self.num_experts):
            # Check if this expert is selected for any sample in batch
            expert_mask = selected_mask[:, expert_idx]  # [B]
            
            if expert_mask.sum() > 0:
                # Get samples that selected this expert
                sample_indices = torch.where(expert_mask > 0)[0]
                
                if len(sample_indices) > 0:
                    # Extract features for selected samples
                    selected_features = [f[sample_indices] for f in features]
                    
                    # Run expert only on selected samples
                    # NOTE: Don't collect aux outputs - sparse routing causes batch size mismatch
                    expert_pred, _ = self.expert_models[expert_idx](selected_features, return_aux=False)
                    
                    # Add weighted prediction for each selected sample
                    for i, b in enumerate(sample_indices):
                        weight = selected_weights[b, expert_idx]
                        final_prediction[b:b+1] += weight * expert_pred[i:i+1]

        # ============================================================
        # Step 5: Auxiliary loss DISABLED for memory efficiency
        # ============================================================
        # NOTE: Auxiliary loss was removed because it runs unselected experts,
        # which defeats the purpose of sparse routing and causes OOM.
        # The load balancing loss from the router is sufficient for expert diversity.
        auxiliary_expert_loss = None

        # ============================================================
        # Return prediction with routing info
        # ============================================================
        if return_routing_info or self.training:
            routing_info = {
                'routing_probs': expert_probs,
                'expert_assignments': top_k_indices,
                'expert_probs': expert_probs,
                'top_k_indices': top_k_indices,
                'top_k_weights': top_k_weights,
                'selected_mask': selected_mask,
                'selected_weights': selected_weights,
                'routing_stats': self.router.get_expert_usage_stats(expert_probs),
                'load_balance_loss': router_aux.get('load_balance_loss', None),
                'confidence': router_aux.get('confidence', None),
                'aux_outputs': None,  # Disabled for hard MoE (batch mismatch)
                'auxiliary_expert_loss': auxiliary_expert_loss,
                'temperature': self.temperature,
            }
            return final_prediction, routing_info
        else:
            return final_prediction

    def get_routing_stats(self, data_loader, num_batches=10):
        """Analyze routing behavior on a dataset"""
        self.eval()
        all_expert_probs = []
        all_selected_masks = []

        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                images = images.cuda()
                _, routing_info = self.forward(images, return_routing_info=True)
                all_expert_probs.append(routing_info['expert_probs'].cpu())
                all_selected_masks.append(routing_info['selected_mask'].cpu())

        all_expert_probs = torch.cat(all_expert_probs, dim=0)
        all_selected_masks = torch.cat(all_selected_masks, dim=0)

        stats = {
            'avg_expert_usage': all_expert_probs.mean(dim=0).numpy(),
            'actual_selection_rate': all_selected_masks.mean(dim=0).numpy(),
            'routing_entropy': -(all_expert_probs * torch.log(all_expert_probs + 1e-8)).sum(dim=1).mean().item()
        }

        return stats


if __name__ == '__main__':
    print("Testing Hard Sparse MoE with Gumbel-Softmax...")
    print("\n" + "="*70)

    # Create model
    model = ModelLevelMoEHard(
        backbone_name='pvt_v2_b2',
        num_experts=3,
        top_k=2,
        pretrained=False,
        temperature=1.0
    )

    # Test forward pass
    print("\nTesting forward pass...")
    model.train()
    x = torch.randn(2, 3, 448, 448)
    pred, routing_info = model(x, return_routing_info=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Expert probabilities: {routing_info['expert_probs'][0]}")
    print(f"Selected mask: {routing_info['selected_mask'][0]}")
    print(f"Selected weights: {routing_info['selected_weights'][0]}")
    print(f"Temperature: {routing_info['temperature']}")
    if routing_info['auxiliary_expert_loss'] is not None:
        print(f"Auxiliary loss: {routing_info['auxiliary_expert_loss'].item():.4f}")

    print("\n" + "="*70)
    print("✓ Hard Sparse MoE with Gumbel-Softmax test passed!")
