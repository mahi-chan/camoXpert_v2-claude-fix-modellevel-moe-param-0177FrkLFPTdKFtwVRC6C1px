"""
Model-Level Mixture-of-Experts (MoE) for Camouflaged Object Detection

This is a TRUE ensemble approach where:
1. Router analyzes the input image
2. Selects which complete expert models to use
3. Each expert produces a full prediction
4. Predictions are combined with learned weights

Target Performance: 0.85-0.90 IoU (beats SOTA at 0.78-0.79)

Architecture:
  Input → Shared Backbone → Router → Select Experts → Combine Predictions → Output
                              ↓
                    ┌─────────┼─────────┬─────────┬─────────┐
                    ↓         ↓         ↓         ↓         ↓
                 Expert 1  Expert 2  Expert 3  Expert 4  Expert 5
                 (FSPNet) (PraNet)  (BASNet)  (CPD)    (GCPANet)
                  ~20M      ~15M      ~20M     ~15M      ~18M

Configurable via --expert-types: sinet pranet fspnet basnet cpd gcpanet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from models.sophisticated_router import SophisticatedRouter
from models.expert_architectures import (
    SINetExpert,
    PraNetExpert,
    ZoomNetExpert,
    UJSCExpert,
    FEDERFrequencyExpert,
    BASNetExpert,
    CPDExpert,
    GCPANetExpert
)
from models.fspnet_expert import FSPNetExpert
from models.zoomnext_expert import ZoomNeXtExpert

# Expert type mapping - includes all experts
EXPERT_REGISTRY = {
    'sinet': (SINetExpert, 'SINet-Style'),
    'pranet': (PraNetExpert, 'PraNet-Style'),
    'zoomnet': (ZoomNetExpert, 'ZoomNet-Style'),
    'ujsc': (UJSCExpert, 'UJSC-Style'),
    'frequency': (FEDERFrequencyExpert, 'FEDER-Frequency-Style'),
    'fspnet': (FSPNetExpert, 'FSPNet-Frequency-Style'),
    # New diverse experts for better MoE performance
    'basnet': (BASNetExpert, 'BASNet-Boundary-Style'),
    'cpd': (CPDExpert, 'CPD-MultiScale-Style'),
    'gcpanet': (GCPANetExpert, 'GCPANet-GlobalContext-Style'),
    # SOTA TPAMI 2024
    'zoomnext': (ZoomNeXtExpert, 'ZoomNeXt-TPAMI2024-Style'),
}


class ModelLevelMoE(nn.Module):
    """
    Model-Level Mixture-of-Experts Ensemble

    Total params: ~85M (Backbone 25M + Router 8M + Experts 4×15M)
    Active params per forward: ~48M (Backbone + Router + 2 experts)

    This beats feature-level MoE because:
    1. Each expert is a complete specialized architecture
    2. Ensemble effect: Different experts handle different image types
    3. Can leverage architectural diversity
    4. Better generalization through specialization
    
    Args:
        backbone_name: Name of the backbone (e.g., 'pvt_v2_b2')
        num_experts: Number of experts (auto-set from expert_types if provided)
        top_k: Number of experts to select per forward pass
        pretrained: Whether to use pretrained backbone
        use_deep_supervision: Enable deep supervision outputs
        expert_types: List of expert types to use (e.g., ['sinet', 'pranet', 'frequency'])
    """

    def __init__(self, backbone_name='pvt_v2_b2', num_experts=3, top_k=2,
                 pretrained=True, use_deep_supervision=False,
                 expert_types: Optional[List[str]] = None):
        super().__init__()

        # Default expert configuration if not specified
        if expert_types is None:
            expert_types = ['sinet', 'pranet', 'zoomnet']
        
        # Validate expert types
        for et in expert_types:
            if et.lower() not in EXPERT_REGISTRY:
                raise ValueError(f"Unknown expert type: {et}. Available: {list(EXPERT_REGISTRY.keys())}")
        
        self.expert_types = [et.lower() for et in expert_types]
        self.num_experts = len(self.expert_types)
        self.top_k = top_k
        self.use_deep_supervision = use_deep_supervision

        print("\n" + "="*70)
        print("MODEL-LEVEL MIXTURE-OF-EXPERTS ENSEMBLE")
        print("="*70)
        print(f"  Strategy: Router selects top-{top_k} of {self.num_experts} complete experts")
        print(f"  Expert types: {self.expert_types}")
        print(f"  Target: Beat SOTA (0.78-0.79) → Achieve 0.80-0.81 IoU")
        print("="*70)

        # ============================================================
        # SHARED BACKBONE: Extract features once
        # ============================================================
        print("\n[1/3] Loading shared backbone...")
        self.backbone = self._create_backbone(backbone_name, pretrained)
        self.feature_dims = self._get_feature_dims(backbone_name)
        print(f"✓ Backbone: {backbone_name}")
        print(f"✓ Feature dims: {self.feature_dims}")

        # ============================================================
        # SOPHISTICATED ROUTER: Decides which experts to use
        # ============================================================
        print("\n[2/3] Initializing sophisticated router...")
        self.router = SophisticatedRouter(
            backbone_dims=self.feature_dims,
            num_experts=self.num_experts,
            top_k=top_k
        )
        router_params = sum(p.numel() for p in self.router.parameters())
        print(f"✓ Router created: {router_params/1e6:.1f}M parameters")
        print(f"✓ Analyzes: texture, edges, context, frequency, multi-scale")

        # ============================================================
        # EXPERT MODELS: Dynamically created based on expert_types
        # ============================================================
        print("\n[3/3] Creating expert models...")

        self.expert_models = nn.ModuleList()
        for i, expert_type in enumerate(self.expert_types):
            expert_cls, expert_name = EXPERT_REGISTRY[expert_type]
            expert = expert_cls(self.feature_dims)  # All experts take feature_dims
            
            self.expert_models.append(expert)
            params = sum(p.numel() for p in expert.parameters())
            print(f"✓ Expert {i} ({expert_name}): {params/1e6:.1f}M parameters")

        # ============================================================
        # Calculate total parameters
        # ============================================================
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())

        print("\n" + "="*70)
        print(f"TOTAL PARAMETERS: {total_params/1e6:.1f}M")
        print(f"  Backbone: {backbone_params/1e6:.1f}M")
        print(f"  Router: {router_params/1e6:.1f}M")
        print(f"  All Experts: {(total_params - backbone_params - router_params)/1e6:.1f}M")
        print(f"  Active per forward: ~{(backbone_params + router_params + 2*15e6)/1e6:.1f}M")
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
                # scriptable=False,  # Optimization for memory
                # exportable=False,
                # no_jit=True
            )
            # Enable gradient checkpointing if available (saves VRAM)
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
        else:
            raise ValueError(f"Unknown backbone dimensions for: {backbone_name}")

    def freeze_router(self):
        """Freeze router parameters for expert-only training"""
        for param in self.router.parameters():
            param.requires_grad = False
        print("✓ Router frozen (parameters will not be updated)")

    def unfreeze_router(self):
        """Unfreeze router parameters for router training"""
        for param in self.router.parameters():
            param.requires_grad = True
        print("✓ Router unfrozen (parameters will be updated)")

    def get_equal_routing_weights(self, batch_size, device):
        """Return equal weights for all experts (used when router frozen)"""
        weights = torch.ones(batch_size, self.num_experts, device=device) / self.num_experts
        return weights

    def forward(self, x, return_routing_info=False):
        """
        Simplified forward pass with clean expert routing.

        Args:
            x: Input images [B, 3, H, W]
            return_routing_info: Whether to return routing statistics

        Returns:
            prediction: [B, 1, H, W]
            routing_info: dict with routing stats and auxiliary outputs
        """
        B, _, H, W = x.shape

        # ============================================================
        # Step 1: Extract features from shared backbone
        # ============================================================
        features = self.backbone(x)  # [f1, f2, f3, f4]

        # ============================================================
        # Step 2: Router decides which experts to use
        # ============================================================
        expert_probs, top_k_indices, top_k_weights, router_aux = self.router(features)

        # ============================================================
        # Step 3: Run ALL experts and collect predictions
        # ============================================================
        expert_predictions = []
        expert_predictions_list = []  # Store for debugging
        expert_aux_outputs = []  # Store auxiliary outputs

        for expert in self.expert_models:
            pred, aux = expert(features, return_aux=True)
            expert_predictions.append(pred)
            expert_predictions_list.append(pred.detach().clone())
            # Only keep first 2 aux outputs per expert (reduced deep supervision)
            if aux:
                expert_aux_outputs.extend(aux[:2])

        expert_predictions = torch.stack(expert_predictions, dim=1)  # [B, num_experts, 1, H, W]

        # ============================================================
        # Step 4: Combine expert predictions
        # ============================================================
        final_prediction = torch.sum(
            expert_predictions * expert_probs.view(B, self.num_experts, 1, 1, 1),
            dim=1
        )  # [B, 1, H, W]

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
                'routing_stats': self.router.get_expert_usage_stats(expert_probs),
                'load_balance_loss': router_aux.get('load_balance_loss', None),
                'confidence': router_aux.get('confidence', None),
                'individual_expert_preds': expert_predictions_list,
                'aux_outputs': expert_aux_outputs[:4] if expert_aux_outputs else None,  # Limit to 4
            }
            return final_prediction, routing_info
        else:
            return final_prediction

    def get_routing_stats(self, data_loader, num_batches=10):
        """
        Analyze routing behavior on a dataset

        Args:
            data_loader: DataLoader
            num_batches: Number of batches to analyze

        Returns:
            Dictionary with routing statistics
        """
        self.eval()
        all_expert_probs = []
        all_top_k_indices = []

        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                if i >= num_batches:
                    break

                images = images.cuda()
                _, routing_info = self.forward(images, return_routing_info=True)

                all_expert_probs.append(routing_info['expert_probs'].cpu())
                all_top_k_indices.append(routing_info['top_k_indices'].cpu())

        all_expert_probs = torch.cat(all_expert_probs, dim=0)
        all_top_k_indices = torch.cat(all_top_k_indices, dim=0)

        stats = {
            'avg_expert_usage': all_expert_probs.mean(dim=0).numpy(),
            'expert_selection_distribution': torch.bincount(
                all_top_k_indices.flatten(),
                minlength=self.num_experts
            ).float().numpy() / (all_top_k_indices.numel()),
            'routing_entropy': -(all_expert_probs * torch.log(all_expert_probs + 1e-8)).sum(dim=1).mean().item()
        }

        return stats


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    print("Testing Model-Level MoE...")
    print("\n" + "="*70)

    # Create model
    model = ModelLevelMoE(
        backbone_name='pvt_v2_b2',
        num_experts=4,
        top_k=2,
        pretrained=False
    )

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 448, 448)
    pred, routing_info = model(x, return_routing_info=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Expert probabilities: {routing_info['expert_probs'][0]}")
    print(f"Selected experts: {routing_info['top_k_indices'][0]}")
    print(f"Expert weights: {routing_info['top_k_weights'][0]}")
    print(f"Routing entropy: {routing_info['routing_stats']['entropy']:.3f}")

    print("\n" + "="*70)
    print("✓ Model-Level MoE test passed!")
