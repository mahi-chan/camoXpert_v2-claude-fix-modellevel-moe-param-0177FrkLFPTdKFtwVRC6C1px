"""
Exponential Moving Average (EMA) for Model Weights

EMA maintains a moving average of model parameters during training,
which typically provides better generalization and more stable predictions.

Usage:
    model = YourModel()
    ema = EMA(model, decay=0.999)

    # During training
    for batch in dataloader:
        loss = train_step(model, batch)
        optimizer.step()
        ema.update()  # Update EMA weights

    # During validation/inference
    ema.apply_shadow()  # Apply EMA weights
    metrics = validate(model)
    ema.restore()  # Restore original weights for continued training
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average for model parameters.

    Maintains shadow copies of model weights and updates them as:
        shadow_weight = decay * shadow_weight + (1 - decay) * current_weight

    Args:
        model: PyTorch model
        decay: EMA decay rate (default: 0.999)
               Higher values = smoother averaging, slower adaptation
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0

        # Register shadow parameters
        self.register()

    def register(self):
        """
        Register all trainable parameters to track in EMA.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update EMA shadow weights after each training step.

        Should be called after optimizer.step()
        """
        self.num_updates += 1

        # Adaptive decay (optional, commented out for now)
        # decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in shadow!"
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Apply EMA shadow weights to the model.

        Use this before validation/inference to use smoothed weights.
        Call restore() afterwards to continue training with original weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in shadow!"
                # Backup current weights
                self.backup[name] = param.data.clone()
                # Apply shadow weights
                param.data = self.shadow[name].clone()

    def restore(self):
        """
        Restore original model weights after using EMA weights.

        Call this after validation to continue training with original weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup, f"Parameter {name} not found in backup!"
                param.data = self.backup[name].clone()

        # Clear backup to free memory
        self.backup = {}

    def state_dict(self):
        """
        Get state dict for saving EMA to checkpoint.

        Returns:
            Dictionary with shadow parameters and metadata
        """
        return {
            'shadow': self.shadow,
            'decay': self.decay,
            'num_updates': self.num_updates
        }

    def load_state_dict(self, state_dict):
        """
        Load EMA state from checkpoint.

        Args:
            state_dict: Dictionary from EMA.state_dict()
        """
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)
        self.num_updates = state_dict.get('num_updates', 0)

    def copy_to_model(self):
        """
        Permanently copy EMA weights to model (for final deployment).

        Unlike apply_shadow(), this doesn't keep backups.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in shadow!"
                param.data = self.shadow[name].clone()


# Test EMA
if __name__ == '__main__':
    print("Testing EMA...")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()

    # Create EMA
    ema = EMA(model, decay=0.9)

    print(f"\nInitial weight: {model.fc.weight[0, 0].item():.6f}")

    # Simulate training updates
    for i in range(5):
        # Modify weights
        with torch.no_grad():
            model.fc.weight[0, 0] = torch.tensor(i * 0.1)

        # Update EMA
        ema.update()

        print(f"Step {i+1}: Current={model.fc.weight[0, 0].item():.6f}, "
              f"EMA={ema.shadow['fc.weight'][0, 0].item():.6f}")

    # Test apply and restore
    print("\nTesting apply_shadow() and restore():")
    print(f"Before apply_shadow: {model.fc.weight[0, 0].item():.6f}")

    ema.apply_shadow()
    print(f"After apply_shadow: {model.fc.weight[0, 0].item():.6f}")

    ema.restore()
    print(f"After restore: {model.fc.weight[0, 0].item():.6f}")

    # Test state dict
    state = ema.state_dict()
    print(f"\nEMA state dict keys: {state.keys()}")
    print(f"Number of updates: {state['num_updates']}")

    print("\n✓ EMA test passed!")
