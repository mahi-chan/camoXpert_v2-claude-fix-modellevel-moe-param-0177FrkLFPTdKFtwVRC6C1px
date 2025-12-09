import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str = 'checkpoint.pth'):
    """
    Save model checkpoint.

    Args:
        state (dict): State dictionary containing model and optimizer states.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into.

    Returns:
        int: Epoch number from the checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f" Checkpoint loaded from {checkpoint_path}")
    return checkpoint.get('epoch', 0)


def count_parameters(model: torch.nn.Module):
    """
    Count the total and trainable parameters in a model.

    Args:
        model (torch.nn.Module): Model to count parameters for.

    Returns:
        tuple: Total parameters, Trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def create_directory(path: str):
    """
    Create a directory if it does not exist.

    Args:
        path (str): Path to the directory.
    """
    Path(path).mkdir(parents=True, exist_ok=True)