"""
Pre-trained Model Weights
========================

This directory contains pre-trained model weights for the PVA-ReAct framework.

File Structure:
- enhanced_model_weights.pth: Weights for EnhancedReactionModel
- multitask_model_weights.pth: Weights for MultiTaskReactionModel
- config.json: Model configuration files

Usage:
    from models.model_architectures import EnhancedReactionModel
    import torch
    
    model = EnhancedReactionModel(vocab_size=50265)
    model.load_state_dict(torch.load('models/pretrained_models/enhanced_model_weights.pth'))
"""

import os
from typing import Dict, Any
import json

def get_pretrained_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for pre-trained models.
    
    Args:
        model_name: Name of pre-trained model ('enhanced' or 'multitask')
        
    Returns:
        config: Model configuration dictionary
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        all_configs = json.load(f)
    
    if model_name not in all_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(all_configs.keys())}")
    
    return all_configs[model_name]

def list_pretrained_models() -> list:
    """List available pre-trained models."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if not os.path.exists(config_path):
        return []
    
    with open(config_path, 'r') as f:
        all_configs = json.load(f)
    
    return list(all_configs.keys())

__all__ = ['get_pretrained_config', 'list_pretrained_models']
