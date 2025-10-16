"""
PVA-ReAct Model Architectures
=============================

This package contains the neural network architectures for the PVA-ReAct framework.

Available Models:
- EnhancedReactionModel: Advanced model with multi-head attention and DFT feature processing
- MultiTaskReactionModel: Basic multi-task model for yield and condition prediction
"""

from .model_architectures import EnhancedReactionModel, MultiTaskReactionModel

__all__ = [
    'EnhancedReactionModel',
    'MultiTaskReactionModel'
]

__version__ = '1.0.0'
