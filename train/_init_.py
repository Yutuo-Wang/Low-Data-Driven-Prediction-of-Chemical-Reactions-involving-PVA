"""
PVA-ReAct Training Framework
============================

This package contains training utilities and modules for the PVA-ReAct framework.

Available Modules:
- train_main: Main training script and training loop
- training_utils: Training utilities, evaluation functions, and metrics
- early_stopping: Early stopping implementation and training monitoring
"""

from .early_stopping import EarlyStopping, TrainingMonitor, LearningRateScheduler
from .training_utils import (
    TrainingHistory, 
    ModelCheckpoint, 
    evaluate_model, 
    save_model, 
    load_model,
    train_multitask,
    calculate_metrics
)

__all__ = [
    'EarlyStopping',
    'TrainingMonitor', 
    'LearningRateScheduler',
    'TrainingHistory',
    'ModelCheckpoint',
    'evaluate_model',
    'save_model',
    'load_model',
    'train_multitask',
    'calculate_metrics'
]

__version__ = '1.0.0'
