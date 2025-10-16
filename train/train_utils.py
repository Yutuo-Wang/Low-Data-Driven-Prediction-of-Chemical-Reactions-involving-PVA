import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)

class TrainingHistory:
    """
    Comprehensive training history tracker.
    
    This class tracks and manages training metrics, losses, and learning rates
    throughout the training process.
    """
    
    def __init__(self):
        """Initialize training history."""
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_yield_loss': [],
            'train_condition_loss': [],
            'val_loss': [],
            'val_r2': [],
            'val_accuracy': [],
            'val_mae': [],
            'val_rmse': [],
            'learning_rates': [],
            'grad_norms': [],
            'time_per_epoch': []
        }
        
        self.best_metrics = {
            'best_r2': -float('inf'),
            'best_accuracy': 0.0,
            'best_epoch': 0,
            'best_loss': float('inf')
        }
    
    def update_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                    learning_rate: float, grad_norm: float = None, epoch_time: float = None):
        """
        Update history with new epoch results.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            learning_rate: Current learning rate
            grad_norm: Gradient norm (optional)
            epoch_time: Time taken for epoch (optional)
        """
        self.history['epoch'].append(epoch)
        
        # Training metrics
        self.history['train_loss'].append(train_metrics.get('total_loss', 0))
        self.history['train_yield_loss'].append(train_metrics.get('yield_loss', 0))
        self.history['train_condition_loss'].append(train_metrics.get('condition_loss', 0))
        
        # Validation metrics
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['val_r2'].append(val_metrics.get('r2', 0))
        self.history['val_accuracy'].append(val_metrics.get('accuracy', 0))
        self.history['val_mae'].append(val_metrics.get('mae', 0))
        self.history['val_rmse'].append(val_metrics.get('rmse', 0))
        
        # Training state
        self.history['learning_rates'].append(learning_rate)
        if grad_norm is not None:
            self.history['grad_norms'].append(grad_norm)
        if epoch_time is not None:
            self.history['time_per_epoch'].append(epoch_time)
        
        # Update best metrics
        current_r2 = val_metrics.get('r2', -float('inf'))
        current_accuracy = val_metrics.get('accuracy', 0)
        current_loss = val_metrics.get('loss', float('inf'))
        
        if current_r2 > self.best_metrics['best_r2']:
            self.best_metrics['best_r2'] = current_r2
            self.best_metrics['best_epoch'] = epoch
        
        if current_accuracy > self.best_metrics['best_accuracy']:
            self.best_metrics['best_accuracy'] = current_accuracy
        
        if current_loss < self.best_metrics['best_loss']:
            self.best_metrics['best_loss'] = current_loss
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get metrics from the latest epoch."""
        if not self.history['epoch']:
            return {}
        
        latest = {}
        for key in self.history:
            if self.history[key]:
                latest[key] = self.history[key][-1]
        
        return latest
    
    def plot_training_history(self, save_path: str = None, show: bool = False):
        """
        Plot comprehensive training history.
        
        Args:
            save_path: Path to save the plot (optional)
            show: Whether to display the plot
        """
        if not self.history['epoch']:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = self.history['epoch']
        
        # Plot 1: Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: R² score
        axes[0, 1].plot(epochs, self.history['val_r2'], 'g-', label='Validation R²')
        axes[0, 1].axhline(y=self.best_metrics['best_r2'], color='r', linestyle='--', 
                          label=f'Best R²: {self.best_metrics["best_r2"]:.4f}')
        axes[0, 1].set_title('Validation R² Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Accuracy
        axes[0, 2].plot(epochs, self.history['val_accuracy'], 'purple', label='Validation Accuracy')
        axes[0, 2].axhline(y=self.best_metrics['best_accuracy'], color='r', linestyle='--',
                          label=f'Best Accuracy: {self.best_metrics["best_accuracy"]:.2%}')
        axes[0, 2].set_title('Validation Accuracy (±5%)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Learning rate
        if self.history['learning_rates']:
            axes[1, 0].plot(epochs, self.history['learning_rates'], 'orange')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Gradient norms
        if self.history['grad_norms']:
            axes[1, 1].plot(epochs, self.history['grad_norms'], 'brown')
            axes[1, 1].set_title('Gradient Norms')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Component losses
        axes[1, 2].plot(epochs, self.history['train_yield_loss'], 'blue', label='Yield Loss', alpha=0.7)
        axes[1, 2].plot(epochs, self.history['train_condition_loss'], 'green', label='Condition Loss', alpha=0.7)
        axes[1, 2].set_title('Component Training Losses')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_history(self, filepath: str):
        """Save training history to JSON file."""
        history_data = {
            'history': self.history,
            'best_metrics': self.best_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Training history saved to {filepath}")
    
    def load_history(self, filepath: str):
        """Load training history from JSON file."""
        with open(filepath, 'r') as f:
            history_data = json.load(f)
        
        self.history = history_data['history']
        self.best_metrics = history_data['best_metrics']
        
        logger.info(f"Training history loaded from {filepath}")


class ModelCheckpoint:
    """
    Model checkpoint manager for saving and loading model states.
    """
    
    def __init__(self, save_dir: str, save_best_only: bool = True, 
                 checkpoint_freq: int = 10, max_checkpoints: int = 5):
        """
        Initialize model checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            save_best_only: Whether to only save best model
            checkpoint_freq: Frequency of checkpoint saving (epochs)
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_metric = -float('inf')
        self.checkpoint_files = []
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler._LRScheduler,
                       epoch: int, metrics: Dict[str, float], is_best: bool = False,
                       training_history: TrainingHistory = None):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
            training_history: Training history object
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if training_history:
            checkpoint['training_history'] = training_history
        
        # Always save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save periodic checkpoints
        if epoch % self.checkpoint_freq == 0:
            periodic_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch:04d}.pth')
            torch.save(checkpoint, periodic_path)
            self._manage_checkpoints(periodic_path)
        
        # Save best model
        if is_best or not self.save_best_only:
            current_metric = metrics.get('r2', metrics.get('accuracy', 0))
            
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                best_path = os.path.join(self.save_dir, 'model_best.pth')
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved with metric: {current_metric:.4f}")
    
    def _manage_checkpoints(self, new_checkpoint: str):
        """Manage checkpoint files to avoid exceeding maximum count."""
        self.checkpoint_files.append(new_checkpoint)
        
        # Keep only the most recent checkpoints
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoint_files.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: torch.optim.Optimizer = None,
                       scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            
        Returns:
            checkpoint: Loaded checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {checkpoint['epoch']}")
        
        return checkpoint


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 5.0) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Threshold for accuracy calculation
        
    Returns:
        metrics: Dictionary of calculated metrics
    """
    metrics = {}
    
    # Basic regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Percentage-based metrics
    absolute_errors = np.abs(y_true - y_pred)
    percentage_errors = (absolute_errors / (np.abs(y_true) + 1e-8)) * 100
    
    metrics['mape'] = np.mean(percentage_errors)
    metrics['accuracy'] = np.mean(absolute_errors < threshold)
    metrics['accuracy_10'] = np.mean(absolute_errors < 10.0)  # 10% threshold
    
    # Statistical metrics
    metrics['std_error'] = np.std(y_true - y_pred)
    metrics['bias'] = np.mean(y_pred - y_true)
    
    # Correlation metrics
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
    
    return metrics


def evaluate_model(model: nn.Module, loader: torch.utils.data.DataLoader, 
                  device: torch.device, threshold: float = 5.0) -> Dict[str, Any]:
    """
    Comprehensive model evaluation function.
    
    Args:
        model: Model to evaluate
        loader: Data loader for evaluation
        device: Device to run evaluation on
        threshold: Accuracy threshold
        
    Returns:
        metrics: Comprehensive evaluation metrics
    """
    model.eval()
    all_yield_preds = []
    all_yield_true = []
    all_condition_preds = []
    all_condition_true = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            reactant_ids = batch['reactant_ids'].to(device)
            product_ids = batch['product_ids'].to(device)
            features = batch['features'].to(device)
            yield_true = batch['yield_value'].to(device)
            
            yield_pred, condition_pred = model(reactant_ids, product_ids, features)
            
            all_yield_preds.append(yield_pred.cpu().numpy())
            all_yield_true.append(yield_true.cpu().numpy())
            
            # Condition predictions (first two features are temperature and time)
            condition_true = features[:, :2].cpu().numpy()
            all_condition_preds.append(condition_pred.cpu().numpy())
            all_condition_true.append(condition_true)
    
    # Concatenate all predictions and targets
    yield_preds = np.concatenate(all_yield_preds)
    yield_true = np.concatenate(all_yield_true)
    condition_preds = np.concatenate(all_condition_preds)
    condition_true = np.concatenate(all_condition_true)
    
    # Calculate yield metrics
    yield_metrics = calculate_metrics(yield_true, yield_preds, threshold)
    yield_metrics['preds'] = yield_preds
    yield_metrics['labels'] = yield_true
    
    # Calculate condition metrics
    condition_metrics = {}
    for i, name in enumerate(['temperature', 'time']):
        condition_metrics[f'{name}_mse'] = mean_squared_error(condition_true[:, i], condition_preds[:, i])
        condition_metrics[f'{name}_mae'] = mean_absolute_error(condition_true[:, i], condition_preds[:, i])
        condition_metrics[f'{name}_r2'] = r2_score(condition_true[:, i], condition_preds[:, i])
    
    # Combined metrics
    combined_metrics = {
        'yield': yield_metrics,
        'conditions': condition_metrics,
        'loss': yield_metrics['mse'] + 0.3 * np.mean([condition_metrics['temperature_mae'], condition_metrics['time_mae']])
    }
    
    return combined_metrics


def save_model(model: nn.Module, path: str, metadata: Dict = None):
    """
    Save model with metadata.
    
    Args:
        model: Model to save
        path: Path to save model
        metadata: Additional metadata to save
    """
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        save_data.update(metadata)
    
    torch.save(save_data, path)
    logger.info(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: torch.device = None) -> Dict:
    """
    Load model with metadata.
    
    Args:
        model: Model to load state into
        path: Path to saved model
        device: Device to load model on
        
    Returns:
        metadata: Model metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded from {path}")
    
    # Return metadata (excluding model state)
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    return metadata


def train_multitask(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device,
                   training_monitor: Any = None, experiment_dir: str = None,
                   epochs: int = 2000, early_stopping_patience: int = 10,
                   gradient_clip: float = 1.0, warmup_epochs: int = 10,
                   checkpoint_freq: int = 10) -> nn.Module:
    """
    Enhanced multi-task training function with comprehensive monitoring.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Training device
        training_monitor: Training monitor for early stopping
        experiment_dir: Experiment directory for saving
        epochs: Maximum number of epochs
        early_stopping_patience: Early stopping patience
        gradient_clip: Gradient clipping value
        warmup_epochs: Number of warmup epochs
        checkpoint_freq: Checkpoint frequency
        
    Returns:
        model: Trained model
    """
    from training.early_stopping import EarlyStopping
    
    # Initialize components
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001)
    history = TrainingHistory()
    checkpoint_manager = ModelCheckpoint(
        os.path.join(experiment_dir, 'checkpoints') if experiment_dir else 'checkpoints',
        checkpoint_freq=checkpoint_freq
    )
    
    # Loss functions
    yield_criterion = nn.MSELoss()
    condition_criterion = nn.L1Loss()
    
    best_r2 = -float('inf')
    best_model_state = None
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0.0
        yield_losses = 0.0
        condition_losses = 0.0
        grad_norms = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            
            # Move data to device
            reactant_ids = batch['reactant_ids'].to(device)
            product_ids = batch['product_ids'].to(device)
            features = batch['features'].to(device)
            yield_true = batch['yield_value'].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                yield_pred, condition_pred = model(reactant_ids, product_ids, features)
                
                # Calculate losses
                loss_yield = yield_criterion(yield_pred, yield_true)
                loss_condition = condition_criterion(condition_pred, features[:, :2])
                
                # Combined loss with warmup for condition loss
                condition_weight = 0.3 * min(1.0, (epoch + 1) / warmup_epochs)
                loss = loss_yield + condition_weight * loss_condition
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Calculate gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate losses
            total_loss += loss.item()
            yield_losses += loss_yield.item()
            condition_losses += loss_condition.item()
        
        # Validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        train_metrics = {
            'total_loss': total_loss / len(train_loader),
            'yield_loss': yield_losses / len(train_loader),
            'condition_loss': condition_losses / len(train_loader)
        }
        
        history.update_epoch(
            epoch + 1, train_metrics, val_metrics, current_lr, 
            avg_grad_norm, epoch_time
        )
        
        # Print progress
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f} | "
                   f"Yield Loss: {train_metrics['yield_loss']:.4f} | "
                   f"Condition Loss: {train_metrics['condition_loss']:.4f}")
        logger.info(f"Val R²: {val_metrics['yield']['r2']:.4f} | "
                   f"Val Accuracy: {val_metrics['yield']['accuracy']:.2%} | "
                   f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"LR: {current_lr:.2e} | Grad Norm: {avg_grad_norm:.2f} | Time: {epoch_time:.1f}s")
        
        # Check for best model
        current_r2 = val_metrics['yield']['r2']
        is_best = current_r2 > best_r2
        
        if is_best:
            best_r2 = current_r2
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model with R²: {best_r2:.4f}")
        
        # Save checkpoint
        if experiment_dir:
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_metrics,
                is_best=is_best, training_history=history
            )
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_metrics['yield']['r2'])
        
        # Early stopping check
        early_stopping(val_metrics['yield']['r2'])
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with R²: {best_r2:.4f}")
    
    # Save final training history
    if experiment_dir:
        history_path = os.path.join(experiment_dir, 'training_history.json')
        history.save_history(history_path)
        
        plot_path = os.path.join(experiment_dir, 'training_plot.png')
        history.plot_training_history(plot_path)
    
    return model


# Utility function for quick training
def quick_train(model: nn.Module, train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader, device: torch.device,
               epochs: int = 100, learning_rate: float = 1e-4) -> nn.Module:
    """
    Quick training function for rapid prototyping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        epochs: Number of epochs
        learning_rate: Learning rate
        
    Returns:
        model: Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    return train_multitask(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        early_stopping_patience=10
    )


# Test function
if __name__ == "__main__":
    # Test training utilities
    print("Testing training utilities...")
    
    # Test metrics calculation
    y_true = np.random.rand(100) * 100
    y_pred = y_true + np.random.normal(0, 5, 100)
    
    metrics = calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    # Test training history
    history = TrainingHistory()
    print("Training history initialized successfully")
    
    print("All training utilities tested successfully!")
