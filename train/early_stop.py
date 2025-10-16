import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any
import time
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Enhanced early stopping implementation with multiple criteria.
    
    This class monitors training progress and stops training when
    the model stops improving based on specified criteria.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0, 
                 mode: str = 'max', restore_best_weights: bool = True,
                 verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/R²
            restore_best_weights: Whether to restore best weights on stop
            verbose: Whether to print progress messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = 0
        
        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")
    
    def __call__(self, current_score: float, model: torch.nn.Module = None, epoch: int = 0):
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation score
            model: Model to save weights from (optional)
            epoch: Current epoch number
            
        Returns:
            should_stop: Whether training should stop
        """
        if self.mode == 'min':
            score = -current_score
        else:
            score = current_score
        
        if self.best_score is None:
            self.best_score = score
            self._save_best_weights(model, epoch, current_score)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping: {self.counter}/{self.patience} - '
                           f'Best: {self.best_score:.6f}, Current: {score:.6f}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and model is not None:
                    self._restore_best_weights(model)
                    if self.verbose:
                        logger.info('EarlyStopping: Restored best weights')
        else:
            self.best_score = score
            self.counter = 0
            self._save_best_weights(model, epoch, current_score)
            if self.verbose:
                logger.info(f'EarlyStopping: New best score: {self.best_score:.6f}')
        
        return self.early_stop
    
    def _save_best_weights(self, model: torch.nn.Module, epoch: int, score: float):
        """Save best model weights."""
        if model is not None and self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.best_score_actual = score
    
    def _restore_best_weights(self, model: torch.nn.Module):
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = 0
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about the best model."""
        return {
            'best_score': self.best_score_actual if hasattr(self, 'best_score_actual') else self.best_score,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'early_stop': self.early_stop
        }


class TrainingMonitor:
    """
    Comprehensive training monitor with multiple stopping criteria.
    
    This class monitors various aspects of training and can trigger
    stopping based on multiple criteria including:
    - Validation performance plateau
    - Training divergence
    - Overfitting detection
    - Resource limits
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 max_epochs: int = 2000, max_time_hours: float = 24.0,
                 overfitting_threshold: float = 0.1, divergence_threshold: float = 10.0):
        """
        Initialize training monitor.
        
        Args:
            patience: Patience for early stopping
            min_delta: Minimum improvement threshold
            max_epochs: Maximum number of epochs
            max_time_hours: Maximum training time in hours
            overfitting_threshold: Train-val loss ratio threshold
            divergence_threshold: Loss value threshold for divergence
        """
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.max_time_hours = max_time_hours
        self.overfitting_threshold = overfitting_threshold
        self.divergence_threshold = divergence_threshold
        
        self.start_time = time.time()
        self.epoch_times = []
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.learning_rates = []
        
        self.best_epoch = 0
        self.best_val_score = -float('inf')
        self.best_val_loss = float('inf')
        
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        self.stop_reasons = {
            'early_stopping': False,
            'max_epochs': False,
            'time_limit': False,
            'overfitting': False,
            'divergence': False
        }
    
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               val_score: float, learning_rate: float, epoch_time: float):
        """
        Update monitor with new epoch results.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            val_score: Validation score (accuracy/R²)
            learning_rate: Current learning rate
            epoch_time: Time taken for this epoch
        """
        self.epoch_times.append(epoch_time)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_scores.append(val_score)
        self.learning_rates.append(learning_rate)
        
        # Update best scores
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.best_val_loss = val_loss
            self.best_epoch = epoch
    
    def should_stop(self, epoch: int, model: torch.nn.Module = None) -> tuple:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            model: Current model (for weight saving)
            
        Returns:
            should_stop: Whether training should stop
            reason: Reason for stopping
        """
        current_val_score = self.val_scores[-1] if self.val_scores else -float('inf')
        current_train_loss = self.train_losses[-1] if self.train_losses else float('inf')
        current_val_loss = self.val_losses[-1] if self.val_losses else float('inf')
        
        # Check early stopping
        if self.early_stopping(current_val_score, model, epoch):
            self.stop_reasons['early_stopping'] = True
            return True, "early_stopping"
        
        # Check max epochs
        if epoch >= self.max_epochs:
            self.stop_reasons['max_epochs'] = True
            return True, "max_epochs"
        
        # Check time limit
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours >= self.max_time_hours:
            self.stop_reasons['time_limit'] = True
            return True, "time_limit"
        
        # Check for overfitting (train loss much lower than val loss)
        if len(self.train_losses) > 10 and len(self.val_losses) > 10:
            recent_train_loss = np.mean(self.train_losses[-10:])
            recent_val_loss = np.mean(self.val_losses[-10:])
            
            if recent_val_loss > recent_train_loss * (1 + self.overfitting_threshold):
                self.stop_reasons['overfitting'] = True
                return True, "overfitting"
        
        # Check for divergence (loss becomes too large)
        if current_train_loss > self.divergence_threshold or current_val_loss > self.divergence_threshold:
            self.stop_reasons['divergence'] = True
            return True, "divergence"
        
        return False, None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
        
        return {
            'total_epochs': len(self.train_losses),
            'total_time_hours': total_time / 3600,
            'average_epoch_time_seconds': avg_epoch_time,
            'best_epoch': self.best_epoch,
            'best_val_score': self.best_val_score,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_val_score': self.val_scores[-1] if self.val_scores else None,
            'stop_reasons': self.stop_reasons,
            'early_stopping_info': self.early_stopping.get_best_info()
        }
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress metrics."""
        if not self.train_losses:
            logger.warning("No training data to plot")
            return
        
        import matplotlib.pyplot as plt
        
        epochs = list(range(1, len(self.train_losses) + 1))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', alpha=0.7)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', alpha=0.7)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot validation scores
        ax2.plot(epochs, self.val_scores, 'g-', label='Val Score')
        ax2.axhline(y=self.best_val_score, color='r', linestyle='--', 
                   label=f'Best: {self.best_val_score:.4f}')
        ax2.set_title('Validation Score (R²/Accuracy)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot learning rate
        ax3.plot(epochs, self.learning_rates, 'orange')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot epoch times
        ax4.plot(epochs, self.epoch_times, 'purple')
        ax4.set_title('Epoch Time')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training progress plot saved to {save_path}")
        
        plt.close()
    
    def save_monitor_state(self, filepath: str):
        """Save monitor state to file."""
        state = {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'max_epochs': self.max_epochs,
            'max_time_hours': self.max_time_hours,
            'start_time': self.start_time,
            'epoch_times': self.epoch_times,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_scores': self.val_scores,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_score': self.best_val_score,
            'best_val_loss': self.best_val_loss,
            'stop_reasons': self.stop_reasons,
            'early_stopping_state': self.early_stopping.get_best_info(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Training monitor state saved to {filepath}")
    
    def load_monitor_state(self, filepath: str):
        """Load monitor state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.patience = state['patience']
        self.min_delta = state['min_delta']
        self.max_epochs = state['max_epochs']
        self.max_time_hours = state['max_time_hours']
        self.start_time = state['start_time']
        self.epoch_times = state['epoch_times']
        self.train_losses = state['train_losses']
        self.val_losses = state['val_losses']
        self.val_scores = state['val_scores']
        self.learning_rates = state['learning_rates']
        self.best_epoch = state['best_epoch']
        self.best_val_score = state['best_val_score']
        self.best_val_loss = state['best_val_loss']
        self.stop_reasons = state['stop_reasons']
        
        logger.info(f"Training monitor state loaded from {filepath}")


class LearningRateScheduler:
    """
    Enhanced learning rate scheduler with multiple strategies.
    
    This class implements various learning rate scheduling strategies
    including cosine annealing, warm restarts, and custom schedules.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, strategy: str = 'plateau',
                 lr: float = 1e-4, warmup_epochs: int = 10, 
                 decay_factor: float = 0.5, patience: int = 5,
                 min_lr: float = 1e-7, T_0: int = 50, T_mult: int = 2):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            strategy: Scheduling strategy ('plateau', 'cosine', 'step', 'cyclic')
            lr: Initial learning rate
            warmup_epochs: Number of warmup epochs
            decay_factor: Factor for reducing learning rate
            patience: Patience for plateau scheduler
            min_lr: Minimum learning rate
            T_0: Initial period for cosine annealing
            T_mult: Multiplicative factor for cosine annealing
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.initial_lr = lr
        self.warmup_epochs = warmup_epochs
        self.decay_factor = decay_factor
        self.patience = patience
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        
        self.epoch = 0
        self.best_score = -float('inf')
        
        # Initialize appropriate scheduler
        if strategy == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=decay_factor, patience=patience,
                min_lr=min_lr, verbose=True
            )
        elif strategy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr, last_epoch=-1
            )
        elif strategy == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=patience, gamma=decay_factor
            )
        elif strategy == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=min_lr, max_lr=lr,
                step_size_up=2000, mode='triangular2'
            )
        else:
            raise ValueError(f"Unknown scheduler strategy: {strategy}")
    
    def step(self, score: float = None):
        """
        Update learning rate.
        
        Args:
            score: Current validation score (for plateau scheduler)
        """
        self.epoch += 1
        
        # Warmup phase
        if self.epoch <= self.warmup_epochs:
            warmup_factor = self.epoch / self.warmup_epochs
            lr = self.initial_lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return
        
        # Use appropriate scheduler step
        if self.strategy == 'plateau' and score is not None:
            self.scheduler.step(score)
            self.best_score = max(self.best_score, score)
        else:
            self.scheduler.step()
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr_history(self) -> List[float]:
        """Get learning rate history (if available)."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            return [self.get_current_lr()]
    
    def state_dict(self):
        """Get scheduler state dictionary."""
        return {
            'scheduler_state': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_score': self.best_score,
            'strategy': self.strategy
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state dictionary."""
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.epoch = state_dict.get('epoch', 0)
        self.best_score = state_dict.get('best_score', -float('inf'))
        self.strategy = state_dict.get('strategy', 'plateau')


# Example usage and testing
if __name__ == "__main__":
    # Test early stopping
    print("Testing EarlyStopping...")
    early_stop = EarlyStopping(patience=3, min_delta=0.01)
    
    scores = [0.5, 0.6, 0.65, 0.66, 0.67, 0.675, 0.676, 0.677]
    
    for i, score in enumerate(scores):
        should_stop = early_stop(score)
        print(f"Epoch {i+1}: Score={score:.3f}, ShouldStop={should_stop}")
        if should_stop:
            break
    
    # Test training monitor
    print("\nTesting TrainingMonitor...")
    monitor = TrainingMonitor(patience=3, max_epochs=5)
    
    for epoch in range(1, 6):
        monitor.update(
            epoch=epoch,
            train_loss=1.0 / epoch,
            val_loss=1.2 / epoch,
            val_score=0.5 + 0.1 * epoch,
            learning_rate=1e-4 / epoch,
            epoch_time=10.0
        )
        
        should_stop, reason = monitor.should_stop(epoch)
        print(f"Epoch {epoch}: ShouldStop={should_stop}, Reason={reason}")
        
        if should_stop:
            break
    
    summary = monitor.get_training_summary()
    print(f"Training summary: {summary}")
    
    print("All early stopping tests completed successfully!")
