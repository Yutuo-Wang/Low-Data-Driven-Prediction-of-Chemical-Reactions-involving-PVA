import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import yaml
import json
from datetime import datetime
import logging

# Import from our modules
from models.model_architectures import EnhancedReactionModel, create_model
from features.dft_extractor import EnhancedDFTFeatureExtractor, DFTConfig
from features.smiles_processor import SMILESProcessor, SMILESProcessorConfig
from utils.data_loader import preprocess_data, split_dataset, ReactionDataset
from training.training_utils import (
    train_multitask, 
    evaluate_model, 
    save_model, 
    TrainingHistory,
    calculate_metrics
)
from training.early_stopping import EarlyStopping, TrainingMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PVATrainer:
    """
    Main trainer class for PVA-ReAct model training.
    
    This class handles the complete training pipeline including:
    - Data loading and preprocessing
    - Model initialization and configuration
    - Training loop execution
    - Model evaluation and saving
    - Experiment tracking
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the PVA trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_id = self._generate_experiment_id()
        
        # Initialize components
        self.tokenizer = None
        self.dft_extractor = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = TrainingHistory()
        
        logger.info(f"PVA Trainer initialized with device: {self.device}")
        logger.info(f"Experiment ID: {self.experiment_id}")
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            config: Configuration dictionary
        """
        default_config = {
            'data': {
                'data_path': 'reaction_data.csv',
                'test_size': 0.1,
                'val_size': 0.125,
                'batch_size': 16,
                'max_length': 200
            },
            'model': {
                'type': 'enhanced',
                'input_dim': 128,
                'hidden_dim': 256,
                'lstm_layers': 2,
                'transformer_layers': 4,
                'condition_dim': 2,
                'dft_feature_dim': 16,
                'dropout_rate': 0.1
            },
            'training': {
                'epochs': 2000,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'early_stopping_patience': 10,
                'gradient_clip': 1.0,
                'warmup_epochs': 10
            },
            'dft': {
                'basis_set': 'def2svp',
                'functional': 'B3LYP',
                'save_orbitals': False
            },
            'smiles': {
                'tokenizer_name': "seyonec/PubChem10M_SMILES_BPE_396_250",
                'max_length': 200,
                'canonicalize': True
            },
            'logging': {
                'save_dir': 'experiments',
                'checkpoint_freq': 10,
                'log_metrics': True,
                'save_best_only': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge configurations
            def deep_merge(default, user):
                for key, value in user.items():
                    if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                        deep_merge(default[key], value)
                    else:
                        default[key] = value
                return default
            
            config = deep_merge(default_config.copy(), user_config)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            config = default_config
            if config_path:
                logger.warning(f"Config file {config_path} not found, using defaults")
            else:
                logger.info("Using default configuration")
        
        return config
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"pva_react_{timestamp}"
    
    def setup_experiment(self):
        """Setup experiment directory and logging."""
        save_dir = os.path.join(self.config['logging']['save_dir'], self.experiment_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Update save directory in config
        self.config['logging']['save_dir'] = save_dir
        
        logger.info(f"Experiment setup complete. Files will be saved to: {save_dir}")
    
    def initialize_components(self):
        """Initialize all training components."""
        logger.info("Initializing training components...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['smiles']['tokenizer_name']
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize DFT extractor
        dft_config = DFTConfig(
            basis_set=self.config['dft']['basis_set'],
            functional=self.config['dft']['functional'],
            save_orbitals=self.config['dft']['save_orbitals']
        )
        self.dft_extractor = EnhancedDFTFeatureExtractor(dft_config)
        
        # Initialize model
        model_config = self.config['model']
        self.model = create_model(
            model_type=model_config['type'],
            vocab_size=self.tokenizer.vocab_size,
            **{k: v for k, v in model_config.items() if k != 'type'}
        ).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info("All components initialized successfully")
    
    def load_data(self):
        """Load and preprocess training data."""
        logger.info("Loading and preprocessing data...")
        
        # Load data
        data_path = self.config['data']['data_path']
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(data)} samples")
        
        # Preprocess data
        data, scaler = preprocess_data(data)
        
        # Split dataset
        train_data, val_data, test_data = split_dataset(
            data, 
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size']
        )
        
        logger.info(f"Dataset split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create datasets
        train_dataset = ReactionDataset(
            train_data, self.tokenizer, self.dft_extractor,
            max_length=self.config['data']['max_length']
        )
        val_dataset = ReactionDataset(
            val_data, self.tokenizer, self.dft_extractor,
            max_length=self.config['data']['max_length']
        )
        test_dataset = ReactionDataset(
            test_data, self.tokenizer, self.dft_extractor,
            max_length=self.config['data']['max_length']
        )
        
        # Create data loaders
        batch_size = self.config['data']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Save scaler for inference
        scaler_path = os.path.join(self.config['logging']['save_dir'], 'scaler.pkl')
        torch.save(scaler, scaler_path)
        
        return train_loader, val_loader, test_loader, scaler
    
    def train(self):
        """Execute the complete training pipeline."""
        logger.info("Starting PVA-ReAct training pipeline...")
        
        # Setup experiment
        self.setup_experiment()
        
        # Initialize components
        self.initialize_components()
        
        # Load data
        train_loader, val_loader, test_loader, scaler = self.load_data()
        
        # Initialize training monitor
        training_monitor = TrainingMonitor(
            patience=self.config['training']['early_stopping_patience'],
            min_delta=0.001
        )
        
        # Training configuration
        training_config = {
            'epochs': self.config['training']['epochs'],
            'early_stopping_patience': self.config['training']['early_stopping_patience'],
            'gradient_clip': self.config['training']['gradient_clip'],
            'warmup_epochs': self.config['training']['warmup_epochs'],
            'checkpoint_freq': self.config['logging']['checkpoint_freq']
        }
        
        # Execute training
        logger.info("Starting model training...")
        trained_model = train_multitask(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            training_monitor=training_monitor,
            experiment_dir=self.config['logging']['save_dir'],
            **training_config
        )
        
        # Save final model
        model_path = os.path.join(self.config['logging']['save_dir'], 'final_model.pth')
        save_model(trained_model, model_path)
        logger.info(f"Final model saved to {model_path}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(trained_model, test_loader, self.device)
        
        # Save test results
        test_results_path = os.path.join(self.config['logging']['save_dir'], 'test_results.json')
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Test Results - R²: {test_metrics['r2']:.4f}, "
                   f"Accuracy: {test_metrics['accuracy']:.2%}")
        
        return trained_model, test_metrics
    
    def resume_training(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to training checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize components first
        self.initialize_components()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_metrics = checkpoint['best_metrics']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
        
        # Continue training
        return self.train()


def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description='PVA-ReAct Model Training')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--data_path', type=str, default='reaction_data.csv',
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = PVATrainer(args.config)
        
        # Override config with command line arguments if provided
        if args.data_path:
            trainer.config['data']['data_path'] = args.data_path
        if args.epochs:
            trainer.config['training']['epochs'] = args.epochs
        if args.batch_size:
            trainer.config['data']['batch_size'] = args.batch_size
        if args.learning_rate:
            trainer.config['training']['learning_rate'] = args.learning_rate
        
        # Start training
        if args.resume:
            trained_model, test_metrics = trainer.resume_training(args.resume)
        else:
            trained_model, test_metrics = trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
