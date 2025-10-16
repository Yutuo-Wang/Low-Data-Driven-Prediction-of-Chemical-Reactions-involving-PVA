import yaml
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
import os
from copy import deepcopy

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    # Model type
    model_type: str = 'enhanced'  # 'enhanced' or 'multitask'
    
    # Architecture parameters
    input_dim: int = 128
    hidden_dim: int = 256
    lstm_layers: int = 2
    transformer_layers: int = 4
    condition_dim: int = 2
    dft_feature_dim: int = 16
    
    # Regularization
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5
    
    # Attention
    num_heads: int = 8
    dim_feedforward: int = 512
    
    # Initialization
    initialization: str = 'xavier'  # 'xavier', 'kaiming', 'orthogonal'
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.model_type not in ['enhanced', 'multitask']:
            errors.append(f"Invalid model_type: {self.model_type}")
        
        if self.input_dim <= 0:
            errors.append("input_dim must be positive")
        
        if self.hidden_dim <= 0:
            errors.append("hidden_dim must be positive")
        
        if self.lstm_layers < 1:
            errors.append("lstm_layers must be at least 1")
        
        if self.transformer_layers < 1:
            errors.append("transformer_layers must be at least 1")
        
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            errors.append("dropout_rate must be between 0 and 1")
        
        if self.num_heads < 1:
            errors.append("num_heads must be at least 1")
        
        return errors

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Training parameters
    epochs: int = 2000
    batch_size: int = 16
    learning_rate: float = 1e-4
    warmup_epochs: int = 10
    
    # Optimization
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'plateau'  # 'plateau', 'cosine', 'step', 'cyclic'
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Gradient handling
    gradient_clip: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Checkpointing
    checkpoint_freq: int = 10
    save_best_only: bool = True
    max_checkpoints: int = 5
    
    # Validation
    val_freq: int = 1
    val_batch_size: int = 32
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.epochs < 1:
            errors.append("epochs must be at least 1")
        
        if self.batch_size < 1:
            errors.append("batch_size must be at least 1")
        
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.optimizer not in ['adam', 'adamw', 'sgd']:
            errors.append(f"Invalid optimizer: {self.optimizer}")
        
        if self.scheduler not in ['plateau', 'cosine', 'step', 'cyclic']:
            errors.append(f"Invalid scheduler: {self.scheduler}")
        
        if self.early_stopping_patience < 1:
            errors.append("early_stopping_patience must be at least 1")
        
        if self.gradient_clip < 0:
            errors.append("gradient_clip must be non-negative")
        
        return errors

@dataclass
class DFTConfig:
    """Configuration for DFT calculations."""
    # Calculation parameters
    basis_set: str = 'def2svp'
    functional: str = 'B3LYP'
    convergence_tolerance: float = 1e-6
    max_iterations: int = 100
    
    # Resource management
    memory_limit: int = 4000  # MB
    num_threads: int = 4
    
    # Output control
    save_orbitals: bool = False
    orbital_dir: str = 'orbitals'
    
    # Feature selection
    calculate_fukui: bool = True
    calculate_reactivity: bool = True
    
    # Caching
    use_cache: bool = True
    cache_dir: str = 'dft_cache'
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        common_basis_sets = ['sto-3g', '6-31g', 'def2svp', 'def2tzvp', 'cc-pvdz']
        if self.basis_set not in common_basis_sets:
            errors.append(f"Uncommon basis set: {self.basis_set}")
        
        common_functionals = ['B3LYP', 'PBE', 'PBE0', 'M06-2X', 'ωB97X-D']
        if self.functional not in common_functionals:
            errors.append(f"Uncommon functional: {self.functional}")
        
        if self.convergence_tolerance <= 0:
            errors.append("convergence_tolerance must be positive")
        
        if self.max_iterations < 1:
            errors.append("max_iterations must be at least 1")
        
        if self.memory_limit < 100:
            errors.append("memory_limit must be at least 100 MB")
        
        if self.num_threads < 1:
            errors.append("num_threads must be at least 1")
        
        return errors

@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    # Experiment identification
    experiment_name: str = 'pva_react_experiment'
    experiment_id: str = ''
    description: str = 'PVA-ReAct chemical reaction prediction'
    
    # Directory structure
    base_dir: str = 'experiments'
    data_dir: str = 'data'
    model_dir: str = 'models'
    results_dir: str = 'results'
    logs_dir: str = 'logs'
    
    # Data configuration
    data_file: str = 'reaction_data.csv'
    test_size: float = 0.1
    val_size: float = 0.125
    random_seed: int = 42
    
    # SMILES processing
    tokenizer_name: str = "seyonec/PubChem10M_SMILES_BPE_396_250"
    max_length: int = 200
    canonicalize_smiles: bool = True
    
    # Feature scaling
    normalize_features: bool = True
    normalize_target: bool = False
    feature_scaler: str = 'standard'
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'training.log'
    wandb_project: str = ''  # Weights & Biases project name
    wandb_entity: str = ''   # Weights & Biases entity
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if not self.experiment_name:
            errors.append("experiment_name cannot be empty")
        
        if self.test_size <= 0 or self.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        
        if self.val_size <= 0 or self.val_size >= 1:
            errors.append("val_size must be between 0 and 1")
        
        if self.max_length < 1:
            errors.append("max_length must be at least 1")
        
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_log_levels:
            errors.append(f"Invalid log_level: {self.log_level}")
        
        if self.feature_scaler not in ['standard', 'minmax', 'none']:
            errors.append(f"Invalid feature_scaler: {self.feature_scaler}")
        
        return errors
    
    def get_experiment_dir(self) -> str:
        """Get experiment directory path."""
        if self.experiment_id:
            return os.path.join(self.base_dir, self.experiment_id)
        else:
            return os.path.join(self.base_dir, self.experiment_name)
    
    def setup_directories(self):
        """Create experiment directory structure."""
        exp_dir = self.get_experiment_dir()
        directories = [
            exp_dir,
            os.path.join(exp_dir, self.data_dir),
            os.path.join(exp_dir, self.model_dir),
            os.path.join(exp_dir, self.results_dir),
            os.path.join(exp_dir, self.logs_dir)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"Experiment directories created in {exp_dir}")

@dataclass
class FullConfig:
    """Complete configuration for PVA-ReAct framework."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dft: DFTConfig = field(default_factory=DFTConfig)
    
    # Additional metadata
    version: str = '1.0.0'
    timestamp: str = ''
    git_hash: str = ''
    
    def validate(self) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        errors.extend(self.experiment.validate())
        errors.extend(self.model.validate())
        errors.extend(self.training.validate())
        errors.extend(self.dft.validate())
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'experiment': asdict(self.experiment),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'dft': asdict(self.dft),
            'version': self.version,
            'timestamp': self.timestamp,
            'git_hash': self.git_hash
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FullConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'experiment' in config_dict:
            for key, value in config_dict['experiment'].items():
                if hasattr(config.experiment, key):
                    setattr(config.experiment, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if 'dft' in config_dict:
            for key, value in config_dict['dft'].items():
                if hasattr(config.dft, key):
                    setattr(config.dft, key, value)
        
        # Metadata
        if 'version' in config_dict:
            config.version = config_dict['version']
        if 'timestamp' in config_dict:
            config.timestamp = config_dict['timestamp']
        if 'git_hash' in config_dict:
            config.git_hash = config_dict['git_hash']
        
        return config


def load_config(config_path: str) -> FullConfig:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Loaded configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        config = FullConfig.from_dict(config_dict)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.warning(f"Configuration validation warnings: {errors}")
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config(config: FullConfig, config_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration object to save
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    
    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                # Default to YAML
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


def validate_config(config: Union[FullConfig, Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validate configuration and return results.
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        is_valid: Whether configuration is valid
        errors: List of validation errors
    """
    if isinstance(config, dict):
        config_obj = FullConfig.from_dict(config)
    else:
        config_obj = config
    
    errors = config_obj.validate()
    is_valid = len(errors) == 0
    
    return is_valid, errors


def update_config(config: FullConfig, updates: Dict[str, Any]) -> FullConfig:
    """
    Update configuration with new values.
    
    Args:
        config: Original configuration
        updates: Dictionary of updates
        
    Returns:
        updated_config: Updated configuration
    """
    updated_config = deepcopy(config)
    
    for section, values in updates.items():
        if hasattr(updated_config, section):
            section_obj = getattr(updated_config, section)
            for key, value in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {section}.{key}")
        else:
            logger.warning(f"Unknown configuration section: {section}")
    
    return updated_config


def create_default_config() -> FullConfig:
    """Create a default configuration."""
    return FullConfig()


def generate_config_template(output_path: str = None) -> str:
    """
    Generate a configuration template.
    
    Args:
        output_path: Path to save template (optional)
        
    Returns:
        template: Configuration template as string
    """
    default_config = create_default_config()
    config_dict = default_config.to_dict()
    
    if output_path:
        save_config(default_config, output_path)
        return output_path
    else:
        return yaml.dump(config_dict, default_flow_style=False, indent=2)


# Example usage and testing
if __name__ == "__main__":
    print("Testing configuration utilities...")
    
    try:
        # Create default configuration
        config = create_default_config()
        print("Default configuration created")
        
        # Validate configuration
        is_valid, errors = validate_config(config)
        print(f"Configuration valid: {is_valid}")
        if errors:
            print(f"Validation errors: {errors}")
        
        # Save configuration
        save_config(config, 'test_config.yaml')
        print("Configuration saved to test_config.yaml")
        
        # Load configuration
        loaded_config = load_config('test_config.yaml')
        print("Configuration loaded successfully")
        
        # Update configuration
        updates = {
            'experiment': {'experiment_name': 'test_experiment'},
            'training': {'learning_rate': 1e-3}
        }
        updated_config = update_config(loaded_config, updates)
        print("Configuration updated successfully")
        
        # Generate template
        template = generate_config_template()
        print("Configuration template generated")
        
        # Clean up
        if os.path.exists('test_config.yaml'):
            os.remove('test_config.yaml')
        
        print("All configuration tests passed!")
        
    except Exception as e:
        print(f"Configuration test failed: {e}")
