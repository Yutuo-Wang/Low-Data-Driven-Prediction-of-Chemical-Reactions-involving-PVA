import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
import json
from pathlib import Path
import pickle
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

@dataclass
class DataLoaderConfig:
    """Configuration for data loading and preprocessing."""
    batch_size: int = 16
    test_size: float = 0.1
    val_size: float = 0.125
    random_state: int = 42
    shuffle: bool = True
    max_length: int = 200
    normalize_features: bool = True
    normalize_target: bool = False
    feature_scaler: str = 'standard'  # 'standard', 'minmax', or None
    target_scaler: str = None  # 'standard', 'minmax', or None

class ReactionDataset(Dataset):
    """
    Enhanced dataset class for PVA reaction data.
    
    This class handles loading, preprocessing, and batching of reaction data
    including SMILES strings, reaction conditions, and DFT features.
    """
    
    def __init__(self, data: pd.DataFrame, tokenizer, dft_extractor, 
                 config: DataLoaderConfig = None, is_training: bool = True):
        """
        Initialize reaction dataset.
        
        Args:
            data: DataFrame containing reaction data
            tokenizer: SMILES tokenizer
            dft_extractor: DFT feature extractor
            config: Data loader configuration
            is_training: Whether this is for training (affects preprocessing)
        """
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.dft_extractor = dft_extractor
        self.config = config or DataLoaderConfig()
        self.is_training = is_training
        
        # Feature cache for efficiency
        self.feature_cache = {}
        self.smiles_cache = {}
        
        # Initialize scalers
        self.feature_scaler = None
        self.target_scaler = None
        
        # Preprocess data
        self._preprocess_data()
        
        logger.info(f"ReactionDataset initialized with {len(self.data)} samples")
        logger.info(f"Feature dimension: {self._get_feature_dimension()}")
    
    def _preprocess_data(self):
        """Preprocess the dataset."""
        # Ensure required columns exist
        required_columns = ['reactant_smiles', 'product_smiles', 'yield']
        optional_columns = ['temperature', 'time', 'reaction_type']
        
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Handle missing values
        self.data = self.data.dropna(subset=required_columns)
        
        # Convert yield to float
        self.data['yield'] = pd.to_numeric(self.data['yield'], errors='coerce')
        self.data = self.data.dropna(subset=['yield'])
        
        # Ensure yield is within reasonable bounds
        self.data = self.data[(self.data['yield'] >= 0) & (self.data['yield'] <= 100)]
        
        # Handle temperature and time
        for col in ['temperature', 'time']:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                # Fill missing values with median
                if self.data[col].isna().any():
                    median_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_val)
                    logger.info(f"Filled missing {col} with median: {median_val}")
        
        # Initialize scalers if needed
        if self.config.normalize_features and self.is_training:
            self._initialize_feature_scaler()
        
        if self.config.normalize_target and self.is_training:
            self._initialize_target_scaler()
    
    def _initialize_feature_scaler(self):
        """Initialize feature scaler based on training data."""
        feature_columns = ['temperature', 'time'] if 'temperature' in self.data.columns else []
        
        if feature_columns:
            feature_data = self.data[feature_columns].values
            
            if self.config.feature_scaler == 'standard':
                self.feature_scaler = StandardScaler()
            elif self.config.feature_scaler == 'minmax':
                self.feature_scaler = MinMaxScaler()
            else:
                logger.warning(f"Unknown scaler: {self.config.feature_scaler}")
                return
            
            self.feature_scaler.fit(feature_data)
            logger.info("Feature scaler initialized")
    
    def _initialize_target_scaler(self):
        """Initialize target scaler based on training data."""
        target_data = self.data['yield'].values.reshape(-1, 1)
        
        if self.config.target_scaler == 'standard':
            self.target_scaler = StandardScaler()
        elif self.config.target_scaler == 'minmax':
            self.target_scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown target scaler: {self.config.target_scaler}")
            return
        
        self.target_scaler.fit(target_data)
        logger.info("Target scaler initialized")
    
    def _get_feature_dimension(self) -> int:
        """Calculate total feature dimension."""
        # Conditions (temperature, time)
        condition_dim = 2 if 'temperature' in self.data.columns else 0
        
        # DFT features (reactant + product)
        dft_dim = len(self.dft_extractor.feature_labels) * 2
        
        return condition_dim + dft_dim
    
    def _get_dft_features(self, smiles: str) -> np.ndarray:
        """
        Get DFT features for a SMILES string with caching.
        
        Args:
            smiles: SMILES string
            
        Returns:
            features: DFT features array
        """
        if not isinstance(smiles, str) or len(smiles) == 0:
            return np.zeros(len(self.dft_extractor.feature_labels))
        
        if smiles in self.feature_cache:
            return self.feature_cache[smiles]
        
        features, _ = self.dft_extractor.calculate(smiles)
        self.feature_cache[smiles] = features
        
        return features
    
    def _preprocess_smiles(self, smiles: str) -> str:
        """
        Preprocess SMILES string with caching.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            processed_smiles: Processed SMILES string
        """
        if not isinstance(smiles, str) or len(smiles) == 0:
            return "C"  # Default to methane
        
        if smiles in self.smiles_cache:
            return self.smiles_cache[smiles]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                processed_smiles = Chem.MolToSmiles(mol, canonical=True)
                self.smiles_cache[smiles] = processed_smiles
                return processed_smiles
        except:
            pass
        
        # Return original if processing fails
        self.smiles_cache[smiles] = smiles
        return smiles
    
    def _encode_smiles(self, smiles: str) -> torch.Tensor:
        """
        Encode SMILES string to token IDs.
        
        Args:
            smiles: SMILES string
            
        Returns:
            token_ids: Token IDs tensor
        """
        processed_smiles = self._preprocess_smiles(smiles)
        
        try:
            encoding = self.tokenizer(
                processed_smiles,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            return encoding['input_ids'].squeeze(0).long()
        except Exception as e:
            logger.warning(f"Tokenization failed for {smiles}: {e}")
            # Return empty encoding
            return torch.zeros(self.config.max_length, dtype=torch.long)
    
    def _get_conditions(self, row: pd.Series) -> np.ndarray:
        """
        Get reaction conditions as array.
        
        Args:
            row: Data row
            
        Returns:
            conditions: Conditions array [temperature, time]
        """
        if 'temperature' in row.index and 'time' in row.index:
            conditions = np.array([row['temperature'], row['time']], dtype=np.float32)
            
            # Apply scaling if scaler is available
            if self.feature_scaler is not None:
                conditions = self.feature_scaler.transform(conditions.reshape(1, -1))[0]
            
            return conditions
        else:
            return np.zeros(2, dtype=np.float32)
    
    def _get_yield(self, yield_value: float) -> float:
        """
        Get processed yield value.
        
        Args:
            yield_value: Raw yield value
            
        Returns:
            processed_yield: Processed yield value
        """
        if self.target_scaler is not None:
            scaled = self.target_scaler.transform([[yield_value]])[0, 0]
            return scaled
        else:
            return yield_value
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Dictionary containing:
                   - reactant_ids: Tokenized reactant SMILES
                   - product_ids: Tokenized product SMILES
                   - features: Combined features tensor
                   - yield_value: Reaction yield
                   - metadata: Additional sample information
        """
        row = self.data.iloc[idx]
        
        try:
            # Get DFT features
            reactant_features = self._get_dft_features(row['reactant_smiles'])
            product_features = self._get_dft_features(row['product_smiles'])
            
            # Get conditions
            conditions = self._get_conditions(row)
            
            # Combine all features
            features = np.concatenate([conditions, reactant_features, product_features])
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Tokenize SMILES
            reactant_ids = self._encode_smiles(row['reactant_smiles'])
            product_ids = self._encode_smiles(row['product_smiles'])
            
            # Get yield
            yield_value = self._get_yield(row['yield'])
            yield_tensor = torch.tensor(yield_value, dtype=torch.float32)
            
            # Prepare metadata
            metadata = {
                'reactant_smiles': row['reactant_smiles'],
                'product_smiles': row['product_smiles'],
                'original_yield': float(row['yield'])
            }
            
            # Add optional fields
            for field in ['temperature', 'time', 'reaction_type']:
                if field in row.index:
                    metadata[field] = row[field]
            
            return {
                'reactant_ids': reactant_ids,
                'product_ids': product_ids,
                'features': features_tensor,
                'yield_value': yield_tensor,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return zero tensors for failed samples
            return {
                'reactant_ids': torch.zeros(self.config.max_length, dtype=torch.long),
                'product_ids': torch.zeros(self.config.max_length, dtype=torch.long),
                'features': torch.zeros(self._get_feature_dimension(), dtype=torch.float32),
                'yield_value': torch.tensor(0.0, dtype=torch.float32),
                'metadata': {'error': str(e)}
            }
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        feature_names = []
        
        # Condition features
        if 'temperature' in self.data.columns:
            feature_names.extend(['temperature', 'time'])
        
        # DFT features
        dft_labels = self.dft_extractor.feature_labels
        feature_names.extend([f'reactant_{label}' for label in dft_labels])
        feature_names.extend([f'product_{label}' for label in dft_labels])
        
        return feature_names
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'n_samples': len(self.data),
            'yield_stats': {
                'mean': float(self.data['yield'].mean()),
                'std': float(self.data['yield'].std()),
                'min': float(self.data['yield'].min()),
                'max': float(self.data['yield'].max()),
                'median': float(self.data['yield'].median())
            }
        }
        
        if 'temperature' in self.data.columns:
            stats['temperature_stats'] = {
                'mean': float(self.data['temperature'].mean()),
                'std': float(self.data['temperature'].std()),
                'min': float(self.data['temperature'].min()),
                'max': float(self.data['temperature'].max())
            }
        
        if 'time' in self.data.columns:
            stats['time_stats'] = {
                'mean': float(self.data['time'].mean()),
                'std': float(self.data['time'].std()),
                'min': float(self.data['time'].min()),
                'max': float(self.data['time'].max())
            }
        
        if 'reaction_type' in self.data.columns:
            stats['reaction_types'] = self.data['reaction_type'].value_counts().to_dict()
        
        return stats
    
    def save_scalers(self, save_dir: str):
        """Save fitted scalers to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.feature_scaler is not None:
            with open(os.path.join(save_dir, 'feature_scaler.pkl'), 'wb') as f:
                pickle.dump(self.feature_scaler, f)
        
        if self.target_scaler is not None:
            with open(os.path.join(save_dir, 'target_scaler.pkl'), 'wb') as f:
                pickle.dump(self.target_scaler, f)
        
        logger.info(f"Scalers saved to {save_dir}")
    
    def load_scalers(self, save_dir: str):
        """Load fitted scalers from disk."""
        feature_scaler_path = os.path.join(save_dir, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(save_dir, 'target_scaler.pkl')
        
        if os.path.exists(feature_scaler_path):
            with open(feature_scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
        
        if os.path.exists(target_scaler_path):
            with open(target_scaler_path, 'rb') as f:
                self.target_scaler = pickle.load(f)
        
        logger.info("Scalers loaded successfully")


def preprocess_data(data: pd.DataFrame, config: DataLoaderConfig = None) -> Tuple[pd.DataFrame, Any]:
    """
    Preprocess reaction data.
    
    Args:
        data: Raw reaction data
        config: Data loader configuration
        
    Returns:
        processed_data: Preprocessed data
        scaler: Fitted feature scaler
    """
    config = config or DataLoaderConfig()
    
    # Create a copy to avoid modifying original data
    processed_data = data.copy()
    
    # Standardize column names
    column_mapping = {
        'reactant': 'reactant_smiles',
        'product': 'product_smiles',
        'yield_value': 'yield',
        'temp': 'temperature',
        'reaction_time': 'time'
    }
    
    processed_data.columns = [column_mapping.get(col, col) for col in processed_data.columns]
    
    # Ensure required columns
    required_columns = ['reactant_smiles', 'product_smiles', 'yield']
    for col in required_columns:
        if col not in processed_data.columns:
            raise ValueError(f"Required column '{col}' not found")
    
    # Remove duplicates
    initial_size = len(processed_data)
    processed_data = processed_data.drop_duplicates(subset=['reactant_smiles', 'product_smiles'])
    if len(processed_data) < initial_size:
        logger.info(f"Removed {initial_size - len(processed_data)} duplicate reactions")
    
    # Handle missing values
    processed_data = processed_data.dropna(subset=required_columns)
    
    # Convert yield to numeric
    processed_data['yield'] = pd.to_numeric(processed_data['yield'], errors='coerce')
    processed_data = processed_data.dropna(subset=['yield'])
    
    # Filter reasonable yield values
    processed_data = processed_data[(processed_data['yield'] >= 0) & (processed_data['yield'] <= 100)]
    
    # Handle condition columns
    for col in ['temperature', 'time']:
        if col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            # Fill missing with median
            if processed_data[col].isna().any():
                median_val = processed_data[col].median()
                processed_data[col] = processed_data[col].fillna(median_val)
                logger.info(f"Filled missing {col} with median: {median_val}")
    
    # Initialize scaler if needed
    scaler = None
    if config.normalize_features and 'temperature' in processed_data.columns:
        feature_columns = ['temperature', 'time']
        feature_data = processed_data[feature_columns].values
        
        if config.feature_scaler == 'standard':
            scaler = StandardScaler()
        elif config.feature_scaler == 'minmax':
            scaler = MinMaxScaler()
        
        if scaler:
            processed_data[feature_columns] = scaler.fit_transform(feature_data)
    
    logger.info(f"Data preprocessing completed: {len(processed_data)} samples")
    
    return processed_data, scaler


def split_dataset(data: pd.DataFrame, config: DataLoaderConfig = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data: Processed reaction data
        config: Data loader configuration
        
    Returns:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
    """
    config = config or DataLoaderConfig()
    
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        data,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=config.shuffle
    )
    
    # Second split: separate validation set from train+val
    val_size_adjusted = config.val_size / (1 - config.test_size)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size_adjusted,
        random_state=config.random_state,
        shuffle=config.shuffle
    )
    
    logger.info(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


def load_reaction_data(file_path: str, format: str = 'auto') -> pd.DataFrame:
    """
    Load reaction data from various file formats.
    
    Args:
        file_path: Path to data file
        format: File format ('csv', 'json', 'excel', 'auto')
        
    Returns:
        data: Loaded reaction data
    """
    if format == 'auto':
        # Infer format from file extension
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            format = 'csv'
        elif ext in ['.json', '.jsonl']:
            format = 'json'
        elif ext in ['.xlsx', '.xls']:
            format = 'excel'
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    try:
        if format == 'csv':
            data = pd.read_csv(file_path)
        elif format == 'json':
            data = pd.read_json(file_path, orient='records')
        elif format == 'excel':
            data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded {len(data)} reactions from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise


def save_reaction_data(data: pd.DataFrame, file_path: str, format: str = 'auto'):
    """
    Save reaction data to file.
    
    Args:
        data: Reaction data to save
        file_path: Output file path
        format: File format ('csv', 'json', 'excel', 'auto')
    """
    if format == 'auto':
        # Infer format from file extension
        ext = Path(file_path).suffix.lower()
        if ext == '.csv':
            format = 'csv'
        elif ext in ['.json', '.jsonl']:
            format = 'json'
        elif ext in ['.xlsx', '.xls']:
            format = 'excel'
        else:
            format = 'csv'  # Default to CSV
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format == 'csv':
            data.to_csv(file_path, index=False)
        elif format == 'json':
            data.to_json(file_path, orient='records', indent=2)
        elif format == 'excel':
            data.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(data)} reactions to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    print("Testing data_loader utilities...")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'reactant_smiles': ['CCO', 'c1ccccc1', 'CC(=O)O'],
        'product_smiles': ['CCOC(=O)O', 'c1ccccc1O', 'CC(=O)OC'],
        'yield': [85.5, 72.3, 91.2],
        'temperature': [80, 100, 60],
        'time': [4, 2, 6]
    })
    
    # Test preprocessing
    try:
        processed_data, scaler = preprocess_data(sample_data)
        print(f"Preprocessed data: {len(processed_data)} samples")
        
        # Test dataset splitting
        train, val, test = split_dataset(processed_data)
        print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Test data loading/saving
        save_reaction_data(processed_data, 'test_data.csv')
        loaded_data = load_reaction_data('test_data.csv')
        print(f"Loaded data: {len(loaded_data)} samples")
        
        # Clean up
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
        
        print("All data_loader tests passed!")
        
    except Exception as e:
        print(f"Data loader test failed: {e}")
