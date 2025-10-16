import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import json
import os
from pathlib import Path

# Import from our modules
from models.model_architectures import EnhancedReactionModel, MultiTaskReactionModel
from features.dft_extractor import EnhancedDFTFeatureExtractor, DFTFeatureExtractor
from features.smiles_processor import SMILESProcessor, SMILESProcessorConfig
from utils.data_loader import preprocess_data

logger = logging.getLogger(__name__)

class PVAReactionPredictor:
    """
    Main PVA reaction predictor for inference and prediction tasks.
    
    This class provides a unified interface for:
    - Loading trained models
    - Making yield and condition predictions
    - Batch prediction on multiple reactions
    - Result analysis and reporting
    """
    
    def __init__(self, model_path: str, config_path: str = None, device: str = None):
        """
        Initialize the PVA reaction predictor.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to model configuration file (optional)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.dft_extractor = None
        self.scaler = None
        self.config = {}
        
        # Load model and components
        self._load_model_and_components()
        
        logger.info(f"PVAReactionPredictor initialized on device: {self.device}")
    
    def _load_model_and_components(self):
        """Load model and all necessary components."""
        try:
            # Load configuration if provided
            if self.config_path and os.path.exists(self.config_path):
                import yaml
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            
            # Initialize tokenizer
            tokenizer_name = self.config.get('smiles', {}).get('tokenizer_name', 
                                                             "seyonec/PubChem10M_SMILES_BPE_396_250")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize DFT extractor
            dft_config = self.config.get('dft', {})
            self.dft_extractor = EnhancedDFTFeatureExtractor(
                basis_set=dft_config.get('basis_set', 'def2svp'),
                functional=dft_config.get('functional', 'B3LYP'),
                save_orbitals=dft_config.get('save_orbitals', False)
            )
            
            # Determine model type and initialize
            model_config = self.config.get('model', {})
            model_type = model_config.get('type', 'enhanced')
            
            if model_type == 'enhanced':
                self.model = EnhancedReactionModel(
                    vocab_size=self.tokenizer.vocab_size,
                    **{k: v for k, v in model_config.items() if k != 'type'}
                )
            else:
                self.model = MultiTaskReactionModel(
                    vocab_size=self.tokenizer.vocab_size,
                    **{k: v for k, v in model_config.items() if k != 'type'}
                )
            
            # Load model weights
            self.model = self._load_model_weights(self.model, self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler if available
            scaler_path = os.path.join(os.path.dirname(self.model_path), 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = torch.load(scaler_path)
            
            logger.info(f"Model loaded successfully: {model_type}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model and components: {e}")
            raise
    
    def _load_model_weights(self, model: torch.nn.Module, model_path: str) -> torch.nn.Module:
        """
        Load model weights from checkpoint.
        
        Args:
            model: Model instance
            model_path: Path to model weights
            
        Returns:
            model: Model with loaded weights
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    
    def _preprocess_inputs(self, reactant_smiles: str, product_smiles: str, 
                          temperature: float = None, time: float = None) -> Dict[str, torch.Tensor]:
        """
        Preprocess input data for model inference.
        
        Args:
            reactant_smiles: Reactant SMILES string
            product_smiles: Product SMILES string
            temperature: Reaction temperature (optional)
            time: Reaction time (optional)
            
        Returns:
            inputs: Dictionary of preprocessed inputs
        """
        try:
            # Calculate DFT features
            reactant_features = self.dft_extractor.calculate(reactant_smiles)[0]
            product_features = self.dft_extractor.calculate(product_smiles)[0]
            
            # Prepare condition features
            if temperature is not None and time is not None and self.scaler is not None:
                # Scale conditions if scaler is available
                conditions = np.array([[temperature, time]], dtype=np.float32)
                conditions_scaled = self.scaler.transform(conditions)[0]
            else:
                conditions_scaled = np.array([0.0, 0.0])  # Default values
            
            # Combine all features
            features = np.concatenate([conditions_scaled, reactant_features, product_features])
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Tokenize SMILES
            def encode_smiles(smiles):
                return self.tokenizer(
                    smiles,
                    padding='max_length',
                    truncation=True,
                    max_length=200,
                    return_tensors='pt'
                )['input_ids'].squeeze(0).long()
            
            reactant_ids = encode_smiles(reactant_smiles).unsqueeze(0).to(self.device)
            product_ids = encode_smiles(product_smiles).unsqueeze(0).to(self.device)
            
            return {
                'reactant_ids': reactant_ids,
                'product_ids': product_ids,
                'features': features_tensor
            }
            
        except Exception as e:
            logger.error(f"Input preprocessing failed: {e}")
            raise
    
    def _postprocess_outputs(self, yield_pred: torch.Tensor, condition_pred: torch.Tensor,
                           temperature: float = None, time: float = None) -> Dict[str, Any]:
        """
        Postprocess model outputs to meaningful values.
        
        Args:
            yield_pred: Raw yield prediction
            condition_pred: Raw condition prediction
            temperature: Original temperature (for reference)
            time: Original time (for reference)
            
        Returns:
            results: Processed prediction results
        """
        try:
            # Convert tensors to numpy
            yield_value = yield_pred.cpu().numpy()[0] if yield_pred.dim() > 0 else yield_pred.cpu().numpy()
            
            # Process condition predictions
            condition_array = condition_pred.cpu().numpy()[0]
            
            # If scaler is available, inverse transform conditions
            if self.scaler is not None:
                # Create dummy array for inverse transform
                dummy_conditions = np.zeros((1, len(self.scaler.mean_)))
                dummy_conditions[0, :2] = condition_array
                original_conditions = self.scaler.inverse_transform(dummy_conditions)[0, :2]
                pred_temperature, pred_time = original_conditions
            else:
                pred_temperature, pred_time = condition_array
            
            # Ensure yield is within reasonable bounds
            yield_value = np.clip(yield_value, 0, 100)
            
            # Calculate confidence score (simplified)
            confidence = self._calculate_confidence(yield_value, condition_array)
            
            return {
                'yield': float(yield_value),
                'temperature': float(pred_temperature),
                'time': float(pred_time),
                'confidence': float(confidence),
                'original_temperature': temperature,
                'original_time': time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Output postprocessing failed: {e}")
            return {
                'yield': 0.0,
                'temperature': 0.0,
                'time': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_confidence(self, yield_pred: float, condition_pred: np.ndarray) -> float:
        """
        Calculate prediction confidence score.
        
        Args:
            yield_pred: Predicted yield
            condition_pred: Predicted conditions
            
        Returns:
            confidence: Confidence score between 0 and 1
        """
        # Base confidence on yield prediction stability
        yield_confidence = min(1.0, yield_pred / 100.0)
        
        # Confidence based on condition predictions (more stable = higher confidence)
        condition_stability = 1.0 - min(1.0, np.std(condition_pred) / 10.0)
        
        # Combined confidence
        confidence = 0.7 * yield_confidence + 0.3 * condition_stability
        
        return np.clip(confidence, 0, 1)
    
    def predict_single_reaction(self, reactant_smiles: str, product_smiles: str,
                               temperature: float = None, time: float = None) -> Dict[str, Any]:
        """
        Predict yield and conditions for a single reaction.
        
        Args:
            reactant_smiles: Reactant SMILES string
            product_smiles: Product SMILES string
            temperature: Reaction temperature (optional)
            time: Reaction time (optional)
            
        Returns:
            prediction: Dictionary with prediction results
        """
        try:
            # Validate inputs
            if not reactant_smiles or not product_smiles:
                raise ValueError("Reactant and product SMILES are required")
            
            logger.info(f"Predicting reaction: {reactant_smiles} -> {product_smiles}")
            
            # Preprocess inputs
            inputs = self._preprocess_inputs(reactant_smiles, product_smiles, temperature, time)
            
            # Model inference
            with torch.no_grad():
                yield_pred, condition_pred = self.model(
                    inputs['reactant_ids'],
                    inputs['product_ids'],
                    inputs['features']
                )
            
            # Postprocess outputs
            results = self._postprocess_outputs(yield_pred, condition_pred, temperature, time)
            results['reactant_smiles'] = reactant_smiles
            results['product_smiles'] = product_smiles
            
            logger.info(f"Prediction completed: Yield={results['yield']:.1f}%, "
                       f"Temp={results['temperature']:.1f}°C, Time={results['time']:.1f}h")
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'error': str(e),
                'reactant_smiles': reactant_smiles,
                'product_smiles': product_smiles,
                'yield': 0.0,
                'temperature': 0.0,
                'time': 0.0,
                'confidence': 0.0
            }
    
    def predict_batch_reactions(self, reactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict yields and conditions for multiple reactions.
        
        Args:
            reactions: List of reaction dictionaries with:
                      - reactant_smiles
                      - product_smiles
                      - temperature (optional)
                      - time (optional)
            
        Returns:
            predictions: List of prediction results
        """
        predictions = []
        
        for i, reaction in enumerate(reactions):
            try:
                logger.info(f"Processing reaction {i+1}/{len(reactions)}")
                
                prediction = self.predict_single_reaction(
                    reactant_smiles=reaction['reactant_smiles'],
                    product_smiles=reaction['product_smiles'],
                    temperature=reaction.get('temperature'),
                    time=reaction.get('time')
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Failed to process reaction {i+1}: {e}")
                predictions.append({
                    'error': str(e),
                    'reactant_smiles': reaction.get('reactant_smiles', ''),
                    'product_smiles': reaction.get('product_smiles', ''),
                    'yield': 0.0,
                    'temperature': 0.0,
                    'time': 0.0,
                    'confidence': 0.0
                })
        
        return predictions
    
    def predict_from_dataframe(self, df: pd.DataFrame, 
                              reactant_col: str = 'reactant_smiles',
                              product_col: str = 'product_smiles',
                              temperature_col: str = None,
                              time_col: str = None) -> pd.DataFrame:
        """
        Predict reactions from a pandas DataFrame.
        
        Args:
            df: DataFrame containing reaction data
            reactant_col: Column name for reactant SMILES
            product_col: Column name for product SMILES
            temperature_col: Column name for temperature (optional)
            time_col: Column name for time (optional)
            
        Returns:
            results_df: DataFrame with prediction results
        """
        reactions = []
        
        for idx, row in df.iterrows():
            reaction_data = {
                'reactant_smiles': row[reactant_col],
                'product_smiles': row[product_col]
            }
            
            if temperature_col and temperature_col in row:
                reaction_data['temperature'] = row[temperature_col]
            if time_col and time_col in row:
                reaction_data['time'] = row[time_col]
            
            reactions.append(reaction_data)
        
        predictions = self.predict_batch_reactions(reactions)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(predictions)
        
        # Add original data
        results_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        return results_df
    
    def save_predictions(self, predictions: List[Dict[str, Any]], 
                        output_path: str = None) -> str:
        """
        Save prediction results to file.
        
        Args:
            predictions: List of prediction results
            output_path: Output file path (optional)
            
        Returns:
            output_path: Path where results were saved
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pva_predictions_{timestamp}.json"
        
        # Convert to DataFrame for better handling
        df = pd.DataFrame(predictions)
        
        if output_path.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        elif output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        else:
            output_path = output_path + '.json'
            df.to_json(output_path, orient='records', indent=2)
        
        logger.info(f"Predictions saved to {output_path}")
        return output_path


# Utility functions for quick predictions
def load_trained_model(model_path: str, config_path: str = None) -> PVAReactionPredictor:
    """
    Quick function to load a trained model.
    
    Args:
        model_path: Path to trained model
        config_path: Path to model configuration
        
    Returns:
        predictor: Initialized PVA reaction predictor
    """
    return PVAReactionPredictor(model_path, config_path)


def predict_reaction_yield(reactant_smiles: str, product_smiles: str,
                          model_path: str, temperature: float = None, 
                          time: float = None) -> float:
    """
    Quick function to predict reaction yield.
    
    Args:
        reactant_smiles: Reactant SMILES
        product_smiles: Product SMILES
        model_path: Path to trained model
        temperature: Reaction temperature
        time: Reaction time
        
    Returns:
        yield_pred: Predicted yield
    """
    predictor = load_trained_model(model_path)
    result = predictor.predict_single_reaction(reactant_smiles, product_smiles, temperature, time)
    return result['yield']


def predict_reaction_conditions(reactant_smiles: str, product_smiles: str,
                               model_path: str) -> Tuple[float, float]:
    """
    Quick function to predict reaction conditions.
    
    Args:
        reactant_smiles: Reactant SMILES
        product_smiles: Product SMILES
        model_path: Path to trained model
        
    Returns:
        temperature: Predicted temperature
        time: Predicted time
    """
    predictor = load_trained_model(model_path)
    result = predictor.predict_single_reaction(reactant_smiles, product_smiles)
    return result['temperature'], result['time']


def batch_predict(reaction_list: List[Dict[str, Any]], 
                 model_path: str) -> List[Dict[str, Any]]:
    """
    Quick function for batch predictions.
    
    Args:
        reaction_list: List of reaction dictionaries
        model_path: Path to trained model
        
    Returns:
        predictions: List of prediction results
    """
    predictor = load_trained_model(model_path)
    return predictor.predict_batch_reactions(reaction_list)


# Example usage and testing
if __name__ == "__main__":
    # Test the predictor
    print("Testing PVAReactionPredictor...")
    
    # Example usage
    try:
        # Initialize predictor
        predictor = PVAReactionPredictor(
            model_path="final_reaction_model.pth",
            config_path="config.yaml"
        )
        
        # Single prediction
        result = predictor.predict_single_reaction(
            reactant_smiles="CCO",
            product_smiles="CCOC(=O)O",
            temperature=80,
            time=4
        )
        
        print("Single prediction result:")
        print(json.dumps(result, indent=2))
        
        # Batch prediction
        reactions = [
            {
                'reactant_smiles': 'CCO',
                'product_smiles': 'CCOC(=O)O',
                'temperature': 80,
                'time': 4
            },
            {
                'reactant_smiles': 'c1ccccc1',
                'product_smiles': 'c1ccccc1O',
                'temperature': 100,
                'time': 2
            }
        ]
        
        batch_results = predictor.predict_batch_reactions(reactions)
        print(f"\nBatch predictions completed: {len(batch_results)} reactions")
        
        # Save results
        predictor.save_predictions(batch_results, "test_predictions.json")
        
    except Exception as e:
        print(f"Prediction test failed: {e}")
    
    print("Prediction testing completed!")
