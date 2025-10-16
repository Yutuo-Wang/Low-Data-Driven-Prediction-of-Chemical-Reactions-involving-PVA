import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from scipy.optimize import minimize
from copy import deepcopy
import time

# Import from our modules
from features.dft_extractor import EnhancedDFTFeatureExtractor
from features.smiles_processor import SMILESProcessor

logger = logging.getLogger(__name__)

class ConditionOptimizer:
    """
    Reaction condition optimizer for finding optimal temperature and time.
    
    This class uses optimization algorithms to find reaction conditions
    that maximize predicted yield while satisfying constraints.
    """
    
    def __init__(self, model: torch.nn.Module, dft_extractor: EnhancedDFTFeatureExtractor,
                 tokenizer: any, scaler: any, device: torch.device):
        """
        Initialize condition optimizer.
        
        Args:
            model: Trained PVA-ReAct model
            dft_extractor: DFT feature extractor
            tokenizer: SMILES tokenizer
            scaler: Feature scaler
            device: Computation device
        """
        self.model = model
        self.dft_extractor = dft_extractor
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.device = device
        
        self.model.eval()
        
        # Default bounds for optimization
        self.temp_bounds = (0, 200)  # °C
        self.time_bounds = (0.1, 24)  # hours
        
        # Optimization configuration
        self.optimization_config = {
            'method': 'L-BFGS-B',
            'max_iter': 100,
            'tol': 1e-4,
            'disp': False
        }
        
        logger.info("ConditionOptimizer initialized")
    
    def _prepare_fixed_features(self, reactant_smiles: str, product_smiles: str) -> Dict[str, torch.Tensor]:
        """
        Prepare fixed features that don't change during optimization.
        
        Args:
            reactant_smiles: Reactant SMILES
            product_smiles: Product SMILES
            
        Returns:
            fixed_features: Dictionary of fixed input features
        """
        # Calculate DFT features (fixed during optimization)
        reactant_features = self.dft_extractor.calculate(reactant_smiles)[0]
        product_features = self.dft_extractor.calculate(product_smiles)[0]
        
        # Tokenize SMILES (fixed during optimization)
        def encode_smiles(smiles):
            return self.tokenizer(
                smiles,
                padding='max_length',
                truncation=True,
                max_length=200,
                return_tensors='pt'
            )['input_ids'].squeeze(0).long()
        
        reactant_ids = encode_smiles(reactant_smiles)
        product_ids = encode_smiles(product_smiles)
        
        return {
            'reactant_ids': reactant_ids,
            'product_ids': product_ids,
            'reactant_features': reactant_features,
            'product_features': product_features
        }
    
    def _predict_yield(self, conditions: np.ndarray, fixed_features: Dict[str, torch.Tensor]) -> float:
        """
        Predict yield for given conditions.
        
        Args:
            conditions: Array of [temperature, time] (scaled)
            fixed_features: Fixed input features
            
        Returns:
            yield_pred: Predicted yield
        """
        try:
            # Prepare full feature vector
            features = np.concatenate([
                conditions,
                fixed_features['reactant_features'],
                fixed_features['product_features']
            ])
            
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Expand fixed inputs to batch size 1
            reactant_ids = fixed_features['reactant_ids'].unsqueeze(0).to(self.device)
            product_ids = fixed_features['product_ids'].unsqueeze(0).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                yield_pred, _ = self.model(reactant_ids, product_ids, features_tensor)
            
            return yield_pred.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Yield prediction failed: {e}")
            return 0.0
    
    def _objective_function(self, conditions: np.ndarray, fixed_features: Dict[str, torch.Tensor]) -> float:
        """
        Objective function for optimization (to be minimized).
        
        Args:
            conditions: Scaled conditions [temp, time]
            fixed_features: Fixed input features
            
        Returns:
            negative_yield: Negative of predicted yield (for minimization)
        """
        yield_pred = self._predict_yield(conditions, fixed_features)
        return -yield_pred  # Negative for minimization
    
    def _scale_conditions(self, temperature: float, time: float) -> np.ndarray:
        """
        Scale conditions using the fitted scaler.
        
        Args:
            temperature: Temperature in °C
            time: Time in hours
            
        Returns:
            scaled_conditions: Scaled conditions array
        """
        if self.scaler is not None:
            conditions = np.array([[temperature, time]], dtype=np.float32)
            scaled = self.scaler.transform(conditions)[0]
            return scaled
        else:
            return np.array([temperature / 100.0, time / 10.0])  # Simple scaling
    
    def _unscale_conditions(self, scaled_conditions: np.ndarray) -> Tuple[float, float]:
        """
        Unscale conditions to original units.
        
        Args:
            scaled_conditions: Scaled conditions array
            
        Returns:
            temperature: Temperature in °C
            time: Time in hours
        """
        if self.scaler is not None:
            # Create dummy array for inverse transform
            dummy = np.zeros((1, len(self.scaler.mean_)))
            dummy[0, :2] = scaled_conditions
            original = self.scaler.inverse_transform(dummy)[0, :2]
            return original[0], original[1]
        else:
            return scaled_conditions[0] * 100.0, scaled_conditions[1] * 10.0
    
    def _get_scaled_bounds(self) -> List[Tuple[float, float]]:
        """
        Get scaled bounds for optimization.
        
        Returns:
            scaled_bounds: Scaled bounds for temperature and time
        """
        if self.scaler is not None:
            # Scale the bounds
            bounds_array = np.array([list(self.temp_bounds), list(self.time_bounds)])
            scaled_bounds = self.scaler.transform(
                np.concatenate([bounds_array, np.zeros((2, len(self.scaler.mean_) - 2))], axis=1)
            )[:, :2]
            
            return [(scaled_bounds[0, 0], scaled_bounds[1, 0]), 
                    (scaled_bounds[0, 1], scaled_bounds[1, 1])]
        else:
            # Simple scaling
            return [(self.temp_bounds[0] / 100.0, self.temp_bounds[1] / 100.0),
                    (self.time_bounds[0] / 10.0, self.time_bounds[1] / 10.0)]
    
    def optimize_conditions(self, reactant_smiles: str, product_smiles: str,
                          initial_temp: float = 50.0, initial_time: float = 2.0,
                          method: str = None, max_iter: int = None) -> Dict[str, any]:
        """
        Optimize reaction conditions to maximize yield.
        
        Args:
            reactant_smiles: Reactant SMILES
            product_smiles: Product SMILES
            initial_temp: Initial temperature guess
            initial_time: Initial time guess
            method: Optimization method
            max_iter: Maximum iterations
            
        Returns:
            results: Optimization results
        """
        start_time = time.time()
        
        try:
            # Prepare fixed features
            fixed_features = self._prepare_fixed_features(reactant_smiles, product_smiles)
            
            # Scale initial conditions
            x0 = self._scale_conditions(initial_temp, initial_time)
            
            # Get scaled bounds
            bounds = self._get_scaled_bounds()
            
            # Update optimization config
            config = self.optimization_config.copy()
            if method:
                config['method'] = method
            if max_iter:
                config['max_iter'] = max_iter
            
            # Perform optimization
            result = minimize(
                fun=self._objective_function,
                x0=x0,
                args=(fixed_features,),
                bounds=bounds,
                **{k: v for k, v in config.items() if k != 'disp'}
            )
            
            # Process results
            optimal_scaled = result.x
            optimal_temp, optimal_time = self._unscale_conditions(optimal_scaled)
            optimal_yield = -result.fun  # Convert back from negative
            
            # Calculate initial yield for comparison
            initial_yield = self._predict_yield(x0, fixed_features)
            
            optimization_time = time.time() - start_time
            
            results = {
                'success': result.success,
                'optimal_temperature': optimal_temp,
                'optimal_time': optimal_time,
                'predicted_yield': optimal_yield,
                'initial_temperature': initial_temp,
                'initial_time': initial_time,
                'initial_yield': initial_yield,
                'improvement': optimal_yield - initial_yield,
                'iterations': result.nit,
                'optimization_time': optimization_time,
                'message': result.message,
                'reactant_smiles': reactant_smiles,
                'product_smiles': product_smiles
            }
            
            logger.info(f"Optimization completed: "
                       f"Temp={optimal_temp:.1f}°C, Time={optimal_time:.1f}h, "
                       f"Yield={optimal_yield:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Condition optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'reactant_smiles': reactant_smiles,
                'product_smiles': product_smiles
            }
    
    def grid_search_conditions(self, reactant_smiles: str, product_smiles: str,
                             temp_range: Tuple[float, float] = None,
                             time_range: Tuple[float, float] = None,
                             temp_steps: int = 10, time_steps: int = 10) -> Dict[str, any]:
        """
        Perform grid search over condition space.
        
        Args:
            reactant_smiles: Reactant SMILES
            product_smiles: Product SMILES
            temp_range: Temperature range (min, max)
            time_range: Time range (min, max)
            temp_steps: Number of temperature steps
            time_steps: Number of time steps
            
        Returns:
            results: Grid search results
        """
        start_time = time.time()
        
        # Use default ranges if not provided
        temp_range = temp_range or self.temp_bounds
        time_range = time_range or self.time_bounds
        
        # Prepare fixed features
        fixed_features = self._prepare_fixed_features(reactant_smiles, product_smiles)
        
        # Generate grid
        temp_values = np.linspace(temp_range[0], temp_range[1], temp_steps)
        time_values = np.linspace(time_range[0], time_range[1], time_steps)
        
        grid_results = []
        best_yield = -1
        best_conditions = None
        
        # Evaluate all grid points
        for temp in temp_values:
            for time_val in time_values:
                scaled_conditions = self._scale_conditions(temp, time_val)
                yield_pred = self._predict_yield(scaled_conditions, fixed_features)
                
                result_point = {
                    'temperature': temp,
                    'time': time_val,
                    'yield': yield_pred
                }
                grid_results.append(result_point)
                
                if yield_pred > best_yield:
                    best_yield = yield_pred
                    best_conditions = (temp, time_val)
        
        grid_time = time.time() - start_time
        
        return {
            'best_temperature': best_conditions[0],
            'best_time': best_conditions[1],
            'best_yield': best_yield,
            'grid_results': grid_results,
            'temp_range': temp_range,
            'time_range': time_range,
            'temp_steps': temp_steps,
            'time_steps': time_steps,
            'total_evaluations': len(grid_results),
            'grid_time': grid_time,
            'reactant_smiles': reactant_smiles,
            'product_smiles': product_smiles
        }
    
    def set_bounds(self, temp_bounds: Tuple[float, float], time_bounds: Tuple[float, float]):
        """
        Set bounds for condition optimization.
        
        Args:
            temp_bounds: Temperature bounds (min, max)
            time_bounds: Time bounds (min, max)
        """
        self.temp_bounds = temp_bounds
        self.time_bounds = time_bounds
        logger.info(f"Bounds updated: Temp={temp_bounds}, Time={time_bounds}")


class EnhancedConditionOptimizer(ConditionOptimizer):
    """
    Enhanced condition optimizer with additional features.
    
    This class extends the basic optimizer with:
    - Multiple optimization algorithms
    - Constraint handling
    - Multi-start optimization
    - Sensitivity analysis
    """
    
    def __init__(self, model: torch.nn.Module, dft_extractor: EnhancedDFTFeatureExtractor,
                 tokenizer: any, scaler: any, device: torch.device):
        """Initialize enhanced condition optimizer."""
        super().__init__(model, dft_extractor, tokenizer, scaler, device)
        
        # Additional optimization methods
        self.available_methods = ['L-BFGS-B', 'SLSQP', 'trust-constr', 'COBYLA']
        
        # Multi-start configuration
        self.n_restarts = 5
        
        # Sensitivity analysis configuration
        self.sensitivity_steps = 5
    
    def multi_start_optimization(self, reactant_smiles: str, product_smiles: str,
                               n_starts: int = None) -> Dict[str, any]:
        """
        Perform multi-start optimization from different initial points.
        
        Args:
            reactant_smiles: Reactant SMILES
            product_smiles: Product SMILES
            n_starts: Number of starting points
            
        Returns:
            results: Multi-start optimization results
        """
        n_starts = n_starts or self.n_restarts
        
        # Generate random initial points within bounds
        np.random.seed(42)  # For reproducibility
        
        initial_points = []
        for _ in range(n_starts):
            temp = np.random.uniform(self.temp_bounds[0], self.temp_bounds[1])
            time_val = np.random.uniform(self.time_bounds[0], self.time_bounds[1])
            initial_points.append((temp, time_val))
        
        # Run optimization from each starting point
        all_results = []
        best_result = None
        best_yield = -1
        
        for i, (temp, time_val) in enumerate(initial_points):
            logger.info(f"Multi-start optimization {i+1}/{n_starts}")
            
            result = self.optimize_conditions(
                reactant_smiles, product_smiles, temp, time_val
            )
            
            all_results.append(result)
            
            if result.get('success', False) and result['predicted_yield'] > best_yield:
                best_yield = result['predicted_yield']
                best_result = result
        
        return {
            'best_result': best_result,
            'all_results': all_results,
            'n_starts': n_starts,
            'best_yield': best_yield,
            'reactant_smiles': reactant_smiles,
            'product_smiles': product_smiles
        }
    
    def sensitivity_analysis(self, reactant_smiles: str, product_smiles: str,
                           base_temp: float = None, base_time: float = None,
                           temp_variation: float = 20.0, time_variation: float = 2.0) -> Dict[str, any]:
        """
        Perform sensitivity analysis around base conditions.
        
        Args:
            reactant_smiles: Reactant SMILES
            product_smiles: Product SMILES
            base_temp: Base temperature
            base_time: Base time
            temp_variation: Temperature variation range
            time_variation: Time variation range
            
        Returns:
            results: Sensitivity analysis results
        """
        # Use optimized conditions if base not provided
        if base_temp is None or base_time is None:
            opt_result = self.optimize_conditions(reactant_smiles, product_smiles)
            if opt_result['success']:
                base_temp = opt_result['optimal_temperature']
                base_time = opt_result['optimal_time']
            else:
                base_temp = 80.0
                base_time = 4.0
        
        # Prepare fixed features
        fixed_features = self._prepare_fixed_features(reactant_smiles, product_smiles)
        
        # Temperature sensitivity
        temp_sensitivity = []
        temp_range = np.linspace(base_temp - temp_variation, base_temp + temp_variation, 
                               self.sensitivity_steps)
        
        for temp in temp_range:
            scaled_conditions = self._scale_conditions(temp, base_time)
            yield_pred = self._predict_yield(scaled_conditions, fixed_features)
            temp_sensitivity.append({
                'temperature': temp,
                'yield': yield_pred,
                'time': base_time
            })
        
        # Time sensitivity
        time_sensitivity = []
        time_range = np.linspace(base_time - time_variation, base_time + time_variation,
                               self.sensitivity_steps)
        
        for time_val in time_range:
            scaled_conditions = self._scale_conditions(base_temp, time_val)
            yield_pred = self._predict_yield(scaled_conditions, fixed_features)
            time_sensitivity.append({
                'temperature': base_temp,
                'time': time_val,
                'yield': yield_pred
            })
        
        return {
            'base_temperature': base_temp,
            'base_time': base_time,
            'temp_sensitivity': temp_sensitivity,
            'time_sensitivity': time_sensitivity,
            'temp_variation': temp_variation,
            'time_variation': time_variation,
            'reactant_smiles': reactant_smiles,
            'product_smiles': product_smiles
        }
    
    def compare_optimization_methods(self, reactant_smiles: str, product_smiles: str,
                                   initial_temp: float = 50.0, initial_time: float = 2.0) -> Dict[str, any]:
        """
        Compare different optimization methods.
        
        Args:
            reactant_smiles: Reactant SMILES
            product_smiles: Product SMILES
            initial_temp: Initial temperature
            initial_time: Initial time
            
        Returns:
            results: Method comparison results
        """
        method_results = {}
        
        for method in self.available_methods:
            try:
                logger.info(f"Testing optimization method: {method}")
                
                result = self.optimize_conditions(
                    reactant_smiles, product_smiles, initial_temp, initial_time, method
                )
                
                method_results[method] = {
                    'success': result['success'],
                    'optimal_temperature': result.get('optimal_temperature', 0),
                    'optimal_time': result.get('optimal_time', 0),
                    'predicted_yield': result.get('predicted_yield', 0),
                    'iterations': result.get('iterations', 0),
                    'optimization_time': result.get('optimization_time', 0),
                    'message': result.get('message', '')
                }
                
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                method_results[method] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Find best method
        best_method = None
        best_yield = -1
        
        for method, result in method_results.items():
            if result['success'] and result['predicted_yield'] > best_yield:
                best_yield = result['predicted_yield']
                best_method = method
        
        return {
            'method_results': method_results,
            'best_method': best_method,
            'best_yield': best_yield,
            'reactant_smiles': reactant_smiles,
            'product_smiles': product_smiles
        }


# Utility functions
def optimize_reaction_conditions(reactant_smiles: str, product_smiles: str,
                               model_path: str, config_path: str = None,
                               method: str = 'L-BFGS-B') -> Dict[str, any]:
    """
    Quick function to optimize reaction conditions.
    
    Args:
        reactant_smiles: Reactant SMILES
        product_smiles: Product SMILES
        model_path: Path to trained model
        config_path: Path to model configuration
        method: Optimization method
        
    Returns:
        results: Optimization results
    """
    from .predict_main import load_trained_model
    
    predictor = load_trained_model(model_path, config_path)
    
    optimizer = EnhancedConditionOptimizer(
        model=predictor.model,
        dft_extractor=predictor.dft_extractor,
        tokenizer=predictor.tokenizer,
        scaler=predictor.scaler,
        device=predictor.device
    )
    
    return optimizer.optimize_conditions(reactant_smiles, product_smiles, method=method)


def validate_reaction_conditions(reactant_smiles: str, product_smiles: str,
                               temperature: float, time: float,
                               model_path: str, config_path: str = None) -> Dict[str, any]:
    """
    Validate reaction conditions and predict yield.
    
    Args:
        reactant_smiles: Reactant SMILES
        product_smiles: Product SMILES
        temperature: Temperature to validate
        time: Time to validate
        model_path: Path to trained model
        config_path: Path to model configuration
        
    Returns:
        results: Validation results
    """
    from .predict_main import load_trained_model
    
    predictor = load_trained_model(model_path, config_path)
    
    # Use the predictor to get yield prediction
    result = predictor.predict_single_reaction(
        reactant_smiles, product_smiles, temperature, time
    )
    
    # Add validation information
    result['validation'] = {
        'temperature_provided': temperature,
        'time_provided': time,
        'temperature_difference': abs(result['temperature'] - temperature),
        'time_difference': abs(result['time'] - time),
        'conditions_reasonable': result['yield'] > 50.0  # Simple heuristic
    }
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("Testing ConditionOptimizer...")
    
    # This would typically be used with a loaded model
    # For testing, we'll create a mock scenario
    
    class MockModel:
        def __init__(self):
            pass
        
        def eval(self):
            pass
        
        def __call__(self, reactant_ids, product_ids, features):
            # Mock prediction that prefers moderate temperatures and times
            batch_size = reactant_ids.shape[0]
            temp = features[:, 0].cpu().numpy() * 100  # Assuming simple scaling
            time_val = features[:, 1].cpu().numpy() * 10
            
            # Simple yield model for testing
            optimal_temp = 80.0
            optimal_time = 4.0
            
            yield_pred = 100.0 - 0.1 * (temp - optimal_temp)**2 - 2.0 * (time_val - optimal_time)**2
            yield_pred = np.clip(yield_pred, 0, 100)
            
            condition_pred = features[:, :2]  # Return conditions as-is
            
            return torch.tensor(yield_pred, dtype=torch.float32), condition_pred
    
    # Mock other components
    mock_model = MockModel()
    mock_dft = EnhancedDFTFeatureExtractor()
    mock_tokenizer = None  # Would be real tokenizer in practice
    mock_scaler = None
    mock_device = torch.device('cpu')
    
    # Test optimization
    try:
        optimizer = ConditionOptimizer(
            mock_model, mock_dft, mock_tokenizer, mock_scaler, mock_device
        )
        
        # Test with example reaction
        results = optimizer.optimize_conditions(
            reactant_smiles="CCO",
            product_smiles="CCOC(=O)O",
            initial_temp=50.0,
            initial_time=2.0
        )
        
        print("Optimization results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Optimization test failed: {e}")
    
    print("Condition optimization testing completed!")
