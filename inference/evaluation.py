import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    explained_variance_score,
    mean_absolute_percentage_error
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluator for PVA-ReAct models.
    
    This class provides detailed evaluation metrics and analysis
    for model performance on reaction prediction tasks.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained PVA-ReAct model
            device: Computation device
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.evaluation_history = []
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_dataset(self, data_loader: torch.utils.data.DataLoader,
                        dataset_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            dataset_name: Name of the dataset for reporting
            
        Returns:
            metrics: Comprehensive evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_name} dataset...")
        
        all_yield_preds = []
        all_yield_true = []
        all_condition_preds = []
        all_condition_true = []
        
        # Collect predictions
        with torch.no_grad():
            for batch in data_loader:
                reactant_ids = batch['reactant_ids'].to(self.device)
                product_ids = batch['product_ids'].to(self.device)
                features = batch['features'].to(self.device)
                yield_true = batch['yield_value'].to(self.device)
                
                yield_pred, condition_pred = self.model(reactant_ids, product_ids, features)
                
                all_yield_preds.append(yield_pred.cpu().numpy())
                all_yield_true.append(yield_true.cpu().numpy())
                all_condition_preds.append(condition_pred.cpu().numpy())
                all_condition_true.append(features[:, :2].cpu().numpy())
        
        # Concatenate results
        yield_preds = np.concatenate(all_yield_preds)
        yield_true = np.concatenate(all_yield_true)
        condition_preds = np.concatenate(all_condition_preds)
        condition_true = np.concatenate(all_condition_true)
        
        # Calculate metrics
        yield_metrics = self._calculate_yield_metrics(yield_true, yield_preds)
        condition_metrics = self._calculate_condition_metrics(condition_true, condition_preds)
        combined_metrics = self._calculate_combined_metrics(yield_metrics, condition_metrics)
        
        # Store evaluation results
        evaluation_result = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(yield_true),
            'yield_metrics': yield_metrics,
            'condition_metrics': condition_metrics,
            'combined_metrics': combined_metrics,
            'predictions': {
                'yield_true': yield_true.tolist(),
                'yield_pred': yield_preds.tolist(),
                'condition_true': condition_true.tolist(),
                'condition_pred': condition_preds.tolist()
            }
        }
        
        self.evaluation_history.append(evaluation_result)
        
        logger.info(f"Evaluation completed: R² = {yield_metrics['r2']:.4f}, "
                   f"MAE = {yield_metrics['mae']:.4f}")
        
        return evaluation_result
    
    def _calculate_yield_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive yield prediction metrics.
        
        Args:
            y_true: True yield values
            y_pred: Predicted yield values
            
        Returns:
            metrics: Yield prediction metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # Statistical metrics
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skew'] = stats.skew(residuals)
        
        # Percentage-based accuracy
        absolute_errors = np.abs(residuals)
        metrics['accuracy_5'] = np.mean(absolute_errors < 5.0)  # Within 5%
        metrics['accuracy_10'] = np.mean(absolute_errors < 10.0)  # Within 10%
        metrics['accuracy_15'] = np.mean(absolute_errors < 15.0)  # Within 15%
        
        # Correlation metrics
        correlation, p_value = stats.pearsonr(y_true, y_pred)
        metrics['correlation'] = correlation
        metrics['correlation_p_value'] = p_value
        
        # Additional statistical measures
        metrics['max_error'] = np.max(absolute_errors)
        metrics['median_absolute_error'] = np.median(absolute_errors)
        
        # Prediction intervals (assuming normal distribution of residuals)
        metrics['prediction_interval_95'] = 1.96 * metrics['residual_std']
        
        return metrics
    
    def _calculate_condition_metrics(self, condition_true: np.ndarray, 
                                   condition_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate condition prediction metrics.
        
        Args:
            condition_true: True condition values [temperature, time]
            condition_pred: Predicted condition values [temperature, time]
            
        Returns:
            metrics: Condition prediction metrics
        """
        metrics = {
            'temperature': {},
            'time': {},
            'combined': {}
        }
        
        # Temperature metrics
        temp_true = condition_true[:, 0]
        temp_pred = condition_pred[:, 0]
        
        metrics['temperature']['mse'] = mean_squared_error(temp_true, temp_pred)
        metrics['temperature']['mae'] = mean_absolute_error(temp_true, temp_pred)
        metrics['temperature']['r2'] = r2_score(temp_true, temp_pred)
        metrics['temperature']['correlation'] = stats.pearsonr(temp_true, temp_pred)[0]
        
        # Time metrics
        time_true = condition_true[:, 1]
        time_pred = condition_pred[:, 1]
        
        metrics['time']['mse'] = mean_squared_error(time_true, time_pred)
        metrics['time']['mae'] = mean_absolute_error(time_true, time_pred)
        metrics['time']['r2'] = r2_score(time_true, time_pred)
        metrics['time']['correlation'] = stats.pearsonr(time_true, time_pred)[0]
        
        # Combined condition metrics
        metrics['combined']['mse'] = np.mean([metrics['temperature']['mse'], metrics['time']['mse']])
        metrics['combined']['mae'] = np.mean([metrics['temperature']['mae'], metrics['time']['mae']])
        metrics['combined']['r2'] = np.mean([metrics['temperature']['r2'], metrics['time']['r2']])
        
        return metrics
    
    def _calculate_combined_metrics(self, yield_metrics: Dict[str, float],
                                 condition_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate combined performance metrics.
        
        Args:
            yield_metrics: Yield prediction metrics
            condition_metrics: Condition prediction metrics
            
        Returns:
            metrics: Combined performance metrics
        """
        return {
            'overall_score': 0.7 * yield_metrics['r2'] + 0.3 * condition_metrics['combined']['r2'],
            'weighted_accuracy': 0.8 * yield_metrics['accuracy_10'] + 0.2 * condition_metrics['combined']['r2'],
            'composite_loss': 0.7 * yield_metrics['mse'] + 0.3 * condition_metrics['combined']['mse']
        }
    
    def compare_evaluations(self, evaluation_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple evaluation results.
        
        Args:
            evaluation_names: Names of evaluations to compare
            
        Returns:
            comparison_df: DataFrame with comparison results
        """
        if not evaluation_names:
            evaluation_names = [eval_dict['dataset_name'] for eval_dict in self.evaluation_history]
        
        comparison_data = []
        
        for eval_dict in self.evaluation_history:
            if eval_dict['dataset_name'] in evaluation_names:
                row = {
                    'dataset': eval_dict['dataset_name'],
                    'n_samples': eval_dict['n_samples'],
                    'yield_r2': eval_dict['yield_metrics']['r2'],
                    'yield_mae': eval_dict['yield_metrics']['mae'],
                    'yield_accuracy_10': eval_dict['yield_metrics']['accuracy_10'],
                    'temp_r2': eval_dict['condition_metrics']['temperature']['r2'],
                    'time_r2': eval_dict['condition_metrics']['time']['r2'],
                    'overall_score': eval_dict['combined_metrics']['overall_score']
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_evaluation_results(self, evaluation_result: Dict[str, Any], 
                              save_path: str = None, show: bool = False):
        """
        Plot comprehensive evaluation results.
        
        Args:
            evaluation_result: Evaluation results dictionary
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Model Evaluation - {evaluation_result['dataset_name']}", fontsize=16)
        
        y_true = np.array(evaluation_result['predictions']['yield_true'])
        y_pred = np.array(evaluation_result['predictions']['yield_pred'])
        
        # Plot 1: Yield predictions vs true values
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([0, 100], [0, 100], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('True Yield (%)')
        axes[0, 0].set_ylabel('Predicted Yield (%)')
        axes[0, 0].set_title('Yield Prediction vs True Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = evaluation_result['yield_metrics']['r2']
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        
        # Plot 2: Residuals distribution
        residuals = y_true - y_pred
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Residuals (True - Predicted)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        absolute_errors = np.abs(residuals)
        axes[0, 2].hist(absolute_errors, bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 2].axvline(np.mean(absolute_errors), color='red', linestyle='--', 
                          label=f'MAE: {np.mean(absolute_errors):.2f}')
        axes[0, 2].set_xlabel('Absolute Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Absolute Error Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Temperature predictions
        temp_true = np.array(evaluation_result['predictions']['condition_true'])[:, 0]
        temp_pred = np.array(evaluation_result['predictions']['condition_pred'])[:, 0]
        axes[1, 0].scatter(temp_true, temp_pred, alpha=0.6, s=20, color='green')
        axes[1, 0].plot([temp_true.min(), temp_true.max()], 
                       [temp_true.min(), temp_true.max()], 'r--', alpha=0.8)
        axes[1, 0].set_xlabel('True Temperature (°C)')
        axes[1, 0].set_ylabel('Predicted Temperature (°C)')
        axes[1, 0].set_title('Temperature Prediction')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Time predictions
        time_true = np.array(evaluation_result['predictions']['condition_true'])[:, 1]
        time_pred = np.array(evaluation_result['predictions']['condition_pred'])[:, 1]
        axes[1, 1].scatter(time_true, time_pred, alpha=0.6, s=20, color='purple')
        axes[1, 1].plot([time_true.min(), time_true.max()], 
                       [time_true.min(), time_true.max()], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('True Time (h)')
        axes[1, 1].set_ylabel('Predicted Time (h)')
        axes[1, 1].set_title('Time Prediction')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Metrics summary
        metrics = evaluation_result['yield_metrics']
        metric_names = ['R²', 'MAE', 'RMSE', 'Accuracy@10%']
        metric_values = [metrics['r2'], metrics['mae'], metrics['rmse'], metrics['accuracy_10']]
        
        bars = axes[1, 2].bar(metric_names, metric_values, color=['blue', 'orange', 'red', 'green'])
        axes[1, 2].set_ylabel('Metric Value')
        axes[1, 2].set_title('Key Performance Metrics')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_evaluation_report(self, evaluation_result: Dict[str, Any],
                                 output_path: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_result: Evaluation results
            output_path: Path to save the report
            
        Returns:
            report_path: Path where report was saved
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_report_{timestamp}.json"
        
        # Create comprehensive report
        report = {
            'evaluation_summary': {
                'dataset_name': evaluation_result['dataset_name'],
                'timestamp': evaluation_result['timestamp'],
                'n_samples': evaluation_result['n_samples'],
                'overall_score': evaluation_result['combined_metrics']['overall_score']
            },
            'yield_metrics': evaluation_result['yield_metrics'],
            'condition_metrics': evaluation_result['condition_metrics'],
            'combined_metrics': evaluation_result['combined_metrics'],
            'performance_interpretation': self._interpret_performance(evaluation_result),
            'recommendations': self._generate_recommendations(evaluation_result)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
    
    def _interpret_performance(self, evaluation_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpret model performance and provide insights.
        
        Args:
            evaluation_result: Evaluation results
            
        Returns:
            interpretation: Performance interpretation
        """
        yield_metrics = evaluation_result['yield_metrics']
        condition_metrics = evaluation_result['condition_metrics']
        
        interpretation = {}
        
        # Yield performance interpretation
        r2 = yield_metrics['r2']
        if r2 >= 0.9:
            interpretation['yield_performance'] = "Excellent"
        elif r2 >= 0.8:
            interpretation['yield_performance'] = "Very Good"
        elif r2 >= 0.7:
            interpretation['yield_performance'] = "Good"
        elif r2 >= 0.6:
            interpretation['yield_performance'] = "Fair"
        else:
            interpretation['yield_performance'] = "Needs Improvement"
        
        # Condition performance interpretation
        temp_r2 = condition_metrics['temperature']['r2']
        time_r2 = condition_metrics['time']['r2']
        
        interpretation['temperature_performance'] = "Good" if temp_r2 >= 0.7 else "Needs Improvement"
        interpretation['time_performance'] = "Good" if time_r2 >= 0.7 else "Needs Improvement"
        
        # Overall assessment
        overall_score = evaluation_result['combined_metrics']['overall_score']
        if overall_score >= 0.85:
            interpretation['overall_assessment'] = "Model is ready for production use"
        elif overall_score >= 0.75:
            interpretation['overall_assessment'] = "Model shows good performance, suitable for most applications"
        elif overall_score >= 0.65:
            interpretation['overall_assessment'] = "Model performance is acceptable for limited use cases"
        else:
            interpretation['overall_assessment'] = "Model requires further improvement before deployment"
        
        return interpretation
    
    def _generate_recommendations(self, evaluation_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for model improvement.
        
        Args:
            evaluation_result: Evaluation results
            
        Returns:
            recommendations: List of improvement recommendations
        """
        recommendations = []
        yield_metrics = evaluation_result['yield_metrics']
        condition_metrics = evaluation_result['condition_metrics']
        
        # Yield-related recommendations
        if yield_metrics['r2'] < 0.8:
            recommendations.append("Consider collecting more training data for yield prediction")
        
        if yield_metrics['mae'] > 10:
            recommendations.append("Investigate high-error predictions to identify patterns")
        
        if yield_metrics['accuracy_10'] < 0.8:
            recommendations.append("Focus on improving prediction accuracy for moderate-yield reactions")
        
        # Condition-related recommendations
        if condition_metrics['temperature']['r2'] < 0.7:
            recommendations.append("Temperature prediction may benefit from additional features")
        
        if condition_metrics['time']['r2'] < 0.7:
            recommendations.append("Time prediction could be improved with reaction kinetics data")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Model performance is excellent. Consider deployment for production use")
        
        return recommendations


class ReactionAnalyzer:
    """
    Analyzer for reaction prediction patterns and model behavior.
    
    This class provides insights into how the model performs
    across different types of reactions and conditions.
    """
    
    def __init__(self, evaluator: ModelEvaluator):
        """
        Initialize reaction analyzer.
        
        Args:
            evaluator: Model evaluator instance
        """
        self.evaluator = evaluator
        self.analysis_results = {}
    
    def analyze_reaction_types(self, data_loader: torch.utils.data.DataLoader,
                             reaction_types: List[str] = None) -> Dict[str, Any]:
        """
        Analyze model performance by reaction type.
        
        Args:
            data_loader: Data loader with reaction data
            reaction_types: List of reaction types to analyze
            
        Returns:
            analysis: Performance analysis by reaction type
        """
        # This would typically use metadata about reaction types
        # For now, we'll demonstrate the structure
        
        analysis = {
            'reaction_types': {},
            'summary': {}
        }
        
        # Placeholder for actual implementation
        # In practice, you would filter the data by reaction type
        # and run separate evaluations for each type
        
        logger.info("Reaction type analysis completed")
        return analysis
    
    def analyze_condition_sensitivity(self, reactant_smiles: str, product_smiles: str,
                                    temperature_range: Tuple[float, float] = (50, 150),
                                    time_range: Tuple[float, float] = (1, 8),
                                    n_points: int = 20) -> Dict[str, Any]:
        """
        Analyze model sensitivity to reaction conditions.
        
        Args:
            reactant_smiles: Reactant SMILES
            product_smiles: Product SMILES
            temperature_range: Temperature range to analyze
            time_range: Time range to analyze
            n_points: Number of points per dimension
            
        Returns:
            sensitivity: Sensitivity analysis results
        """
        from .condition_optimizer import EnhancedConditionOptimizer
        
        # This would use the condition optimizer to explore the parameter space
        # For now, return a placeholder structure
        
        sensitivity = {
            'reactant_smiles': reactant_smiles,
            'product_smiles': product_smiles,
            'temperature_range': temperature_range,
            'time_range': time_range,
            'n_points': n_points,
            'sensitivity_analysis': 'Implementation needed'
        }
        
        return sensitivity
    
    def identify_error_patterns(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify patterns in prediction errors.
        
        Args:
            evaluation_result: Evaluation results
            
        Returns:
            error_analysis: Error pattern analysis
        """
        y_true = np.array(evaluation_result['predictions']['yield_true'])
        y_pred = np.array(evaluation_result['predictions']['yield_pred'])
        errors = y_true - y_pred
        
        error_analysis = {
            'high_error_cases': [],
            'error_correlation_with_yield': np.corrcoef(y_true, np.abs(errors))[0, 1],
            'systematic_bias': np.mean(errors),
            'error_distribution': {
                'mean': np.mean(np.abs(errors)),
                'std': np.std(errors),
                'skewness': stats.skew(errors),
                'kurtosis': stats.kurtosis(errors)
            }
        }
        
        # Identify high-error cases (top 10%)
        error_threshold = np.percentile(np.abs(errors), 90)
        high_error_indices = np.where(np.abs(errors) > error_threshold)[0]
        
        for idx in high_error_indices[:10]:  # Limit to first 10
            error_analysis['high_error_cases'].append({
                'index': int(idx),
                'true_yield': float(y_true[idx]),
                'predicted_yield': float(y_pred[idx]),
                'error': float(errors[idx]),
                'absolute_error': float(np.abs(errors[idx]))
            })
        
        return error_analysis


# Utility functions
def evaluate_model_performance(model: torch.nn.Module, 
                             test_loader: torch.utils.data.DataLoader,
                             dataset_name: str = "test") -> Dict[str, Any]:
    """
    Quick function to evaluate model performance.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        dataset_name: Name of the test dataset
        
    Returns:
        evaluation: Evaluation results
    """
    evaluator = ModelEvaluator(model)
    return evaluator.evaluate_dataset(test_loader, dataset_name)


def analyze_reaction_patterns(model: torch.nn.Module,
                            data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
    """
    Quick function to analyze reaction patterns.
    
    Args:
        model: Trained model
        data_loader: Data loader with reaction data
        
    Returns:
        analysis: Reaction pattern analysis
    """
    evaluator = ModelEvaluator(model)
    analyzer = ReactionAnalyzer(evaluator)
    
    evaluation = evaluator.evaluate_dataset(data_loader)
    error_analysis = analyzer.identify_error_patterns(evaluation)
    
    return {
        'evaluation': evaluation,
        'error_analysis': error_analysis
    }


def generate_prediction_report(predictions: List[Dict[str, Any]],
                             output_path: str = None) -> str:
    """
    Generate a comprehensive prediction report.
    
    Args:
        predictions: List of prediction results
        output_path: Output file path
        
    Returns:
        report_path: Path where report was saved
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"prediction_report_{timestamp}.json"
    
    # Calculate summary statistics
    yields = [p['yield'] for p in predictions if 'yield' in p]
    temperatures = [p['temperature'] for p in predictions if 'temperature' in p]
    times = [p['time'] for p in predictions if 'time' in p]
    confidences = [p.get('confidence', 0) for p in predictions]
    
    report = {
        'summary': {
            'total_predictions': len(predictions),
            'average_yield': np.mean(yields) if yields else 0,
            'average_temperature': np.mean(temperatures) if temperatures else 0,
            'average_time': np.mean(times) if times else 0,
            'average_confidence': np.mean(confidences) if confidences else 0,
            'high_confidence_predictions': len([c for c in confidences if c > 0.8])
        },
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Prediction report saved to {output_path}")
    return output_path


# Example usage and testing
if __name__ == "__main__":
    print("Testing ModelEvaluator...")
    
    # Create mock data for testing
    n_samples = 100
    y_true = np.random.uniform(0, 100, n_samples)
    y_pred = y_true + np.random.normal(0, 5, n_samples)  # Add some noise
    
    # Create a mock evaluation result
    mock_evaluation = {
        'dataset_name': 'test',
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'yield_metrics': {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'accuracy_10': np.mean(np.abs(y_true - y_pred) < 10.0)
        },
        'condition_metrics': {
            'temperature': {'r2': 0.75, 'mae': 5.2},
            'time': {'r2': 0.68, 'mae': 0.8},
            'combined': {'r2': 0.715, 'mae': 3.0}
        },
        'combined_metrics': {
            'overall_score': 0.73
        },
        'predictions': {
            'yield_true': y_true.tolist(),
            'yield_pred': y_pred.tolist(),
            'condition_true': np.random.random((n_samples, 2)).tolist(),
            'condition_pred': np.random.random((n_samples, 2)).tolist()
        }
    }
    
    # Test evaluator
    evaluator = ModelEvaluator(None)  # Mock model
    
    try:
        # Test plotting
        evaluator.plot_evaluation_results(mock_evaluation, show=False)
        print("Evaluation plotting test passed")
        
        # Test report generation
        report_path = evaluator.generate_evaluation_report(mock_evaluation)
        print(f"Evaluation report generated: {report_path}")
        
        # Test error analysis
        analyzer = ReactionAnalyzer(evaluator)
        error_analysis = analyzer.identify_error_patterns(mock_evaluation)
        print("Error analysis completed")
        
    except Exception as e:
        print(f"Evaluation test failed: {e}")
    
    print("Model evaluation testing completed!")
