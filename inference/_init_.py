"""
PVA-ReAct Inference and Prediction Framework
============================================

This package contains inference, prediction, and optimization modules for the PVA-ReAct framework.

Available Modules:
- predict_main: Main prediction script and model inference
- condition_optimizer: Reaction condition optimization algorithms
- evaluation: Model evaluation and performance analysis
"""

from .predict_main import (
    PVAReactionPredictor,
    load_trained_model,
    predict_reaction_yield,
    predict_reaction_conditions,
    batch_predict
)

from .condition_optimizer import (
    ConditionOptimizer,
    EnhancedConditionOptimizer,
    optimize_reaction_conditions,
    validate_reaction_conditions
)

from .evaluation import (
    ModelEvaluator,
    ReactionAnalyzer,
    evaluate_model_performance,
    analyze_reaction_patterns,
    generate_prediction_report
)

__all__ = [
    # Prediction
    'PVAReactionPredictor',
    'load_trained_model',
    'predict_reaction_yield',
    'predict_reaction_conditions',
    'batch_predict',
    
    # Condition Optimization
    'ConditionOptimizer',
    'EnhancedConditionOptimizer',
    'optimize_reaction_conditions',
    'validate_reaction_conditions',
    
    # Evaluation
    'ModelEvaluator',
    'ReactionAnalyzer',
    'evaluate_model_performance',
    'analyze_reaction_patterns',
    'generate_prediction_report'
]

__version__ = '1.0.0'
