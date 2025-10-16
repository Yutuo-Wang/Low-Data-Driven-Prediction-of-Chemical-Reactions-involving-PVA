# Pre-trained Model Weights

This directory contains pre-trained weights for the PVA-ReAct models.

## Available Models

### EnhancedReactionModel
- **File**: `enhanced_model_weights.pth`
- **Description**: Advanced model with multi-head attention and DFT feature integration
- **Performance**: 95.0% accuracy with 85% training data
- **Best for**: High-accuracy predictions and complex reaction systems

### MultiTaskReactionModel  
- **File**: `multitask_model_weights.pth`
- **Description**: Efficient multi-task model for yield and condition prediction
- **Performance**: 90.4% accuracy with only 40% training data
- **Best for**: Low-data scenarios and rapid prototyping

## Usage Example

```python
from models.model_architectures import EnhancedReactionModel
from models.pretrained_models import get_pretrained_config
import torch

# Load model configuration
config = get_pretrained_config('enhanced')
model = EnhancedReactionModel(**config)

# Load pre-trained weights
weights_path = 'models/pretrained_models/enhanced_model_weights.pth'
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()
