import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class EnhancedReactionModel(nn.Module):
    """
    Enhanced multi-task reaction model with DFT feature integration and attention mechanisms.
    
    This model combines:
    - SMILES sequence processing via LSTM
    - DFT quantum chemical feature processing
    - Multi-head attention for feature fusion
    - Multi-task learning for yield prediction and condition design
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, lstm_layers: int = 2,
                 transformer_layers: int = 4, vocab_size: int = 50265, condition_dim: int = 2, 
                 dft_feature_dim: int = 16, dropout_rate: float = 0.1, num_heads: int = 8):
        """
        Initialize the enhanced reaction model.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            lstm_layers: Number of LSTM layers
            transformer_layers: Number of transformer layers
            vocab_size: Size of tokenizer vocabulary
            condition_dim: Dimension of condition output (temperature, time)
            dft_feature_dim: Dimension of DFT features
            dropout_rate: Dropout rate for regularization
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dft_feature_dim = dft_feature_dim
        self.condition_dim = condition_dim
        
        # SMILES encoder components
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, lstm_layers, 
            batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # DFT feature processor
        self.dft_processor = nn.Sequential(
            nn.Linear(dft_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-head attention for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 3, 
            num_heads=num_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 3)
        
        # Prediction heads with enhanced architecture
        self.yield_head = self._create_prediction_head(
            input_dim=hidden_dim * 3, 
            hidden_dims=[hidden_dim, hidden_dim // 2], 
            output_dim=1,
            dropout_rate=dropout_rate
        )
        
        self.condition_head = self._create_prediction_head(
            input_dim=hidden_dim * 3,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            output_dim=condition_dim,
            dropout_rate=dropout_rate
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _create_prediction_head(self, input_dim: int, hidden_dims: list, 
                              output_dim: int, dropout_rate: float) -> nn.Sequential:
        """Create a prediction head with multiple hidden layers."""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)  # Add batch normalization for stability
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        # Set forget gate bias to 1 for better gradient flow
                        if len(param) == self.hidden_dim * 4:
                            param.data[self.hidden_dim:self.hidden_dim*2].fill_(1.0)
    
    def forward(self, reactant_ids: torch.Tensor, product_ids: torch.Tensor, 
                features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            reactant_ids: Tokenized reactant SMILES sequences [batch_size, seq_len]
            product_ids: Tokenized product SMILES sequences [batch_size, seq_len]
            features: Combined features [batch_size, feature_dim] including:
                     - Temperature and time (first 2 features)
                     - DFT features for reactant and product
            
        Returns:
            yield_pred: Predicted reaction yield [batch_size]
            condition_pred: Predicted reaction conditions [batch_size, condition_dim]
        """
        batch_size = reactant_ids.size(0)
        
        # ===== SMILES Sequence Processing =====
        # Embed SMILES tokens
        reactant_emb = self.embedding(reactant_ids)  # [batch_size, seq_len, input_dim]
        product_emb = self.embedding(product_ids)    # [batch_size, seq_len, input_dim]
        
        # Process sequences with LSTM
        reactant_out, (reactant_hidden, _) = self.lstm(reactant_emb)
        product_out, (product_hidden, _) = self.lstm(product_emb)
        
        # Use last hidden state from final LSTM layer
        reactant_final = reactant_hidden[-1]  # [batch_size, hidden_dim]
        product_final = product_hidden[-1]    # [batch_size, hidden_dim]
        
        # ===== DFT Feature Processing =====
        # Extract DFT features (skip temperature and time which are first 2 features)
        dft_features = features[:, 2:2+self.dft_feature_dim]  # [batch_size, dft_feature_dim]
        processed_dft = self.dft_processor(dft_features)  # [batch_size, hidden_dim]
        
        # ===== Feature Fusion =====
        # Combine all feature representations
        combined = torch.cat([reactant_final, product_final, processed_dft], dim=1)  # [batch_size, hidden_dim * 3]
        combined = self.layer_norm(combined)
        
        # Apply multi-head attention for feature interaction
        # Reshape for attention: [batch_size, 1, hidden_dim * 3]
        combined_reshaped = combined.unsqueeze(1)
        attended, attention_weights = self.attention(
            combined_reshaped, 
            combined_reshaped, 
            combined_reshaped
        )
        attended = attended.squeeze(1)  # [batch_size, hidden_dim * 3]
        
        # ===== Multi-task Predictions =====
        yield_pred = self.yield_head(attended).squeeze()  # [batch_size]
        condition_pred = self.condition_head(attended)    # [batch_size, condition_dim]
        
        return yield_pred, condition_pred
    
    def get_attention_weights(self, reactant_ids: torch.Tensor, product_ids: torch.Tensor,
                            features: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for model interpretability.
        
        Returns:
            attention_weights: Attention weights from the multi-head attention layer
        """
        batch_size = reactant_ids.size(0)
        
        # Process through the model until attention layer
        reactant_emb = self.embedding(reactant_ids)
        product_emb = self.embedding(product_ids)
        
        reactant_out, (reactant_hidden, _) = self.lstm(reactant_emb)
        product_out, (product_hidden, _) = self.lstm(product_emb)
        
        reactant_final = reactant_hidden[-1]
        product_final = product_hidden[-1]
        
        dft_features = features[:, 2:2+self.dft_feature_dim]
        processed_dft = self.dft_processor(dft_features)
        
        combined = torch.cat([reactant_final, product_final, processed_dft], dim=1)
        combined = self.layer_norm(combined)
        
        # Get attention weights
        combined_reshaped = combined.unsqueeze(1)
        _, attention_weights = self.attention(
            combined_reshaped, 
            combined_reshaped, 
            combined_reshaped,
            need_weights=True
        )
        
        return attention_weights


class MultiTaskReactionModel(nn.Module):
    """
    Multi-task reaction model for simultaneous yield prediction and condition estimation.
    
    This model uses:
    - Transformer architecture for sequence processing
    - Shared feature extraction for multi-task learning
    - Simple but effective architecture for rapid prototyping
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, lstm_layers: int = 2,
                 transformer_layers: int = 4, vocab_size: int = 50265, condition_dim: int = 2,
                 nhead: int = 8, dim_feedforward: int = 512, dropout: float = 0.1):
        """
        Initialize the multi-task reaction model.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            lstm_layers: Number of LSTM layers
            transformer_layers: Number of transformer encoder layers
            vocab_size: Size of tokenizer vocabulary
            condition_dim: Dimension of condition output
            nhead: Number of attention heads in transformer
            dim_feedforward: Dimension of feedforward network in transformer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        
        # Shared feature extraction layers
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, lstm_layers, 
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Transformer encoder for enhanced sequence understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
        # Projection layer to reduce dimension
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multi-task output heads
        self.yield_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Yield between 0-100%
        )
        
        self.condition_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, condition_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, reactant_ids: torch.Tensor, product_ids: torch.Tensor, 
                features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multi-task model.
        
        Args:
            reactant_ids: Tokenized reactant SMILES [batch_size, seq_len]
            product_ids: Tokenized product SMILES [batch_size, seq_len]
            features: Additional features (DFT, conditions) [batch_size, feature_dim]
            
        Returns:
            yield_pred: Predicted yield [batch_size]
            condition_pred: Predicted conditions [batch_size, condition_dim]
        """
        # Shared feature extraction
        reactant_emb = self.embedding(reactant_ids)  # [batch_size, seq_len, input_dim]
        product_emb = self.embedding(product_ids)    # [batch_size, seq_len, input_dim]
        
        # LSTM sequence processing
        reactant_out, _ = self.lstm(reactant_emb)    # [batch_size, seq_len, hidden_dim]
        product_out, _ = self.lstm(product_emb)      # [batch_size, seq_len, hidden_dim]
        
        # Feature concatenation along sequence dimension
        combined = torch.cat([reactant_out, product_out], dim=2)  # [batch_size, seq_len, hidden_dim * 2]
        combined = self.proj(combined)  # [batch_size, seq_len, hidden_dim]
        
        # Transformer processing
        transformer_out = self.transformer(combined)  # [batch_size, seq_len, hidden_dim]
        
        # Global average pooling over sequence dimension
        pooled = transformer_out.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Multi-task output
        yield_pred = self.yield_head(pooled).squeeze(-1) * 100  # Convert to percentage
        condition_pred = self.condition_head(pooled)
        
        return yield_pred, condition_pred
    
    def predict_yield(self, reactant_ids: torch.Tensor, product_ids: torch.Tensor,
                     features: torch.Tensor) -> torch.Tensor:
        """Convenience method for yield prediction only."""
        yield_pred, _ = self.forward(reactant_ids, product_ids, features)
        return yield_pred
    
    def predict_conditions(self, reactant_ids: torch.Tensor, product_ids: torch.Tensor,
                          features: torch.Tensor) -> torch.Tensor:
        """Convenience method for condition prediction only."""
        _, condition_pred = self.forward(reactant_ids, product_ids, features)
        return condition_pred


class PVAAttentionMechanism(nn.Module):
    """
    Custom attention mechanism for PVA reaction patterns.
    
    This module captures specific attention patterns relevant to PVA chemistry,
    such as hydroxyl group interactions and polymer chain dynamics.
    """
    
    def __init__(self, hidden_dim: int, pva_specific_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pva_specific_dim = pva_specific_dim
        
        # PVA-specific attention queries
        self.hydroxyl_query = nn.Parameter(torch.randn(pva_specific_dim, hidden_dim))
        self.chain_query = nn.Parameter(torch.randn(pva_specific_dim, hidden_dim))
        
        # Attention projection
        self.query_proj = nn.Linear(hidden_dim, pva_specific_dim)
        self.key_proj = nn.Linear(hidden_dim, pva_specific_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        nn.init.xavier_uniform_(self.hydroxyl_query)
        nn.init.xavier_uniform_(self.chain_query)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PVA-specific attention.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            
        Returns:
            attended: Attended features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        queries = self.query_proj(x)  # [batch_size, seq_len, pva_specific_dim]
        keys = self.key_proj(x)       # [batch_size, seq_len, pva_specific_dim]
        values = self.value_proj(x)   # [batch_size, seq_len, hidden_dim]
        
        # PVA-specific attention patterns
        hydroxyl_attention = torch.matmul(queries, self.hydroxyl_query.t())  # [batch_size, seq_len, pva_specific_dim]
        chain_attention = torch.matmul(queries, self.chain_query.t())         # [batch_size, seq_len, pva_specific_dim]
        
        # Combine attention scores
        attention_scores = hydroxyl_attention + chain_attention  # [batch_size, seq_len, pva_specific_dim]
        attention_weights = F.softmax(attention_scores, dim=1)   # [batch_size, seq_len, pva_specific_dim]
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights.transpose(1, 2), values)  # [batch_size, pva_specific_dim, hidden_dim]
        attended_values = attended_values.transpose(1, 2)  # [batch_size, hidden_dim, pva_specific_dim]
        
        # Combine with original features
        combined = torch.cat([x, attended_values], dim=2)  # [batch_size, seq_len, hidden_dim * 2]
        output = self.output_proj(combined)  # [batch_size, seq_len, hidden_dim]
        output = self.layer_norm(output + x)  # Residual connection
        
        return output


# Utility function to create model from configuration
def create_model(model_type: str = 'enhanced', **kwargs) -> nn.Module:
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model to create ('enhanced' or 'multitask')
        **kwargs: Model configuration parameters
        
    Returns:
        model: Initialized model instance
    """
    model_configs = {
        'enhanced': EnhancedReactionModel,
        'multitask': MultiTaskReactionModel
    }
    
    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_configs.keys())}")
    
    model_class = model_configs[model_type]
    return model_class(**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Test the models
    batch_size, seq_len, vocab_size = 4, 50, 1000
    
    # Create sample inputs
    reactant_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    product_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    features = torch.randn(batch_size, 20)  # 2 conditions + 16 DFT features
    
    # Test EnhancedReactionModel
    print("Testing EnhancedReactionModel...")
    enhanced_model = EnhancedReactionModel(vocab_size=vocab_size)
    yield_pred, condition_pred = enhanced_model(reactant_ids, product_ids, features)
    print(f"Enhanced model - Yield shape: {yield_pred.shape}, Condition shape: {condition_pred.shape}")
    print(f"Enhanced model parameters: {count_parameters(enhanced_model):,}")
    
    # Test MultiTaskReactionModel
    print("\nTesting MultiTaskReactionModel...")
    multitask_model = MultiTaskReactionModel(vocab_size=vocab_size)
    yield_pred, condition_pred = multitask_model(reactant_ids, product_ids, features)
    print(f"Multitask model - Yield shape: {yield_pred.shape}, Condition shape: {condition_pred.shape}")
    print(f"Multitask model parameters: {count_parameters(multitask_model):,}")
    
    # Test factory function
    print("\nTesting model factory...")
    model = create_model('enhanced', vocab_size=vocab_size, hidden_dim=128)
    print(f"Factory-created model parameters: {count_parameters(model):,}")
    
    print("\nAll models tested successfully!")
