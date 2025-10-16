import torch
import torch.nn as nn

class EnhancedReactionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, transformer_layers, vocab_size, condition_dim=2, dft_feature_dim=16):
        super().__init__()
        
        # SMILES encoder
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, dropout=0.1)
        
        # DFT feature processor
        self.dft_processor = nn.Sequential(
            nn.Linear(dft_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Multi-head attention for feature fusion
        self.attention = nn.MultiheadAttention(hidden_dim * 3, num_heads=8, dropout=0.1, batch_first=True)
        
        # Prediction heads
        self.yield_head = self._create_head(hidden_dim * 3, output_dim=1)
        self.condition_head = self._create_head(hidden_dim * 3, output_dim=condition_dim)
        
    def _create_head(self, input_dim, output_dim=1):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, reactant_ids, product_ids, features):
        batch_size = reactant_ids.size(0)
        
        # Process SMILES
        reactant_emb = self.embedding(reactant_ids)
        product_emb = self.embedding(product_ids)
        
        reactant_out, _ = self.lstm(reactant_emb)
        product_out, _ = self.lstm(product_emb)
        
        # Use last hidden states
        reactant_final = reactant_out[:, -1, :]
        product_final = product_out[:, -1, :]
        
        # Process DFT features
        dft_features = self.dft_processor(features[:, 2:])  # Skip temperature and time
        
        # Combine all features
        combined = torch.cat([reactant_final, product_final, dft_features], dim=1)
        
        # Apply attention
        attended, _ = self.attention(
            combined.unsqueeze(1), 
            combined.unsqueeze(1), 
            combined.unsqueeze(1)
        )
        attended = attended.squeeze(1)
        
        # Predictions
        yield_pred = self.yield_head(attended).squeeze()
        condition_pred = self.condition_head(attended)
        
        return yield_pred, condition_pred


class MultiTaskReactionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, lstm_layers=2,
                 transformer_layers=4, vocab_size=50265, condition_dim=2):
        super().__init__()

        # Shared feature extraction layers
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            num_encoder_layers=transformer_layers,
            batch_first=True
        )
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Multi-task output heads
        self.yield_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.condition_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, condition_dim)
        )

    def forward(self, reactant_ids, product_ids, features):
        # Shared feature extraction
        reactant_emb = self.embedding(reactant_ids)
        product_emb = self.embedding(product_ids)

        # LSTM sequence processing
        reactant_out, _ = self.lstm(reactant_emb)
        product_out, _ = self.lstm(product_emb)

        # Feature concatenation
        combined = torch.cat([reactant_out, product_out], dim=2)
        combined = self.proj(combined)

        # Transformer processing
        transformer_out = self.transformer(combined, combined)
        pooled = transformer_out[:, -1, :]  # Take the last position of the sequence

        # Multi-task output (ensure correct dimensions)
        yield_pred = self.yield_head(pooled)
        condition_pred = self.condition_head(pooled)

        return yield_pred.squeeze(-1), condition_pred
