import torch
import torch.nn as nn

class DFTEnhancedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, transformer_layers, vocab_size):
        super(DFTEnhancedModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            num_encoder_layers=transformer_layers
        )
        self.fc = nn.Linear(hidden_dim, 1)  # Output layer
        
    def forward(self, reactant_ids, product_ids, features):
        lstm_out_reactant, _ = self.lstm(reactant_ids)
        lstm_out_product, _ = self.lstm(product_ids)
        
        combined_lstm_out = torch.cat([lstm_out_reactant, lstm_out_product], dim=2)
        src = combined_lstm_out
        tgt = combined_lstm_out
        
        transformer_out = self.transformer(src, tgt)
        
        output = self.fc(transformer_out[:, -1, :])  # Use the last LSTM output
        return output
