import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import RobertaTokenizer
from dft_feature_extractor import DFTFeatureExtractor

class ReactionDataset(Dataset):
    def __init__(self, data, tokenizer, dft_extractor, max_length=200):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.dft_extractor = dft_extractor
        self.max_length = max_length
        self.feature_cache = {}

    def _get_dft_features(self, smiles):
        if not isinstance(smiles, str) or len(smiles) == 0:
            return np.zeros(len(self.dft_extractor.feature_labels))
        
        if smiles in self.feature_cache:
            return self.feature_cache[smiles]
        
        features, _ = self.dft_extractor.calculate(smiles)
        self.feature_cache[smiles] = features
        return features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get reactant and product SMILES features
        reactant_features = self._get_dft_features(row['reactant smiles'])
        product_features = self._get_dft_features(row['product smiles'])
        
        # Concatenate reactant and product features
        features = np.concatenate([
            row[['temperature', 'time']].values.astype(np.float32),
            reactant_features,
            product_features
        ])
        
        def encode(smiles):
            if not isinstance(smiles, str) or len(smiles) == 0:
                smiles = "C"  # Placeholder for invalid SMILES
            return self.tokenizer(
                smiles,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )['input_ids'].squeeze(0).long()
        
        return {
            'reactant_ids': encode(row['reactant smiles']),
            'product_ids': encode(row['product smiles']),
            'features': torch.tensor(features, dtype=torch.float32),
            'yield_value': torch.tensor(float(row['yield']), dtype=torch.float32)
        }
