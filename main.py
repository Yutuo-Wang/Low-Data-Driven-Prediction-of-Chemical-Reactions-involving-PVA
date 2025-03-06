import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from dft_feature_extractor import DFTFeatureExtractor
from data_preprocessing import preprocess_data
from dataset import ReactionDataset
from models import DFTEnhancedModel
from train import train_enhanced
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dft_extractor = DFTFeatureExtractor(
        basis='def2svp',
        xc='B3LYP',
        save_orbital=True
    )
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    data = pd.read_csv("data/reaction_data.csv")
    data = preprocess_data(data)
    
    dataset = ReactionDataset(data, tokenizer, dft_extractor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = DFTEnhancedModel(input_dim=128, hidden_dim=256, lstm_layers=2, transformer_layers=4, vocab_size=len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_enhanced(model, train_loader, optimizer, scheduler, epochs=5)
