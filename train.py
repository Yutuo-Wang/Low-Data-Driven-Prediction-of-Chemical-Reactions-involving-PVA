import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

def train_enhanced(model, loader, optimizer, scheduler, epochs=2):
    model.train()
    scaler = GradScaler()
    
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            reactant_ids = batch['reactant_ids'].to(device)
            product_ids = batch['product_ids'].to(device)
            features = batch['features'].to(device)
            yield_value = batch['yield_value'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():  # Mixed precision training
                output = model(reactant_ids, product_ids, features)
                loss = nn.MSELoss()(output.squeeze(), yield_value)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
