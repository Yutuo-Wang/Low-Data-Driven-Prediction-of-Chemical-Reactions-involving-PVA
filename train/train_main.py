import torch
import pandas as pd
from tqdm import tqdm
from ..utils.config import device, config  # Relative import
from ..utils.data_loader import preprocess_data, split_dataset, create_loaders
from ..models.model_architectures import EnhancedReactionModel, save_model
from ..training.early_stopping import EarlyStopping
from ..training.training_utils import TrainingHistory
from ..inference.evaluation import evaluate_model
from ..features.dft_extractor import EnhancedDFTFeatureExtractor
from transformers import AutoTokenizer

def train_multitask(model, train_loader, val_loader, optimizer, scheduler, epochs=2000, early_stopping_patience=10):
    model.train()
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    history = TrainingHistory()
    
    best_r2 = -float('inf')
    best_model_state = None

    for epoch in range(epochs):
        total_loss = 0.0
        yield_losses = 0.0
        cond_losses = 0.0

        # Training phase
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()

            reactant_ids = batch['reactant_ids'].to(device)
            product_ids = batch['product_ids'].to(device)
            features = batch['features'].to(device)
            yield_true = batch['yield_value'].to(device)

            with autocast():
                yield_pred, cond_pred = model(reactant_ids, product_ids, features)

                loss_yield = nn.MSELoss()(yield_pred, yield_true)
                loss_cond = nn.L1Loss()(cond_pred, features[:, :2])
                loss = loss_yield + 0.3 * loss_cond

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            yield_losses += loss_yield.item()
            cond_losses += loss_cond.item()

        # Validation evaluation
        model.eval()
        eval_metrics = evaluate_model(model, val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history.update(total_loss / len(train_loader), eval_metrics, current_lr)

        # Print training progress
        avg_total = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}")
        print(f"Training Loss: {avg_total:.4f} | Yield Loss: {yield_losses / len(train_loader):.4f} | Condition Loss: {cond_losses / len(train_loader):.4f}")
        print(f"Validation R²: {eval_metrics['r2']:.4f} | MAE: {eval_metrics['mae']:.4f} | Accuracy (±5%): {eval_metrics['accuracy']:.2%}")
        print(f"Learning Rate: {current_lr:.2e}")

        # Save best model
        if eval_metrics['r2'] > best_r2:
            best_r2 = eval_metrics['r2']
            best_model_state = model.state_dict().copy()
            save_model(model, 'best_reaction_model.pth')
            print(f"New best model saved with R²: {best_r2:.4f}")

        # Learning rate adjustment
        scheduler.step(eval_metrics['r2'])
        
        # Early stopping check
        early_stopping(eval_metrics['r2'])
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with R²: {best_r2:.4f}")
    
    # Plot training history
    history.plot()
    
    return model

if __name__ == "__main__":
    # Configuration (overridden by utils/config.py if needed)
    print("Initializing reaction yield prediction system...")

    # Load data
    try:
        data = pd.read_csv("data/reaction_data.csv")  # Assuming data/ dir
        print(f"Loaded dataset with {len(data)} samples")
    except FileNotFoundError:
        print("Error: reaction_data.csv not found")
        print("Please ensure the data file exists in the data/ directory")
        exit(1)

    # Initialize components
    dft_extractor = EnhancedDFTFeatureExtractor(basis='def2svp', xc='B3LYP')
    tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data preprocessing and splitting
    data, scaler = preprocess_data(data)
    train_data, val_data, test_data = split_dataset(data)
    
    print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Create datasets and loaders
    train_loader, val_loader, test_loader = create_loaders(
        train_data, val_data, test_data, tokenizer, dft_extractor,
        batch_size=config['batch_size'], max_length=config['max_length']
    )

    # Initialize model
    model = EnhancedReactionModel(
        input_dim=128,
        hidden_dim=config['hidden_dim'],
        lstm_layers=2,
        transformer_layers=4,
        vocab_size=tokenizer.vocab_size,
        dft_feature_dim=len(dft_extractor.feature_labels)
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training configuration
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Train model
    print("\nStarting model training...")
    trained_model = train_multitask(
        model, train_loader, val_loader, optimizer, scheduler,
        epochs=config['epochs'], early_stopping_patience=config['early_stopping_patience']
    )

    # Save final model
    save_model(trained_model, 'final_reaction_model.pth')
    print("Final model saved")

    # Test set evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(trained_model, test_loader)
    print("\nTest set evaluation:")
    print(f"R²: {test_metrics['r2']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.2%}")
