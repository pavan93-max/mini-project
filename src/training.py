"""
Training utilities for wind power prediction models.
Handles data loading, training loops, and model checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Callable
import json
from tqdm import tqdm


class TimeSeriesDataset(Dataset):
    """Dataset for time-series sequences."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int,
        forecast_horizon: int = 1
    ):
        """
        Parameters:
        -----------
        X : np.ndarray
            Features (n_samples, n_features)
        y : np.ndarray
            Targets (n_samples,)
        sequence_length : int
            Input sequence length
        forecast_horizon : int
            Forecast horizon
        """
        from preprocessing import prepare_sequences
        
        self.X_seq, self.y_seq = prepare_sequences(
            X, y, sequence_length, forecast_horizon, stride=1
        )
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X_seq[idx]), torch.FloatTensor(self.y_seq[idx])


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    physics_loss_fn: Optional[Callable] = None,
    wind_speed: Optional[torch.Tensor] = None,
    theoretical_power: Optional[torch.Tensor] = None
) -> float:
    """
    Train for one epoch.
    
    Parameters:
    -----------
    model : nn.Module
        Model to train
    dataloader : DataLoader
        Training dataloader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : str
        Device
    physics_loss_fn : Callable, optional
        Physics-aware loss function
    wind_speed : torch.Tensor, optional
        Wind speed for physics loss
    theoretical_power : torch.Tensor, optional
        Theoretical power for physics loss
        
    Returns:
    --------
    float
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        
        if physics_loss_fn is not None:
            # Get wind speed and theoretical power for this batch if available
            batch_ws = None
            batch_theoretical = None
            if wind_speed is not None:
                # This is a simplified version - in practice, you'd need to extract
                # wind speed from batch_X or pass it separately
                pass
            loss = physics_loss_fn(predictions, batch_y, batch_ws, batch_theoretical)
        else:
            loss = criterion(predictions, batch_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Validate model.
    
    Parameters:
    -----------
    model : nn.Module
        Model to validate
    dataloader : DataLoader
        Validation dataloader
    criterion : nn.Module
        Loss function
    device : str
        Device
        
    Returns:
    --------
    float
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    save_dir: Optional[Path] = None,
    early_stopping_patience: int = 10,
    physics_loss_fn: Optional[Callable] = None
) -> Dict[str, list]:
    """
    Train model with early stopping.
    
    Parameters:
    -----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    epochs : int
        Number of epochs
    device : str
        Device
    save_dir : Path, optional
        Directory to save checkpoints
    early_stopping_patience : int
        Early stopping patience
    physics_loss_fn : Callable, optional
        Physics-aware loss function
        
    Returns:
    --------
    dict
        Training history
    """
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, physics_loss_fn
        )
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history

