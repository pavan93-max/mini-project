"""
Baseline models for wind power prediction.
Includes persistence, linear regression, Random Forest, and LSTM.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE, handling zero values."""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[list] = None
) -> Dict[str, float]:
    """
    Evaluate model predictions using multiple metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    metrics : list, optional
        List of metric names to compute
        
    Returns:
    --------
    dict
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    
    results = {}
    
    if 'MAE' in metrics:
        results['MAE'] = mean_absolute_error(y_true, y_pred)
    
    if 'RMSE' in metrics:
        results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if 'MAPE' in metrics:
        results['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    
    if 'R2' in metrics:
        results['R2'] = r2_score(y_true, y_pred)
    
    return results


class PersistenceModel:
    """
    Persistence model: predicts next value as the last observed value.
    """
    
    def __init__(self):
        self.last_value = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit persistence model (stores last value)."""
        if len(y) > 0:
            self.last_value = y[-1]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using last observed value."""
        if self.last_value is None:
            raise ValueError("Model not fitted")
        return np.full(len(X), self.last_value)


class LSTMModel(nn.Module):
    """
    LSTM model for time-series prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            LSTM hidden state size
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        output_size : int
            Output size (typically 1 for regression)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor (batch_size, seq_length, input_size)
            
        Returns:
        --------
        torch.Tensor
            Predictions (batch_size, output_size)
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        output = self.fc(self.dropout(last_output))
        return output.squeeze(-1) if output.shape[-1] == 1 else output


def train_lstm(
    model: LSTMModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, list]:
    """
    Train LSTM model.
    
    Parameters:
    -----------
    model : LSTMModel
        LSTM model instance
    X_train : np.ndarray
        Training sequences (n_samples, seq_length, n_features)
    y_train : np.ndarray
        Training targets (n_samples,)
    X_val : np.ndarray
        Validation sequences
    y_val : np.ndarray
        Validation targets
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    device : str
        Device ('cpu' or 'cuda')
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    dict
        Training history with 'train_loss' and 'val_loss' lists
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert to tensors
    # Ensure y is 1D (squeeze if needed)
    if len(y_train.shape) > 1:
        y_train = y_train.squeeze()
    if len(y_val.shape) > 1:
        y_val = y_val.squeeze()
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor).item()
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return history

