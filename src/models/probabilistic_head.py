"""
Probabilistic output heads for uncertainty-aware predictions.
Implements quantile regression and Gaussian negative log-likelihood.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class QuantileRegressionHead(nn.Module):
    """
    Quantile regression head for probabilistic predictions.
    Outputs multiple quantiles (e.g., 0.1, 0.5, 0.9) for uncertainty estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Parameters:
        -----------
        input_dim : int
            Input dimension from backbone
        quantiles : List[float]
            List of quantiles to predict
        hidden_dim : int
            Hidden layer dimension
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.quantiles = sorted(quantiles)
        self.n_quantiles = len(quantiles)
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Quantile-specific outputs
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(self.n_quantiles)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input (batch_size, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Quantile predictions (batch_size, n_quantiles)
        """
        x = self.shared(x)
        quantile_preds = [output(x) for output in self.quantile_outputs]
        return torch.cat(quantile_preds, dim=1)  # (batch_size, n_quantiles)
    
    def get_median(self, quantile_preds: torch.Tensor) -> torch.Tensor:
        """Extract median (0.5 quantile) prediction."""
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        return quantile_preds[:, median_idx]
    
    def get_uncertainty_bounds(
        self,
        quantile_preds: torch.Tensor,
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get lower and upper uncertainty bounds.
        
        Parameters:
        -----------
        quantile_preds : torch.Tensor
            Quantile predictions (batch_size, n_quantiles)
        lower_quantile : float
            Lower quantile for bound
        upper_quantile : float
            Upper quantile for bound
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Lower and upper bounds
        """
        lower_idx = self.quantiles.index(lower_quantile) if lower_quantile in self.quantiles else 0
        upper_idx = self.quantiles.index(upper_quantile) if upper_quantile in self.quantiles else -1
        return quantile_preds[:, lower_idx], quantile_preds[:, upper_idx]


class GaussianNLLHead(nn.Module):
    """
    Gaussian negative log-likelihood head.
    Predicts mean and variance for Gaussian uncertainty.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        min_variance: float = 1e-6
    ):
        """
        Parameters:
        -----------
        input_dim : int
            Input dimension from backbone
        hidden_dim : int
            Hidden layer dimension
        dropout : float
            Dropout rate
        min_variance : float
            Minimum variance to prevent numerical issues
        """
        super().__init__()
        self.min_variance = min_variance
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Mean and variance outputs
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input (batch_size, input_dim)
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and variance predictions
        """
        x = self.shared(x)
        mean = self.mean_head(x).squeeze(-1)
        # Ensure variance is positive
        var = F.softplus(self.var_head(x)).squeeze(-1) + self.min_variance
        return mean, var
    
    def get_uncertainty_bounds(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        coverage: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get uncertainty bounds assuming Gaussian distribution.
        
        Parameters:
        -----------
        mean : torch.Tensor
            Mean predictions
        var : torch.Tensor
            Variance predictions
        coverage : float
            Desired coverage (e.g., 0.8 for 80% confidence interval)
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Lower and upper bounds
        """
        std = torch.sqrt(var)
        # Z-score for desired coverage (e.g., 1.28 for 80% CI)
        z_score = torch.tensor(1.28 if coverage == 0.8 else 1.96 if coverage == 0.95 else 1.0)
        lower = mean - z_score * std
        upper = mean + z_score * std
        return lower, upper


def quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float]
) -> torch.Tensor:
    """
    Quantile loss (pinball loss) for quantile regression.
    
    Parameters:
    -----------
    predictions : torch.Tensor
        Quantile predictions (batch_size, n_quantiles)
    targets : torch.Tensor
        Ground truth (batch_size,)
    quantiles : List[float]
        List of quantiles
        
    Returns:
    --------
    torch.Tensor
        Quantile loss
    """
    errors = targets.unsqueeze(1) - predictions  # (batch_size, n_quantiles)
    quantiles_tensor = torch.tensor(quantiles, device=errors.device).unsqueeze(0)
    
    loss = torch.max(
        quantiles_tensor * errors,
        (quantiles_tensor - 1) * errors
    )
    return loss.mean()


def gaussian_nll_loss(
    mean: torch.Tensor,
    var: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.
    
    Parameters:
    -----------
    mean : torch.Tensor
        Mean predictions (batch_size,)
    var : torch.Tensor
        Variance predictions (batch_size,)
    targets : torch.Tensor
        Ground truth (batch_size,)
        
    Returns:
    --------
    torch.Tensor
        NLL loss
    """
    return 0.5 * (torch.log(var) + (targets - mean) ** 2 / var).mean()

