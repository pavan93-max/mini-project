"""
Physics constraints for wind turbine power prediction.
Implements wind power equation and physics-based loss functions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


def wind_power_equation(
    wind_speed: np.ndarray,
    air_density: float = 1.225,  # kg/m³ at sea level, 15°C
    rotor_area: float = 5026.55,  # m² (typical for 80m diameter rotor)
    power_coefficient: float = 0.4,  # Cp, typical max ~0.59 (Betz limit)
    cut_in_speed: float = 3.0,  # m/s
    rated_speed: float = 12.0,  # m/s
    rated_power: float = 2000.0,  # kW
    cut_out_speed: float = 25.0  # m/s
) -> np.ndarray:
    """
    Calculate theoretical wind power using the wind power equation.
    
    P = 0.5 * rho * A * Cp * v^3
    
    Parameters:
    -----------
    wind_speed : np.ndarray
        Wind speed in m/s
    air_density : float
        Air density in kg/m³
    rotor_area : float
        Rotor swept area in m²
    power_coefficient : float
        Power coefficient (Cp), typically 0.3-0.5
    cut_in_speed : float
        Minimum wind speed for power generation (m/s)
    rated_speed : float
        Wind speed at rated power (m/s)
    rated_power : float
        Maximum rated power (kW)
    cut_out_speed : float
        Maximum wind speed before shutdown (m/s)
        
    Returns:
    --------
    np.ndarray
        Theoretical power output (kW)
    """
    wind_speed = np.asarray(wind_speed)
    power = np.zeros_like(wind_speed, dtype=float)
    
    # Below cut-in speed: no power
    mask_below_cutin = wind_speed < cut_in_speed
    power[mask_below_cutin] = 0.0
    
    # Between cut-in and rated: cubic relationship
    mask_operational = (wind_speed >= cut_in_speed) & (wind_speed < rated_speed)
    if np.any(mask_operational):
        # P = 0.5 * rho * A * Cp * v^3
        # Convert from W to kW
        power[mask_operational] = (
            0.5 * air_density * rotor_area * power_coefficient * 
            wind_speed[mask_operational] ** 3 / 1000.0
        )
    
    # At rated speed and above (but below cut-out): constant rated power
    mask_rated = (wind_speed >= rated_speed) & (wind_speed < cut_out_speed)
    power[mask_rated] = rated_power
    
    # Above cut-out: shutdown, no power
    mask_above_cutout = wind_speed >= cut_out_speed
    power[mask_above_cutout] = 0.0
    
    return power


def compute_physics_residual(
    actual_power: np.ndarray,
    theoretical_power: np.ndarray
) -> np.ndarray:
    """
    Compute residual between actual and theoretical power.
    
    Parameters:
    -----------
    actual_power : np.ndarray
        Actual measured power
    theoretical_power : np.ndarray
        Theoretical power from physics equation
        
    Returns:
    --------
    np.ndarray
        Residual (actual - theoretical)
    """
    return actual_power - theoretical_power


class PhysicsLoss(nn.Module):
    """
    Physics-aware loss function that penalizes violations of physical constraints.
    """
    
    def __init__(
        self,
        lambda_physics: float = 0.1,
        lambda_negative: float = 1.0,
        lambda_monotonic: float = 0.5
    ):
        """
        Parameters:
        -----------
        lambda_physics : float
            Weight for physics residual loss
        lambda_negative : float
            Weight for negative power penalty
        lambda_monotonic : float
            Weight for monotonicity penalty
        """
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_negative = lambda_negative
        self.lambda_monotonic = lambda_monotonic
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        wind_speed: Optional[torch.Tensor] = None,
        theoretical_power: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics-aware loss.
        
        Parameters:
        -----------
        predictions : torch.Tensor
            Model predictions (batch_size,)
        targets : torch.Tensor
            Ground truth values (batch_size,)
        wind_speed : torch.Tensor, optional
            Wind speed values for monotonicity check (batch_size,)
        theoretical_power : torch.Tensor, optional
            Theoretical power for physics residual (batch_size,)
            
        Returns:
        --------
        torch.Tensor
            Total loss
        """
        # Data loss (MSE)
        data_loss = self.mse_loss(predictions, targets)
        
        total_loss = data_loss
        
        # Penalty for negative power predictions
        if self.lambda_negative > 0:
            negative_mask = predictions < 0
            negative_penalty = torch.sum(torch.clamp(-predictions[negative_mask], min=0) ** 2)
            total_loss += self.lambda_negative * negative_penalty / len(predictions)
        
        # Monotonicity penalty: power should generally increase with wind speed
        if self.lambda_monotonic > 0 and wind_speed is not None:
            # Sort by wind speed and check if predictions are monotonic
            sorted_indices = torch.argsort(wind_speed)
            sorted_predictions = predictions[sorted_indices]
            sorted_wind = wind_speed[sorted_indices]
            
            # Compute differences
            pred_diffs = sorted_predictions[1:] - sorted_predictions[:-1]
            wind_diffs = sorted_wind[1:] - sorted_wind[:-1]
            
            # Penalize when power decreases while wind speed increases
            violations = (pred_diffs < 0) & (wind_diffs > 0)
            if torch.any(violations):
                monotonic_penalty = torch.sum(pred_diffs[violations] ** 2)
                total_loss += self.lambda_monotonic * monotonic_penalty / len(pred_diffs)
        
        # Physics residual loss
        if self.lambda_physics > 0 and theoretical_power is not None:
            # Residual should be small (model learns correction to physics)
            residual = predictions - theoretical_power
            physics_loss = torch.mean(residual ** 2)
            total_loss += self.lambda_physics * physics_loss
        
        return total_loss


def analyze_power_curve_regions(
    wind_speed: np.ndarray,
    power: np.ndarray,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0
) -> dict:
    """
    Analyze power curve behavior in different regions.
    
    Parameters:
    -----------
    wind_speed : np.ndarray
        Wind speed values
    power : np.ndarray
        Power output values
    cut_in : float
        Cut-in wind speed
    rated : float
        Rated wind speed
    cut_out : float
        Cut-out wind speed
        
    Returns:
    --------
    dict
        Statistics for each region
    """
    regions = {
        'below_cutin': (wind_speed < cut_in),
        'operational': (wind_speed >= cut_in) & (wind_speed < rated),
        'rated': (wind_speed >= rated) & (wind_speed < cut_out),
        'above_cutout': (wind_speed >= cut_out)
    }
    
    stats = {}
    for region_name, mask in regions.items():
        if np.any(mask):
            stats[region_name] = {
                'count': np.sum(mask),
                'mean_power': np.mean(power[mask]),
                'std_power': np.std(power[mask]),
                'mean_wind_speed': np.mean(wind_speed[mask]),
                'max_power': np.max(power[mask]),
                'min_power': np.min(power[mask])
            }
        else:
            stats[region_name] = {'count': 0}
    
    return stats

