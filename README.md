# Physics-Aware Probabilistic Wind Power Output Prediction using Time-Series Transformers

A state-of-the-art wind turbine power output prediction system that combines deep learning with physical constraints to produce accurate and uncertainty-aware predictions.

## Project Overview

This research project implements a physics-aware probabilistic wind power prediction system that:

1. **Achieves competitive accuracy** compared to existing baselines (Persistence, Linear Regression, Random Forest, LSTM)
2. **Incorporates physical constraints** from wind turbine power equations
3. **Produces probabilistic predictions** with uncertainty quantification, not just point estimates
4. **Is fully reproducible** with well-documented code and clear experimental protocols

## Problem Motivation

Wind power prediction is critical for:
- **Grid stability**: Accurate forecasts enable better integration of renewable energy
- **Economic optimization**: Power producers can optimize bidding strategies
- **Maintenance planning**: Predictions help schedule maintenance during low-wind periods

Traditional machine learning models often ignore physical constraints, leading to unrealistic predictions (e.g., negative power, violations of power curve monotonicity). This project addresses these limitations by integrating physics into the learning process.

## Dataset

The project uses SCADA (Supervisory Control and Data Acquisition) data from wind turbines:
- **Source**: `Aventa_AV7_IET_OST_SCADA.csv`
- **Location**: `data/raw/`
- **Features**: Wind speed, wind direction, temperature, rotor speed, pitch angle, power output, and other SCADA measurements
- **Format**: Time-series CSV with timestamp index

**Note**: The system is designed to work with local CSV files only (no internet access required).

## Repository Structure

```
project_root/
│
├── data/
│   ├── raw/                 # Original CSVs (read-only)
│   └── processed/           # Cleaned + aligned data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Data analysis and cleaning
│   ├── 02_physics_analysis.ipynb     # Physics equation analysis
│   ├── 03_baseline_models.ipynb       # Baseline model evaluation
│   ├── 04_transformer_model.ipynb     # Transformer training
│   ├── 05_probabilistic_results.ipynb # Uncertainty quantification
│   ├── 06_autoformer_model.ipynb      # Autoformer model training
│   └── 07_fedformer_model.ipynb        # FEDformer model training
│
├── src/
│   ├── data_loader.py                  # Data loading utilities
│   ├── preprocessing.py                # Data cleaning and feature engineering
│   ├── physics_constraints.py          # Physics equations and loss functions
│   ├── training.py                     # Training utilities
│   ├── evaluation.py                   # Metrics and visualization
│   └── models/
│       ├── baseline_models.py          # Persistence, LR, RF, LSTM
│       ├── transformer.py              # Time-series transformer
│       ├── autoformer.py                # Autoformer model (auto-correlation)
│       ├── fedformer.py                  # FEDformer model (frequency enhanced)
│       └── probabilistic_head.py       # Quantile regression head
│
├── results/
│   ├── figures/            # Generated plots and visualizations
│   ├── metrics/            # Evaluation metrics (CSV)
│   └── checkpoints/        # Saved model weights
│
├── README.md
└── requirements.txt
```

## Physics Integration

### Wind Power Equation

The theoretical power output follows the fundamental wind power equation:

**P = 0.5 × ρ × A × Cp × v³**

Where:
- **ρ (rho)** = air density (kg/m³), typically ~1.225 at sea level
- **A** = rotor swept area (m²), π × (rotor_radius)²
- **Cp** = power coefficient (dimensionless), typically 0.3-0.5 (max ~0.59, Betz limit)
- **v** = wind speed (m/s)

### Power Curve Regions

Wind turbines operate in distinct regions:
1. **Below cut-in speed** (< 3 m/s): No power generation
2. **Operational region** (3-12 m/s): Cubic relationship with wind speed
3. **Rated power region** (12-25 m/s): Constant rated power output
4. **Above cut-out speed** (> 25 m/s): Turbine shutdown for safety

### Physics-Aware Loss Function

The model uses a composite loss that penalizes physical violations:

```
L_total = L_data + λ_physics × L_physics + λ_negative × L_negative + λ_monotonic × L_monotonic
```

Where:
- **L_data**: Standard MSE loss
- **L_physics**: Physics residual loss (difference from theoretical power)
- **L_negative**: Penalty for negative power predictions
- **L_monotonic**: Penalty for violations of monotonicity w.r.t. wind speed

## Model Architecture

### Model Architectures

#### 1. Time-Series Transformer

The core transformer architecture adapted for time-series:

```
Input (batch, seq_len, features)
    ↓
Patch Embedding (divides sequence into patches)
    ↓
Positional Encoding
    ↓
Transformer Encoder Layers (multi-head attention)
    ↓
Global Average Pooling
    ↓
Output Projection
    ↓
Predictions (batch, output_size)
```

**Key Components**:
- **Patch-based encoding**: Divides time series into patches for efficient processing
- **Multi-head self-attention**: Captures long-range dependencies
- **Positional encoding**: Preserves temporal order information

#### 2. Autoformer Model

Transformer with auto-correlation mechanism:
- **Auto-Correlation Mechanism**: Discovers period-based dependencies using FFT
- **Series Decomposition**: Separates trend and seasonal components
- **Efficient for long-term forecasting**: Better handles periodic patterns

#### 3. FEDformer Model

Frequency Enhanced Decomposed Transformer:
- **Frequency Domain Enhancement**: Uses FFT to capture periodic patterns efficiently
- **Series Decomposition**: Separates trend and seasonal components
- **Mixer Layers**: Efficient frequency domain mixing
- **Long-term forecasting**: Optimized for capturing long-range dependencies

### Probabilistic Head

For uncertainty quantification, the model uses **quantile regression**:
- Predicts multiple quantiles (e.g., 0.1, 0.5, 0.9)
- Provides uncertainty bounds (80% confidence interval from 0.1 to 0.9 quantiles)
- Uses pinball loss for training

Alternative: Gaussian NLL head for mean-variance predictions.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

### Setup

```bash
# Clone or navigate to project directory
cd "path/to/project"

# Install dependencies
pip install -r requirements.txt

# Ensure data is in data/raw/ directory
# The CSV file should be: data/raw/Aventa_AV7_IET_OST_SCADA.csv
```

## Usage

### Step-by-Step Execution

Execute notebooks in order:

1. **Data Exploration** (`01_data_exploration.ipynb`)
   - Loads and analyzes raw SCADA data
   - Identifies features and handles missing values
   - Visualizes power curves and time series
   - Saves cleaned data to `data/processed/`

2. **Physics Analysis** (`02_physics_analysis.ipynb`)
   - Computes theoretical power using physics equation
   - Analyzes physics residual (actual - theoretical)
   - Visualizes where physics fails (turbulence, saturation)

3. **Baseline Models** (`03_baseline_models.ipynb`)
   - Trains and evaluates: Persistence, Linear Regression, Random Forest, LSTM
   - Saves baseline metrics for comparison

4. **Transformer Training** (`04_transformer_model.ipynb`)
   - Trains physics-aware transformer
   - Evaluates on test set
   - Saves model checkpoints

5. **Probabilistic Results** (`05_probabilistic_results.ipynb`)
   - Trains probabilistic model with quantile regression
   - Evaluates uncertainty quantification
   - Generates visualizations with uncertainty bounds

6. **Autoformer Model** (`06_autoformer_model.ipynb`)
   - Trains Autoformer with auto-correlation mechanism
   - Handles periodic patterns effectively
   - Series decomposition for trend/seasonal separation

7. **FEDformer Model** (`07_fedformer_model.ipynb`)
   - Trains FEDformer with frequency domain enhancement
   - Uses FFT for efficient periodic pattern capture
   - Series decomposition and mixer layers

### Quick Start Example

```python
from src.data_loader import load_scada_data
from src.preprocessing import clean_scada_data, time_aware_split
from src.models.transformer import TimeSeriesTransformer
from src.physics_constraints import PhysicsLoss

# Load data
df = load_scada_data('data/raw/Aventa_AV7_IET_OST_SCADA.csv')

# Clean and prepare
df_clean = clean_scada_data(df, target_col='power', feature_cols=['wind_speed', ...])
train_df, val_df, test_df = time_aware_split(df_clean)

# Create model
model = TimeSeriesTransformer(input_size=n_features, d_model=128, ...)

# Train with physics loss
physics_loss = PhysicsLoss(lambda_physics=0.1, lambda_negative=1.0, lambda_monotonic=0.5)
# ... training code ...
```

## Evaluation Protocol

### Metrics

**Standard Regression Metrics**:
- **MAE**: Mean Absolute Error (kW)
- **RMSE**: Root Mean Squared Error (kW)
- **MAPE**: Mean Absolute Percentage Error (%)
- **R²**: Coefficient of determination

**Probabilistic Metrics**:
- **Pinball Loss**: Quantile regression loss
- **Coverage Probability**: Fraction of true values within uncertainty bounds
- **Prediction Interval Width**: Average width of uncertainty intervals

### Time-Aware Splitting

**Critical**: All splits are temporal (no data leakage):
- **Train**: 70% (earliest data)
- **Validation**: 15% (middle)
- **Test**: 15% (latest data)

This ensures realistic evaluation on future predictions.

## Results

Results are saved in `results/`:
- **Metrics**: CSV files with evaluation metrics
- **Figures**: Visualizations (power curves, predictions, uncertainty bounds)
- **Checkpoints**: Saved model weights

### Expected Improvements

The physics-aware transformer should show:
1. **Better accuracy** than baselines (especially LSTM)
2. **No negative predictions** (physics constraint)
3. **Monotonic behavior** w.r.t. wind speed
4. **Calibrated uncertainty** (coverage probability ≈ 0.8 for 80% intervals)

## How This Improves Over Existing Work

1. **Physics Integration**: Unlike pure data-driven models, incorporates domain knowledge
2. **Uncertainty Quantification**: Provides uncertainty bounds, not just point estimates
3. **Transformer Architecture**: Captures long-range dependencies better than RNNs
4. **Reproducibility**: Fully documented, no hard-coded paths, reproducible seeds

## Reproducibility

- **Random seeds**: Set to 42 for reproducibility
- **No hard-coded paths**: All paths use `Path` objects relative to project root
- **Version control**: `requirements.txt` pins dependency versions
- **Documentation**: Every function is documented with docstrings

## Future Work

- [ ] Multi-turbine SCADA datasets for robustness testing
- [ ] Ensemble methods combining multiple models
- [ ] Online learning for adapting to changing conditions
- [ ] Integration with weather forecast data
- [ ] Real-time deployment pipeline

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wind_power_prediction_2024,
  title={Physics-Aware Probabilistic Wind Power Output Prediction using Time-Series Transformers},
  author={Your Name},
  year={2024},
  note={Semester Research Project}
}
```

## License

This project is for research purposes. Please ensure you have appropriate permissions for the SCADA dataset.

## Contact

For questions or issues, please open an issue in the repository or contact the project maintainer.

---

**Note**: This is a research artifact designed for academic use. All code follows best practices for reproducibility and documentation.

