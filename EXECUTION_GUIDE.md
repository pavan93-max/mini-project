# Execution Guide - Wind Power Prediction Project

This guide provides step-by-step commands to run the entire project from data processing to final model evaluation.

## Prerequisites

Ensure you have Python 3.8+ installed and the project dependencies.

## Step 0: Setup Environment

```bash
# Navigate to project directory
cd "D:\mini project"

# Create virtual environment (if not already created)
python -m venv mini

# Activate virtual environment
# On Windows:
mini\Scripts\activate
# On Linux/Mac:
# source mini/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Data Exploration and Cleaning

**Notebook:** `01_data_exploration.ipynb`

**Purpose:** Load raw SCADA data, explore it, clean it, and save processed data.

**Command:**
```bash
# Using Jupyter
jupyter notebook notebooks/01_data_exploration.ipynb

# Or using JupyterLab
jupyter lab notebooks/01_data_exploration.ipynb

# Or using VS Code
code notebooks/01_data_exploration.ipynb
```

**What it does:**
- Loads raw CSV from `data/raw/Aventa_AV7_IET_OST_SCADA.csv`
- Analyzes missing values, distributions, correlations
- Cleans data (handles outliers, missing values)
- Saves cleaned data to `data/processed/scada_cleaned.csv` (or `.csv.gz` if large)
- Saves feature mapping to `data/processed/feature_mapping.json`

**Expected Output:**
- `data/processed/scada_cleaned.csv` or `scada_cleaned.csv.gz`
- `data/processed/feature_mapping.json`
- `results/figures/` with visualization plots

---

## Step 2: Physics Analysis

**Notebook:** `02_physics_analysis.ipynb`

**Purpose:** Analyze physics constraints and residuals.

**Command:**
```bash
jupyter notebook notebooks/02_physics_analysis.ipynb
```

**What it does:**
- Loads cleaned data
- Computes theoretical power using physics equation
- Analyzes physics residual (actual - theoretical)
- Visualizes where physics fails (turbulence, saturation)

**Expected Output:**
- `results/figures/` with physics analysis plots

---

## Step 3: Baseline Models

**Notebook:** `03_baseline_models.ipynb`

**Purpose:** Train and evaluate baseline models for comparison.

**Command:**
```bash
jupyter notebook notebooks/03_baseline_models.ipynb
```

**What it does:**
- Trains: Persistence, Linear Regression, Random Forest, LSTM
- Evaluates using: MAE, RMSE, MAPE, R²
- Saves baseline metrics for comparison

**Expected Output:**
- Baseline model metrics
- `results/metrics/` with baseline predictions
- `results/figures/` with baseline model plots

---

## Step 4: Transformer Model

**Notebook:** `04_transformer_model.ipynb`

**Purpose:** Train the physics-aware transformer model.

**Command:**
```bash
jupyter notebook notebooks/04_transformer_model.ipynb
```

**What it does:**
- Prepares sequences for transformer
- Creates and trains TimeSeriesTransformer model
- Applies physics-aware loss (optional)
- Saves model checkpoints
- Evaluates on test set

**Expected Output:**
- `results/checkpoints/best_model.pt` (transformer model)
- `results/metrics/transformer_predictions.csv`
- `results/figures/transformer_training_history.png`
- `results/figures/transformer_predictions.png`

---

## Step 5: Probabilistic Results

**Notebook:** `05_probabilistic_results.ipynb`

**Purpose:** Train probabilistic model with uncertainty quantification.

**Command:**
```bash
jupyter notebook notebooks/05_probabilistic_results.ipynb
```

**What it does:**
- Trains transformer with quantile regression head
- Predicts multiple quantiles (e.g., 0.1, 0.5, 0.9)
- Evaluates uncertainty quantification
- Generates visualizations with uncertainty bounds

**Expected Output:**
- `results/checkpoints/` with probabilistic model
- `results/metrics/` with quantile predictions
- `results/figures/` with uncertainty plots

---

## Step 6: Autoformer Model

**Notebook:** `06_autoformer_model.ipynb`

**Purpose:** Train Autoformer model with auto-correlation mechanism.

**Command:**
```bash
jupyter notebook notebooks/06_autoformer_model.ipynb
```

**What it does:**
- Prepares sequences for Autoformer
- Creates and trains AutoformerModel
- Uses auto-correlation mechanism for periodic patterns
- Series decomposition (trend + seasonal)
- Evaluates on test set

**Expected Output:**
- `results/checkpoints/best_model.pt` (autoformer model)
- `results/metrics/autoformer_predictions.csv`
- `results/figures/autoformer_training_history.png`
- `results/figures/autoformer_predictions.png`

---

## Step 7: FEDformer Model

**Notebook:** `07_fedformer_model.ipynb`

**Purpose:** Train FEDformer model with frequency domain enhancement.

**Command:**
```bash
jupyter notebook notebooks/07_fedformer_model.ipynb
```

**What it does:**
- Prepares sequences for FEDformer
- Creates and trains FEDformerModel
- Uses frequency domain enhancement (FFT)
- Series decomposition and mixer layers
- Evaluates on test set

**Expected Output:**
- `results/checkpoints/best_model.pt` (fedformer model)
- `results/metrics/fedformer_predictions.csv`
- `results/figures/fedformer_training_history.png`
- `results/figures/fedformer_predictions.png`

---

## Quick Run All (Alternative: Using nbconvert)

If you want to run all notebooks from command line without opening Jupyter:

```bash
# Activate environment first
mini\Scripts\activate

# Run all notebooks in sequence
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --output 01_data_exploration_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_physics_analysis.ipynb --output 02_physics_analysis_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_baseline_models.ipynb --output 03_baseline_models_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_transformer_model.ipynb --output 04_transformer_model_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_probabilistic_results.ipynb --output 05_probabilistic_results_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_autoformer_model.ipynb --output 06_autoformer_model_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/07_fedformer_model.ipynb --output 07_fedformer_model_executed.ipynb
```

**Note:** This will execute all cells and may take a long time. Use with caution.

---

## Running Individual Cells (Jupyter)

When working in Jupyter, you can run cells individually:

1. **Run current cell:** `Shift + Enter`
2. **Run current cell and insert below:** `Alt + Enter`
3. **Run all cells:** `Cell > Run All`
4. **Run all cells above:** `Cell > Run All Above`
5. **Restart kernel:** `Kernel > Restart`

---

## Troubleshooting

### Memory Issues

If you encounter memory errors:

1. **Reduce sequence limits in notebooks:**
   - In sequence preparation cells, reduce `max_train_samples`, `max_val_samples`, `max_test_samples`
   - Example: Change `500_000` to `100_000`

2. **Use data sampling:**
   - In `01_data_exploration.ipynb`, use `sample_ratio=0.1` when loading data
   - Example: `load_scada_data(csv_path, sample_ratio=0.1)`

3. **Use smaller batch sizes:**
   - In training cells, reduce `batch_size` from 32 to 16 or 8

### Module Not Found Errors

If you get import errors:

```bash
# Restart Jupyter kernel
# In notebook: Kernel > Restart

# Or reload modules in notebook:
import importlib
import preprocessing
importlib.reload(preprocessing)
```

### File Not Found Errors

Ensure you run notebooks in order:
1. `01_data_exploration.ipynb` must run first (creates cleaned data)
2. All subsequent notebooks depend on cleaned data

---

## Expected Directory Structure After Running

```
D:\mini project\
├── data/
│   ├── raw/
│   │   └── Aventa_AV7_IET_OST_SCADA.csv (your input file)
│   └── processed/
│       ├── scada_cleaned.csv or scada_cleaned.csv.gz
│       └── feature_mapping.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_physics_analysis.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_transformer_model.ipynb
│   ├── 05_probabilistic_results.ipynb
│   ├── 06_autoformer_model.ipynb
│   └── 07_fedformer_model.ipynb
├── results/
│   ├── checkpoints/
│   │   └── best_model.pt (from each model)
│   ├── figures/
│   │   ├── missing_values.png
│   │   ├── power_curve.png
│   │   ├── transformer_training_history.png
│   │   ├── autoformer_training_history.png
│   │   ├── fedformer_training_history.png
│   │   └── ... (other plots)
│   └── metrics/
│       ├── baseline_predictions.csv
│       ├── transformer_predictions.csv
│       ├── autoformer_predictions.csv
│       └── fedformer_predictions.csv
└── src/
    └── models/
        └── ... (model implementations)
```

---

## Summary: Complete Execution Flow

```bash
# 1. Setup
cd "D:\mini project"
mini\Scripts\activate
pip install -r requirements.txt

# 2. Run notebooks in order (open each in Jupyter)
jupyter notebook notebooks/01_data_exploration.ipynb      # First: Clean data
jupyter notebook notebooks/02_physics_analysis.ipynb     # Physics analysis
jupyter notebook notebooks/03_baseline_models.ipynb      # Baselines
jupyter notebook notebooks/04_transformer_model.ipynb    # Transformer
jupyter notebook notebooks/05_probabilistic_results.ipynb  # Probabilistic
jupyter notebook notebooks/06_autoformer_model.ipynb     # Autoformer
jupyter notebook notebooks/07_fedformer_model.ipynb      # FEDformer
```

**Important:** Always run notebooks in numerical order (01 → 02 → 03 → ...) as each depends on outputs from previous notebooks.

---

## Notes

- **Execution Time:** Each notebook may take 10-60+ minutes depending on dataset size and hardware
- **GPU Recommended:** For transformer models (04, 05, 06, 07), GPU will significantly speed up training
- **Memory Requirements:** Large datasets may require 8GB+ RAM. Use sampling if needed
- **Checkpoints:** Model checkpoints are saved automatically. You can resume training if interrupted

---

## Next Steps After Running

1. **Compare Models:** Review metrics in `results/metrics/` to compare model performance
2. **Visualize Results:** Check plots in `results/figures/` for visual comparisons
3. **Best Model:** Identify best performing model from saved checkpoints
4. **Deploy:** Use the best model checkpoint for production predictions

