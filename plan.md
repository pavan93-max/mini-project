You are an expert ML researcher working on a semester-long research project.

PROJECT TITLE:
Physics-Aware Probabilistic Wind Power Output Prediction using Time-Series Transformers

PROJECT GOAL:
Build a state-of-the-art wind turbine power output prediction system that:
1) Achieves competitive or better accuracy than existing baselines
2) Incorporates physical constraints from wind turbine power equations
3) Produces probabilistic (uncertainty-aware) predictions, not just point estimates
4) Is reproducible, well-documented, and research-paper ready

DATASET:
We are using SCADA wind turbine datasets downloaded locally as CSV files:
- Aventa_AV7_IET_OST_SCADA.csv
- (Optional later) multi-turbine SCADA datasets for robustness testing

You must NOT assume any internet access. Work only with local CSVs.

--------------------------------------------------
REPOSITORY STRUCTURE (CREATE IF NOT EXISTS)
--------------------------------------------------

project_root/
│
├── data/
│   ├── raw/                 # original CSVs (read-only)
│   ├── processed/           # cleaned + aligned data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_physics_analysis.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_transformer_model.ipynb
│   ├── 05_probabilistic_results.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── physics_constraints.py
│   ├── models/
│   │   ├── baseline_models.py
│   │   ├── transformer.py
│   │   ├── probabilistic_head.py
│   │
│   ├── training.py
│   ├── evaluation.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│
├── README.md
├── requirements.txt

--------------------------------------------------
STEP 1: DATA EXPLORATION NOTEBOOK
--------------------------------------------------

Create `01_data_exploration.ipynb` that:
- Loads the CSV safely
- Parses timestamps correctly
- Shows:
  - Missing values
  - Sampling frequency
  - Feature distributions
  - Correlation with power output
- Visualizes:
  - Wind speed vs power curve
  - Time series of power, wind speed, direction
- Clearly identifies usable features:
  wind speed, wind direction, temperature, rotor speed, pitch angle, power output

Save cleaned intermediate CSV to data/processed/

--------------------------------------------------
STEP 2: PHYSICS ANALYSIS NOTEBOOK
--------------------------------------------------

Create `02_physics_analysis.ipynb` that:
- Explains the wind power equation:
  P = 0.5 * rho * A * Cp * v^3
- Shows:
  - Theoretical vs actual power
  - Cut-in, rated, cut-out behavior
- Computes a "physics residual":
  residual = actual_power - theoretical_power
- Visualizes where physics fails (turbulence, saturation)

--------------------------------------------------
STEP 3: BASELINE MODELS
--------------------------------------------------

Create `03_baseline_models.ipynb` and `baseline_models.py` with:
- Persistence model
- Linear regression
- Random Forest
- LSTM
Evaluate using:
- MAE
- RMSE
- MAPE
- R²

These baselines are REQUIRED for comparison.

--------------------------------------------------
STEP 4: PHYSICS-AWARE TRANSFORMER
--------------------------------------------------

Create `transformer.py` implementing:
- Time-series Transformer with:
  - Patch-based temporal encoding
  - Multi-head attention
- Input: past T timesteps of SCADA features
- Output: predicted power OR physics correction term

Integrate physics constraints by:
- Penalizing negative power
- Penalizing violation of monotonicity w.r.t wind speed
- Adding physics residual loss:
  L_total = L_data + λ * L_physics

--------------------------------------------------
STEP 5: PROBABILISTIC OUTPUT
--------------------------------------------------

Create `probabilistic_head.py` implementing:
- Quantile regression (q = 0.1, 0.5, 0.9)
OR
- Mean + variance prediction (Gaussian NLL)

Model must output uncertainty bounds.

--------------------------------------------------
STEP 6: TRAINING PIPELINE
--------------------------------------------------

Create `training.py`:
- Sliding window generation
- Train/validation/test split (time-aware)
- Early stopping
- Reproducible seeds
- GPU support if available

--------------------------------------------------
STEP 7: EVALUATION
--------------------------------------------------

Create `evaluation.py` and `05_probabilistic_results.ipynb`:
- Metrics:
  MAE, RMSE, MAPE, R²
- Probabilistic metrics:
  Pinball loss
  Coverage probability
- Visualization:
  - Prediction vs ground truth
  - Uncertainty bands
  - Error vs wind speed

--------------------------------------------------
STEP 8: DOCUMENTATION
--------------------------------------------------

Generate a detailed README.md including:
- Problem motivation
- Dataset description
- Physics integration explanation
- Model architecture diagram (ASCII ok)
- Evaluation protocol
- How this improves over existing work
- How to reproduce results

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------

- Write clean, readable, commented code
- No hard-coded paths
- No data leakage
- Time-aware splits only
- Every notebook must be executable top-to-bottom
- Treat this as a research artifact, not a toy project

BEGIN IMPLEMENTATION STEP BY STEP.
DO NOT SKIP STEPS.
