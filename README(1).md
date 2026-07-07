# Multi-System Chaotic Dynamics Dataset Example

This folder contains a baseline workflow for the Kaggle dataset:

`namandixit07/multi-system-chaotic-dynamics-dataset`

The script does four things:

1. Downloads the dataset with `kagglehub`, or loads a local dataset folder.
2. Finds CSV, TSV, Parquet, NumPy, NPZ, or MAT files.
3. Builds sliding-window sequences from chaotic state variables.
4. Trains a GRU model to predict the next state.

## Install

```bash
cd /workspace/chaotic_dynamics_kaggle
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd D:\path\to\chaotic_dynamics_kaggle
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run With Kaggle Download

```bash
python chaotic_forecast.py
```

If Kaggle asks for authentication, create an API token from your Kaggle account settings, then set it as KaggleHub expects.

## Run With A Manually Downloaded Folder

```bash
python chaotic_forecast.py --data-dir ./multi-system-chaotic-dynamics-dataset
```

## Recommended Explicit Run

If the dataset has columns like `system`, `t`, `x`, `y`, `z`, run:

```bash
python chaotic_forecast.py --system-col system --time-col t --state-cols x,y,z
```

If the column names are different, check the printed column summary and replace those names.

## Useful Options

```bash
python chaotic_forecast.py --seq-len 128 --horizon 5 --epochs 50
python chaotic_forecast.py --no-train
python chaotic_forecast.py --max-rows 100000 --max-windows 50000
```

## Outputs

Outputs are written to `outputs/`:

- `eda_summary.csv`: basic numeric statistics
- `trajectory_preview.png`: 2D/3D trajectory preview
- `training_loss.png`: GRU training loss
- `forecast_preview.png`: true vs predicted preview
- `predictions_head.csv`: first prediction rows
- `metrics.json`: RMSE scores
- `gru_forecaster.pt`: trained PyTorch checkpoint
