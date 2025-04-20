# Elo Merchant Category Recommendation - Training Code

This directory contains the code for training the XGBoost model for the Elo Merchant Category Recommendation challenge.

## Directory Structure

```
src/
├── train.py           # Main training module with model training logic
├── config.py          # Configuration file with all hyperparameters and settings
├── run_training.py    # Script to run the training process with command-line arguments
└── README.md          # This file
```

## Requirements

- Python 3.6+
- Required packages (install via `pip install -r ../requirements.txt`):
  - pandas
  - numpy
  - xgboost
  - scikit-learn
  - matplotlib
  - seaborn

## Data

The training code expects the following data files in the `../datasets/` directory:
- `final_dataset.parquet`: Processed training data
- `test.parquet`: Processed test data

## Configuration

All hyperparameters and settings are defined in `config.py`. You can modify this file to change:
- Data paths
- Training settings (random seed, number of folds, etc.)
- XGBoost parameters
- Feature settings
- Output settings

## Running the Training

### Basic Usage

To train the model with default parameters:

```bash
python src/run_training.py
```

### Customizing Hyperparameters

You can customize hyperparameters using command-line arguments:

```bash
python src/run_training.py --max_depth 8 --eta 0.005 --subsample 0.9
```

Available arguments:
- `--max_depth`: Maximum tree depth for base learners
- `--eta`: Learning rate
- `--subsample`: Subsample ratio of the training instances
- `--colsample_bytree`: Subsample ratio of columns when constructing each tree
- `--num_boost_round`: Number of boosting rounds
- `--early_stopping_rounds`: Number of rounds for early stopping
- `--n_folds`: Number of folds for cross-validation
- `--random_seed`: Random seed for reproducibility

## Output

Each training run creates a directory in `../runs/` with the following structure:

```
runs/YYYYMMDD_HHMMSS/
├── plots/                # Training plots
│   ├── fold_*_feature_importance.png
│   └── ...
├── metrics/              # Metrics for each fold and final results
│   ├── fold_*_metrics.csv
│   └── final_metrics.csv
├── config/               # Configuration files
│   ├── parameters.csv
│   └── run_info.txt
├── checkpoints/          # Model checkpoints
│   └── best_model.json
└── predictions/          # Predictions
    ├── fold_*_predictions.csv
    └── test_predictions.csv
```

The best model is saved to `../saved_models/xgb_model_final.json`.

## Logs

Training logs are saved in the `logs/` directory with timestamps. 