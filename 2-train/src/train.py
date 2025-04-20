import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Import configuration
from config import (
    DATA_DIR, TRAIN_FILE, TEST_FILE, RANDOM_SEED, N_FOLDS, 
    TRAIN_TEST_SPLIT, EARLY_STOPPING_ROUNDS, NUM_BOOST_ROUND, 
    VERBOSE_EVAL, XGB_PARAMS, EXCLUDED_FEATURES,
    SAVE_FEATURE_IMPORTANCE, SAVE_PREDICTIONS, SAVE_METRICS, 
    SAVE_MODEL_CHECKPOINTS
)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), DATA_DIR)
RUNS_DIR = os.path.join(BASE_DIR, 'runs')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# Utility functions
def create_run_directory():
    """Create a new run directory with timestamp and required subdirectories."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(RUNS_DIR, timestamp)
    
    # Create directory structure
    subdirs = ['plots', 'metrics', 'config', 'checkpoints', 'predictions']
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir

def save_metrics(metrics, run_dir, fold=None):
    """Save metrics to CSV file."""
    if not SAVE_METRICS:
        return
        
    metrics_dir = os.path.join(run_dir, 'metrics')
    if fold is not None:
        filename = f'fold_{fold}_metrics.csv'
    else:
        filename = 'final_metrics.csv'
    
    pd.DataFrame([metrics]).to_csv(os.path.join(metrics_dir, filename), index=False)

def save_plot(fig, run_dir, plot_name):
    """Save plot to PNG file."""
    plots_dir = os.path.join(run_dir, 'plots')
    fig.savefig(os.path.join(plots_dir, f'{plot_name}.png'))
    plt.close(fig)

def save_config(params, run_dir, random_seed=42):
    """Save model configuration and run information."""
    config_dir = os.path.join(run_dir, 'config')
    
    # Save parameters
    pd.DataFrame([params]).to_csv(os.path.join(config_dir, 'parameters.csv'), index=False)
    
    # Save run info
    run_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': random_seed,
        'train_test_split': TRAIN_TEST_SPLIT,
        'n_folds': N_FOLDS,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS
    }
    
    with open(os.path.join(config_dir, 'run_info.txt'), 'w') as f:
        json.dump(run_info, f, indent=4)

def save_predictions(predictions, run_dir, fold=None, is_test=False):
    """Save predictions to CSV file."""
    if not SAVE_PREDICTIONS:
        return
        
    predictions_dir = os.path.join(run_dir, 'predictions')
    if is_test:
        filename = 'test_predictions.csv'
    elif fold is not None:
        filename = f'fold_{fold}_predictions.csv'
    else:
        filename = 'predictions.csv'
    
    predictions.to_csv(os.path.join(predictions_dir, filename), index=False)

def train_model():
    """Main training function."""
    # Create run directory
    run_dir = create_run_directory()
    
    # Load data
    train_data = pd.read_parquet(os.path.join(DATA_DIR, TRAIN_FILE))
    test_data = pd.read_parquet(os.path.join(DATA_DIR, TEST_FILE))
    
    # Prepare features and target
    feature_cols = [col for col in train_data.columns if col not in EXCLUDED_FEATURES]
    X = train_data[feature_cols]
    y = train_data['target']
    
    # Save initial configuration
    save_config(XGB_PARAMS, run_dir, RANDOM_SEED)
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Initialize metrics storage
    fold_metrics = []
    best_rmse = float('inf')
    best_model = None
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f'\nTraining fold {fold}')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=VERBOSE_EVAL
        )
        
        # Make predictions
        val_pred = model.predict(dval)
        
        # Calculate metrics
        fold_metric = {
            'fold': fold,
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred),
            'best_iteration': model.best_iteration
        }
        
        # Save fold metrics
        save_metrics(fold_metric, run_dir, fold)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'card_id': train_data.iloc[val_idx]['card_id'],
            'actual': y_val,
            'predicted': val_pred
        })
        save_predictions(predictions_df, run_dir, fold)
        
        # Update best model
        if fold_metric['rmse'] < best_rmse:
            best_rmse = fold_metric['rmse']
            best_model = model
            
            # Save best model checkpoint
            if SAVE_MODEL_CHECKPOINTS:
                model.save_model(os.path.join(run_dir, 'checkpoints', 'best_model.json'))
        
        fold_metrics.append(fold_metric)
        
        # Plot feature importance
        if SAVE_FEATURE_IMPORTANCE:
            fig, ax = plt.subplots(figsize=(10, 6))
            xgb.plot_importance(model, ax=ax)
            save_plot(fig, run_dir, f'fold_{fold}_feature_importance')
    
    # Calculate and save final metrics
    final_metrics = {
        'mean_rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'std_rmse': np.std([m['rmse'] for m in fold_metrics]),
        'mean_mae': np.mean([m['mae'] for m in fold_metrics]),
        'std_mae': np.std([m['mae'] for m in fold_metrics])
    }
    
    save_metrics(final_metrics, run_dir)
    
    # Save best model to saved_models directory
    if best_model is not None:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        best_model.save_model(os.path.join(SAVED_MODELS_DIR, 'xgb_model_final.json'))
    
    # Make predictions on test data
    test_features = test_data[feature_cols]
    dtest = xgb.DMatrix(test_features)
    test_predictions = best_model.predict(dtest)
    
    # Save test predictions
    predictions_df = pd.DataFrame({
        'card_id': test_data['card_id'],
        'target': test_predictions
    })
    save_predictions(predictions_df, run_dir, is_test=True)
    
    return best_model, final_metrics

if __name__ == "__main__":
    best_model, final_metrics = train_model()
    print("\nTraining completed!")
    print(f"Final metrics: {final_metrics}") 