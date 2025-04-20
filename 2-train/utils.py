import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

def create_run_directory():
    """Create a new run directory with timestamp and required subdirectories."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', timestamp)
    
    # Create directory structure
    subdirs = ['plots', 'metrics', 'config', 'checkpoints', 'predictions']
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir

def save_metrics(metrics, run_dir, fold=None):
    """Save metrics to CSV file."""
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
        'train_test_split': 0.8,
        'n_folds': 5,
        'early_stopping_rounds': 50
    }
    
    with open(os.path.join(config_dir, 'run_info.txt'), 'w') as f:
        json.dump(run_info, f, indent=4)

def save_predictions(predictions, run_dir, fold=None, is_test=False):
    """Save predictions to CSV file."""
    predictions_dir = os.path.join(run_dir, 'predictions')
    if is_test:
        filename = 'test_predictions.csv'
    elif fold is not None:
        filename = f'fold_{fold}_predictions.csv'
    else:
        filename = 'predictions.csv'
    
    predictions.to_csv(os.path.join(predictions_dir, filename), index=False) 