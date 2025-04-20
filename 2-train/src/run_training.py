import os
import sys
import logging
import argparse
from datetime import datetime

# Add the parent directory to the path so we can import the training module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import train_model
from src.config import XGB_PARAMS, RANDOM_SEED

# Set up logging
def setup_logging(log_dir='logs'):
    """Set up logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return log_file

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train XGBoost model for Elo Merchant Category Recommendation')
    
    # Add arguments for common hyperparameters
    parser.add_argument('--max_depth', type=int, default=XGB_PARAMS['max_depth'],
                        help='Maximum tree depth for base learners')
    parser.add_argument('--eta', type=float, default=XGB_PARAMS['eta'],
                        help='Learning rate')
    parser.add_argument('--subsample', type=float, default=XGB_PARAMS['subsample'],
                        help='Subsample ratio of the training instances')
    parser.add_argument('--colsample_bytree', type=float, default=XGB_PARAMS['colsample_bytree'],
                        help='Subsample ratio of columns when constructing each tree')
    parser.add_argument('--num_boost_round', type=int, default=1000,
                        help='Number of boosting rounds')
    parser.add_argument('--early_stopping_rounds', type=int, default=50,
                        help='Number of rounds for early stopping')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function to run the training process."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    log_file = setup_logging()
    logging.info(f"Logging to {log_file}")
    
    # Log command line arguments
    logging.info("Command line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    
    # Update XGBoost parameters with command line arguments
    XGB_PARAMS.update({
        'max_depth': args.max_depth,
        'eta': args.eta,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'random_state': args.random_seed
    })
    
    # Run training
    logging.info("Starting training process...")
    try:
        best_model, final_metrics = train_model()
        logging.info(f"Training completed successfully!")
        logging.info(f"Final metrics: {final_metrics}")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 