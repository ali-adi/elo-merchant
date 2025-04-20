"""
Configuration file for the Elo Merchant Category Recommendation model.
All hyperparameters and settings are defined here for easy modification.
"""

# Data paths
DATA_DIR = '../../datasets'
TRAIN_FILE = 'final_dataset.parquet'
TEST_FILE = 'test.parquet'

# Training settings
RANDOM_SEED = 42
N_FOLDS = 5
TRAIN_TEST_SPLIT = 0.8
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 1000
VERBOSE_EVAL = 100

# XGBoost parameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'eta': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': RANDOM_SEED
}

# Feature settings
# 'card_id' is excluded as it's an identifier, not a feature
# 'target' is the dependent variable we're trying to predict
EXCLUDED_FEATURES = ['card_id', 'target']

# Output settings
SAVE_FEATURE_IMPORTANCE = True
SAVE_PREDICTIONS = True
SAVE_METRICS = True
SAVE_MODEL_CHECKPOINTS = True 