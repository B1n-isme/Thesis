"""
Configuration settings for neural forecasting pipeline.
"""
import os
import torch
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MQLoss, DistributionLoss

# Pipeline Configuration
FORECAST_HORIZON = 7
LEVELS = [80, 90]
TEST_LENGTH_MULTIPLIER = 2  # test_length = horizon * 2
SEED = 42

# Data Configuration
DATA_PATH = 'data/final/dataset.parquet'
DATE_COLUMN = 'Date'
TARGET_COLUMN = 'btc_close'
TARGET_RENAMED = 'y'
DATE_RENAMED = 'ds'
UNIQUE_ID_VALUE = 'Bitcoin'

# Model Configuration
FREQUENCY = 'D'
LOCAL_SCALER_TYPE = 'standard'

# Cross-validation Configuration
CV_N_WINDOWS = 5
CV_STEP_SIZE = FORECAST_HORIZON

# Hyperparameter Tuning Configuration
NUM_SAMPLES_PER_MODEL = 1
TUNING_RESULTS_DIR = 'tuning_results'
BEST_HYPERPARAMETERS_CSV = 'tuning_results/best_hyperparameters.csv'

# Ray Configuration
RAY_ADDRESS = 'local'
RAY_NUM_CPUS = os.cpu_count()
RAY_NUM_GPUS = torch.cuda.device_count()

# Loss function mapping
LOSS_MAP = {
    'MAE': MAE(),
    'MAE()': MAE(),
    'MQLoss': MQLoss(),
    'MQLoss()': MQLoss(),
    'RMSE': RMSE(),
    'RMSE()': RMSE(),
    'MSE': MSE(),
    'MSE()': MSE(),
}

# Columns to exclude when processing best hyperparameters
EXCLUDE_HYPERPARAMETER_KEYS = [
    'model_name', 
    'loss', 
    'valid_loss', 
    'best_valid_loss', 
    'training_iteration'
]

# JSON parseable hyperparameter keys
JSON_PARSEABLE_KEYS = [
    'n_pool_kernel_size', 
    'n_freq_downsample', 
    'n_blocks', 
    'mlp_units'
] 