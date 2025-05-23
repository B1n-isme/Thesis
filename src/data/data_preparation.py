"""
Data preparation module for neural forecasting pipeline.
"""
import pandas as pd
from src.config.forecasting_config import (
    DATA_PATH, DATE_COLUMN, TARGET_COLUMN, TARGET_RENAMED, 
    DATE_RENAMED, UNIQUE_ID_VALUE, FORECAST_HORIZON, TEST_LENGTH_MULTIPLIER
)
from src.utils.forecasting_utils import get_historical_exogenous_features, print_data_info


def load_and_prepare_data():
    """Load and prepare the dataset for forecasting."""
    print("Loading and preparing data...")
    
    # Load data
    df = pd.read_parquet(DATA_PATH)
    
    # Rename columns
    df = df.rename(columns={DATE_COLUMN: DATE_RENAMED, TARGET_COLUMN: TARGET_RENAMED})
    
    # Add unique_id and convert date
    df['unique_id'] = UNIQUE_ID_VALUE
    df[DATE_RENAMED] = pd.to_datetime(df[DATE_RENAMED])
    df.reset_index(drop=True, inplace=True)
    
    return df


def split_data(df, horizon, test_length_multiplier):
    """Split data into development and final holdout test sets."""
    
    test_length = horizon * test_length_multiplier
    
    print(f"Forecast horizon (h) set to: {horizon} days")
    
    # Validate data length
    if len(df) <= test_length:
        raise ValueError(
            "Not enough data to create a test set of the desired length. "
            "Decrease test_length or get more data."
        )
    
    # Split data
    df_development = df.iloc[:-test_length].copy()
    df_final_holdout_test = df.iloc[-test_length:].copy()
    
    # Print information
    print_data_info(df, df_development, df_final_holdout_test)
    
    return df_development, df_final_holdout_test


def prepare_forecasting_data(horizon=None, test_length_multiplier=None):
    """Complete data preparation pipeline."""
    if horizon is None:
        horizon = FORECAST_HORIZON
    if test_length_multiplier is None:
        test_length_multiplier = TEST_LENGTH_MULTIPLIER

    # Load and prepare data
    df = load_and_prepare_data()
    
    # Get historical exogenous features
    hist_exog_list = get_historical_exogenous_features(df)
    
    # Split data
    df_development, df_final_holdout_test = split_data(df, horizon, test_length_multiplier)
    
    return df, df_development, df_final_holdout_test, hist_exog_list


if __name__ == "__main__":
    df, df_dev, df_test, hist_exog = prepare_forecasting_data()
    print(f"\nHistorical exogenous features: {len(hist_exog)} features")
    print(f"Sample features: {hist_exog[:5] if len(hist_exog) >= 5 else hist_exog}") 