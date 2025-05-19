import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from pathlib import Path

def process_time_series_data(
    input_path: str,
    output_path: str,
    target_column: str = 'btc_price',
    skew_threshold: float = 0.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Process time series data by handling skewness and calculating log returns.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed data
        target_column: Name of target column to calculate returns for
        skew_threshold: Threshold for skewness transformation
        verbose: Whether to print processing details
    
    Returns:
        Processed DataFrame
    """
    # Load and prepare data
    df = pd.read_csv(input_path, parse_dates=['date'], index_col='date')
    df['Oil_Crude_Price'] = df['Oil_Crude_Price'].clip(lower=0)
    
    # Split features and target
    X = df.drop(columns=[target_column], axis=1)
    y_series = np.log(df[target_column]).diff().dropna()
    y_series.name = target_column
    
    # Process features
    X_processed = _handle_skewness(X, skew_threshold, verbose)
    
    # Combine and save
    processed_df = pd.concat([X_processed, y_series], axis=1).dropna()
    processed_df.to_csv(output_path, index=True)
    
    return processed_df

def _handle_skewness(
    df: pd.DataFrame,
    threshold: float,
    verbose: bool
) -> pd.DataFrame:
    """Handle skewness in DataFrame columns using appropriate transformations."""
    df_copy = df.copy()
    
    for col in df_copy.columns:
        skewness = df_copy[col].skew()
        if verbose:
            print(f"Column: '{col}' - Original Skewness: {skewness:.4f}")
            
        if abs(skewness) <= threshold:
            continue
            
        if skewness > threshold:
            method = 'yeo-johnson' if (df_copy[col] < 0).any() else 'box-cox'
            if verbose:
                print(f"  Action: Applying {method} transformation due to positive skew > {threshold}")
        else:
            method = 'yeo-johnson'
            if verbose:
                print(f"  Action: Applying {method} transformation due to negative skew < -{threshold}")
                
        pt = PowerTransformer(method=method, standardize=False)
        df_copy[col] = pt.fit_transform(df_copy[col].values.reshape(-1, 1))
        
        if verbose:
            print(f"  New Skewness for '{col}': {df_copy[col].skew():.4f}\n")
            
    return df_copy

def inverse_log_return(y_log_return: pd.Series, initial_price: float) -> pd.Series:
    """
    Inverse transform log returns to recover the original price series.
    Args:
        y_log_return: Series of log returns (as produced by np.log(price).diff())
        initial_price: The price at the time step immediately before the first log return
    Returns:
        Series of reconstructed prices (same index as y_log_return, first value is initial_price)
    """
    # Reconstruct log(price) by cumulative sum, then exponentiate
    log_price = np.log(initial_price) + y_log_return.cumsum()
    price = np.exp(log_price)
    # Optionally, prepend the initial price for completeness
    price_full = pd.concat([
        pd.Series([initial_price], index=[y_log_return.index[0] - pd.Timedelta(1, unit='D')]),
        price
    ])
    return price_full

if __name__ == "__main__":
    input_path = 'data/final/final_dataset.csv'
    output_path = 'data/final/final_dataset_processed.csv'
    process_time_series_data(input_path, output_path)