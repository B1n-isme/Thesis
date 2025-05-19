"""
Time series dataset splitting techniques for machine learning pipelines.

This module provides a collection of functions for splitting time series datasets
into train/test or train/validation/test sets using various strategies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional
from sklearn.model_selection import TimeSeriesSplit
from sktime.split import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
    temporal_train_test_split
)


def train_val_test_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    test_size: Union[float, int] = 0.2,
    val_size: Union[float, int] = 0.2,
    gap: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into train, validation, and test sets respecting temporal order.
    
    Args:
        X: Feature dataset, a pandas DataFrame or numpy array
        y: Target variable, a pandas Series or numpy array (optional)
        test_size: Size of test set as a fraction of the entire dataset
        val_size: Size of validation set as a fraction of the entire dataset
        gap: Number of samples to exclude between splits
    
    Returns:
        Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Calculate indices for the splits
    n_samples = len(X)
    test_end = n_samples
    test_start = n_samples - int(n_samples * test_size)
    val_end = test_start - gap
    val_start = val_end - int(n_samples * val_size)
    train_end = val_start - gap
    
    # Create the splits
    X_train = X[:train_end]
    X_val = X[val_start:val_end]
    X_test = X[test_start:test_end]
    
    if y is not None:
        y_train = y[:train_end]
        y_val = y[val_start:val_end]
        y_test = y[test_start:test_end]
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    return X_train, X_val, X_test, None, None, None


def purged_kfold_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Implements the Purged K-Fold Cross-Validation technique for time series data.
    
    This method prevents data leakage by "purging" training data that overlaps with test data
    in time and implements an "embargo" period to avoid sample overlap.
    
    Args:
        X: Feature dataset, a pandas DataFrame or numpy array
        y: Target variable, a pandas Series or numpy array (optional)
        n_splits: Number of folds
        embargo_pct: Percentage of train data to embargo after test set
        
    Returns:
        List of tuples containing train and test indices for each fold
    """
    # Implement basic purged k-fold logic
    indices = np.arange(len(X))
    n_samples = len(indices)
    fold_size = n_samples // n_splits
    embargo_size = int(fold_size * embargo_pct)
    
    splits = []
    for i in range(n_splits):
        # Calculate test indices for this fold
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n_samples)
        test_indices = indices[test_start:test_end]
        
        # Calculate train indices with embargo
        if i == 0:
            # First fold has no embargo before
            embargo_end = test_end + embargo_size
            train_indices = indices[test_end:] if embargo_size == 0 else indices[embargo_end:]
        elif i == n_splits - 1:
            # Last fold has no embargo after
            embargo_start = test_start - embargo_size
            train_indices = indices[:embargo_start]
        else:
            # Middle folds have embargo on both sides
            embargo_start = test_start - embargo_size
            embargo_end = test_end + embargo_size
            train_indices = np.concatenate([indices[:embargo_start], indices[embargo_end:]])
        
        splits.append((train_indices, test_indices))
    
    return splits


def blocked_timeseries_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split time series data into non-overlapping training and validation sets.
    
    This approach divides the data into n_splits blocks and uses one block for
    testing and all preceding blocks for training.
    
    Args:
        X: Feature dataset, a pandas DataFrame or numpy array
        y: Target variable, a pandas Series or numpy array (optional)
        n_splits: Number of splits to generate
    
    Returns:
        List of tuples containing train and test indices for each fold
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    # Calculate block size
    k = int(np.ceil(n_samples / n_splits))
    
    splits = []
    for i in range(n_splits - 1):
        start_idx = i * k
        end_idx = min((i + 1) * k, n_samples)
        
        # Test indices for this fold
        test_indices = indices[start_idx:end_idx]
        
        # Train indices are all indices before start_idx
        if start_idx > 0:
            train_indices = indices[:start_idx]
            splits.append((train_indices, test_indices))
    
    # Last fold
    if (n_splits - 1) * k < n_samples:
        start_idx = (n_splits - 1) * k
        test_indices = indices[start_idx:]
        train_indices = indices[:start_idx]
        splits.append((train_indices, test_indices))
    
    return splits


def get_splitter(
    method: str,
    **kwargs
) -> Union[TimeSeriesSplit, ExpandingWindowSplitter, SingleWindowSplitter, SlidingWindowSplitter]:
    """
    Factory function to create a time series splitter object.
    
    Args:
        method: Splitting method - one of "sklearn", "expanding", "single", "sliding"
        **kwargs: Parameters specific to the chosen splitter
        
    Returns:
        A splitter object that can be used in cross-validation
        
    Raises:
        ValueError: If an invalid method is specified
    """
    if method == "sklearn":
        return TimeSeriesSplit(
            n_splits=kwargs.get("n_splits", 5),
            test_size=kwargs.get("test_size", None),
            gap=kwargs.get("gap", 0)
        )
    
    elif method == "expanding":
        return ExpandingWindowSplitter(
            initial_window=kwargs.get("initial_window"),
            step_length=kwargs.get("step_length", 1),
            fh=kwargs.get("fh", 1)
        )
    
    elif method == "single":
        return SingleWindowSplitter(
            window_length=kwargs.get("window_length"),
            fh=kwargs.get("fh", 1)
        )
    
    elif method == "sliding":
        return SlidingWindowSplitter(
            window_length=kwargs.get("window_length"),
            fh=kwargs.get("fh", 1),
            step_length=kwargs.get("step_length", 1)
        )
    
    else:
        raise ValueError(f"Unknown splitting method: {method}. Choose from 'sklearn', 'expanding', 'single', 'sliding'")


# Example usage scenarios

if __name__ == "__main__":
    """
    Example usage of time series splitting techniques.
    This demonstrates how to use the functions for different scenarios.
    """
    
    
    # Create a simple time series dataset for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    X = pd.DataFrame({
        'date': dates,
        'x1': np.random.randn(1000).cumsum(),
        'x2': np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.randn(1000)*0.2
    })
    X.set_index('date', inplace=True)
    y = X['x1'] * 0.5 + X['x2'] * 0.3 + np.random.randn(1000) * 0.5
    
    # print("Example 1: Basic Train-Validation-Test Split")
    # print("---------------------------------------------")
    # # Example 1: Basic train-validation-test split
    # X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    #     X, y, test_size=0.2, val_size=0.2
    # )
    
    # print(f"Train set size: {len(X_train)}")
    # print(f"Validation set size: {len(X_val)}")
    # print(f"Test set size: {len(X_test)}")
    # print(f"Train period: {X_train.index.min()} to {X_train.index.max()}")
    # print(f"Validation period: {X_val.index.min()} to {X_val.index.max()}")
    # print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")
    
    # # Train a simple model and evaluate
    # model = LinearRegression()
    # model.fit(X_train.drop('x1', axis=1), y_train)  # Use x2 to predict
    
    # val_pred = model.predict(X_val.drop('x1', axis=1))
    # val_mse = mean_squared_error(y_val, val_pred)
    # print(f"Validation MSE: {val_mse:.4f}")
    
    # test_pred = model.predict(X_test.drop('x1', axis=1))
    # test_mse = mean_squared_error(y_test, test_pred)
    # print(f"Test MSE: {test_mse:.4f}")
    
    # print("\nExample 2: Train-Validation-Test with Gap")
    # print("---------------------------------------")
    # # Example 2: Using gaps between splits to avoid data leakage
    # X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    #     X, y, test_size=0.2, val_size=0.2, gap=10
    # )
    
    # print(f"Train set size: {len(X_train)}")
    # print(f"Validation set size: {len(X_val)}")
    # print(f"Test set size: {len(X_test)}")
    # print(f"Train period: {X_train.index.min()} to {X_train.index.max()}")
    # print(f"Validation period: {X_val.index.min()} to {X_val.index.max()}")
    # print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")
    
    print("\nExample 3: Multiple Train-Val Splits using TimeSeriesSplit")
    print("----------------------------------------------------------")
    # Example 3: Creating multiple train-val splits for cross-validation
    # This preserves a final test set and uses splits for train-val
    
    # First, reserve test set
    X_trainval, X_test, y_trainval, y_test = temporal_train_test_split(
        X, y, test_size=0.2
    )
    
    # Now create multiple train-val splits from the remaining data
    tscv = get_splitter(method="sklearn", n_splits=3, test_size=len(X_trainval)//5)
    
    for i, (train_idx, val_idx) in enumerate(tscv.split(X_trainval)):
        X_train_fold, y_train_fold = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
        X_val_fold, y_val_fold = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]
        
        print(f"Fold {i+1}:")
        print(f"  Train size: {len(X_train_fold)}, period: {X_train_fold.index.min()} to {X_train_fold.index.max()}")
        print(f"  Val size: {len(X_val_fold)}, period: {X_val_fold.index.min()} to {X_val_fold.index.max()}")
    
    print(f"Test size: {len(X_test)}, period: {X_test.index.min()} to {X_test.index.max()}")
    
    print("\nExample 4: Expanding Window Evaluation with Final Test")
    print("----------------------------------------------------")
    # Example 4: Expanding window for iterative training with final test set
    
    # Reserve test set
    X_trainval, X_test, y_trainval, y_test = temporal_train_test_split(
        X, y, test_size=0.2
    )
    
    # Create expanding window splits
    initial_window = len(X_trainval) // 3
    expander = get_splitter(
        method="expanding", 
        initial_window=initial_window,
        step_length=len(X_trainval)//10,
        fh=len(X_trainval)//5  # forecast horizon is validation size
    )
    
    for i, (train_idx, val_idx) in enumerate(expander.split(X_trainval)):
        X_train_fold, y_train_fold = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
        X_val_fold, y_val_fold = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]
        
        print(f"Iteration {i+1}:")
        print(f"  Train size: {len(X_train_fold)}, period: {X_train_fold.index.min()} to {X_train_fold.index.max()}")
        print(f"  Val size: {len(X_val_fold)}, period: {X_val_fold.index.min()} to {X_val_fold.index.max()}")
    
    print(f"Final Test size: {len(X_test)}, period: {X_test.index.min()} to {X_test.index.max()}")
    
    print("\nExample 5: Sliding Window Evaluation with Final Test")
    print("--------------------------------------------------")
    # Example 5: Sliding window for more stable training/validation with final test
    
    # Reserve test set
    X_trainval, X_test, y_trainval, y_test = temporal_train_test_split(
        X, y, test_size=0.2
    )
    
    # Create sliding window splits
    window_length = len(X_trainval) // 2
    slider = get_splitter(
        method="sliding", 
        window_length=window_length,
        step_length=len(X_trainval)//10,
        fh=len(X_trainval)//5  # forecast horizon is validation size
    )
    
    for i, (train_idx, val_idx) in enumerate(slider.split(X_trainval)):
        if i >= 3:  # Limit to first 3 windows for brevity
            break
            
        X_train_fold, y_train_fold = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
        X_val_fold, y_val_fold = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]
        
        print(f"Window {i+1}:")
        print(f"  Train size: {len(X_train_fold)}, period: {X_train_fold.index.min()} to {X_train_fold.index.max()}")
        print(f"  Val size: {len(X_val_fold)}, period: {X_val_fold.index.min()} to {X_val_fold.index.max()}")
    
    print(f"Final Test size: {len(X_test)}, period: {X_test.index.min()} to {X_test.index.max()}")
    
    print("\nExample 6: Blocked Time Series Split with Final Test")
    print("--------------------------------------------------")
    # Example 6: Using blocked time series split for train/val with a final test
    
    # Reserve test set
    X_trainval, X_test, y_trainval, y_test = temporal_train_test_split(
        X, y, test_size=0.2
    )
    
    # Get blocked splits
    blocks = blocked_timeseries_split(X_trainval, y_trainval, n_splits=4)
    
    for i, (train_idx, val_idx) in enumerate(blocks):
        X_train_fold, y_train_fold = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
        X_val_fold, y_val_fold = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]
        
        print(f"Block {i+1}:")
        print(f"  Train size: {len(X_train_fold)}, period: {X_train_fold.index.min()} to {X_train_fold.index.max()}")
        print(f"  Val size: {len(X_val_fold)}, period: {X_val_fold.index.min()} to {X_val_fold.index.max()}")
    
    print(f"Final Test size: {len(X_test)}, period: {X_test.index.min()} to {X_test.index.max()}")
    
    print("\nExample 7: Purged K-Fold with Embargo and Final Test")
    print("--------------------------------------------------")
    # Example 7: Using purged k-fold for train/val to prevent leakage, with final test
    
    # Reserve test set
    X_trainval, X_test, y_trainval, y_test = temporal_train_test_split(
        X, y, test_size=0.2
    )
    
    # Get purged splits
    purged_splits = purged_kfold_split(X_trainval, y_trainval, n_splits=3, embargo_pct=0.05)
    
    for i, (train_idx, val_idx) in enumerate(purged_splits):
        X_train_fold, y_train_fold = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
        X_val_fold, y_val_fold = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]
        
        print(f"Fold {i+1} (with purging and embargo):")
        print(f"  Train size: {len(X_train_fold)}, non-contiguous periods due to purging")
        print(f"  Val size: {len(X_val_fold)}, period: {X_val_fold.index.min()} to {X_val_fold.index.max()}")
    
    print(f"Final Test size: {len(X_test)}, period: {X_test.index.min()} to {X_test.index.max()}")
    