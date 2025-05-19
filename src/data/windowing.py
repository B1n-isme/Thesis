import pandas as pd
import numpy as np
from typing import Tuple, Union, List, Any

def create_windowed_dataset( # Renamed to indicate it's an updated version
    features_data: Union[pd.DataFrame, pd.Series],
    target_data: pd.Series,
    lookback_window: int,
    horizon: int = 1,
    stride: int = 1,
    dropna_target: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Creates windowed datasets for time series forecasting.

    This function takes time-indexed features and target data, and generates
    fixed-size lookback windows for features (X) and corresponding future
    target values (y).

    Args:
        features_data (Union[pd.DataFrame, pd.Series]): DataFrame or Series of feature values.
            If a Series is provided, it's treated as a single feature.
            The index must be a DatetimeIndex or similar, alignable with target_data.
        target_data (pd.Series): Series of target values. Must be alignable with features_data.
            (For future extension, this could be a pd.DataFrame for multi-target prediction).
        lookback_window (int): The number of past time steps to use as input features
            for each sample. Must be at least 1.
        horizon (int): The number of time steps ahead to predict from the end of the
            lookback window. horizon=1 means predict the very next step
            after the lookback window. Must be at least 1.
        stride (int): The step size to slide the window across the data.
            stride=1 means the window moves one time step at a time. Must be at least 1.
        dropna_target (bool): If True (default), samples where the target value (current_y_value)
            is NaN will be dropped. If False, NaNs will be included in the target array.

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.Index]:
        - X (np.ndarray): A 3D array of shape (n_samples, lookback_window, n_features)
                          containing the feature windows.
        - y (np.ndarray): A 1D array of shape (n_samples,) containing the target values.
                          (If `target_data` were a DataFrame for multi-target and appropriately
                           processed, `y` could be 2D: (n_samples, n_targets)).
        - target_indices (pd.Index): The original index from `target_data`
                                     corresponding to each `y` value. This helps in
                                     aligning predictions with actual dates.

    Raises:
        TypeError: If `features_data` is not a pandas DataFrame or Series, or
                   if `target_data` is not a pandas Series.
        ValueError: If `lookback_window`, `horizon`, or `stride` are less than 1.
        ValueError: If `features_data` and `target_data` have no common indices,
                    or if data is empty after alignment.
        ValueError: If the total length of the data (after alignment) is too short
                    to form even a single window with the given `lookback_window`
                    and `horizon`.
    """

    # Input Value Validation
    if lookback_window < 1:
        raise ValueError("lookback_window must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")

    # Prepare features DataFrame (use a copy to avoid modifying original)
    if isinstance(features_data, pd.Series):
        features_df = features_data.to_frame()
    else:
        features_df = features_data.copy()

    target_s = target_data.copy() # Use a copy for target as well

    # Index Alignment (Suggestion 7: Refined warning)
    original_feature_len = len(features_df.index)
    original_target_len = len(target_s.index)

    try:
        common_index = features_df.index.intersection(target_s.index)
    except Exception as e:
        raise ValueError(f"Failed to compute common index for features_data and target_data: {e}")


    if len(common_index) < original_feature_len or len(common_index) < original_target_len:
        print(
            f"Warning: Original features_data ({original_feature_len} rows) and "
            f"target_data ({original_target_len} rows) indices do not perfectly match. "
            f"Using intersection of {len(common_index)} rows."
        )

    if common_index.empty:
        raise ValueError(
            "features_data and target_data have no common indices for alignment."
        )

    features_df = features_df.loc[common_index]
    target_s = target_s.loc[common_index]

    if features_df.empty: # If features_df is empty, target_s will also be empty.
        raise ValueError(
            "Data is empty after index alignment. This can occur if original data had no "
            "overlapping time periods or the data within the common period is effectively empty."
        )

    # (Suggestion 1: Redundant length check `len(features_df) != len(target_s)` removed)
    # Alignment ensures they are of the same length if not empty.

    # Check for sufficient data length to form at least one sample
    min_required_len = lookback_window + horizon
    if len(features_df) < min_required_len:
        raise ValueError(
            f"Aligned data length ({len(features_df)}) is too short for the given "
            f"lookback_window ({lookback_window}) and horizon ({horizon}). "
            f"Need at least {min_required_len} data points to form one sample."
        )

    X_list: List[np.ndarray] = []
    y_list: List[Any] = [] # Using Any for broader scalar type compatibility from target_s.iloc[]
    target_idx_list: List[Any] = [] # Suggestion 6: Using Any for broader index type compatibility

    # Calculate the number of possible start points for windows
    # The last data point used for a target is at index: i_max + lookback_window + horizon - 1
    # This must be < len(features_df).
    # So, i_max < len(features_df) - lookback_window - horizon + 1
    # num_possible_starts is the count of valid 'i' values (0-indexed)
    num_possible_starts = len(features_df) - lookback_window - horizon + 1


    # (Suggestion 5: Loop with stride)
    for i in range(0, num_possible_starts, stride):
        feature_window_start_idx = i
        feature_window_end_idx = i + lookback_window
        
        # The target is `horizon` steps *after* the end of the feature window
        target_val_idx = feature_window_end_idx + horizon - 1
        
        # This check should ideally not be needed if num_possible_starts is correct,
        # but it's a safeguard, especially if stride causes overshooting in some edge logic.
        if target_val_idx >= len(target_s):
            break 

        current_X_window = features_df.iloc[feature_window_start_idx:feature_window_end_idx].values
        current_y_value = target_s.iloc[target_val_idx]
        # Note on extensibility for multi-target:
        # If target_s were a DataFrame, current_y_value would be a Series.
        # You might then use current_y_value.values to get a NumPy array for y_list.

        # (Suggestion 2: Handle NaNs in target)
        if dropna_target:
            # pd.isna() handles scalars, Series, etc. .any() ensures if current_y_value is a Series (multi-target),
            # it drops if *any* of the targets are NaN. For scalar, it's direct.
            if pd.isna(current_y_value) if np.isscalar(current_y_value) else pd.isna(current_y_value).any():
                continue
        
        current_target_original_index = target_s.index[target_val_idx]

        X_list.append(current_X_window)
        y_list.append(current_y_value)
        target_idx_list.append(current_target_original_index)

    if not X_list:
        # This can happen if all targets were NaN and dropna_target=True,
        # or if num_possible_starts was positive but stride made the loop not run (e.g. num_possible_starts=1, stride=2)
        # or if num_possible_starts was <=0 (though min_required_len check should catch this).
        num_features = features_df.shape[1]
        return np.array([]).reshape(0, lookback_window, num_features), np.array([]), pd.Index([])

    X_np = np.array(X_list)
    y_np = np.array(y_list) # For pd.Series target, y_np will be 1D.
    target_indices_pd = pd.Index(target_idx_list)

    return X_np, y_np, target_indices_pd