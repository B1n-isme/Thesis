import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional, Dict, Any
from sktime.split import temporal_train_test_split
from src.data.split import get_splitter
from src.data.windowing import create_windowed_dataset
from src.data.transformers import (
    fit_and_apply_skewness_correction,
    apply_fitted_skewness_correction,
    fit_and_apply_scaling,
    apply_fitted_scaling
)
from src.data.loader import load_and_sort_csv

def run_data_processing(data_path: str = 'data/final/dataset.csv',
                        test_split_ratio: float = 0.2,
                        n_cv_splits_for_tscv: int = 3,
                        gap_for_tscv: int = 0,
                        lookback_window: Optional[int] = None,
                        prediction_horizon: int = 1,
                        pred_len: int = 6,
                        stride: int = 1
                        ) -> Dict[str, Any]:
    """
    Processes the time series data, including splitting, skewness transformation,
    scaling, and optional windowing.

    The function performs the following steps:
    1.  Loads and sorts the data.
    2.  Separates features (X) and target (y). Optionally transforms the target
        (e.g., log-difference) and aligns features. A flag `y_was_transformed`
        is set accordingly.
    3.  Splits data into training/validation (trainval) and a final test set.
    4.  Sets up TimeSeriesSplit for cross-validation on the trainval set.
    5.  In a CV loop:
        a.  Fits skewness transformers (PowerTransformer) on the training fold.
        b.  Applies these transformers to the validation fold.
        c.  Fits a StandardScaler on the skew-transformed training fold.
        d.  Applies the scaler to the skew-transformed validation fold.
        e.  If `lookback_window` is provided, creates windowed versions of
            processed training and validation folds and stores them.
    6.  Fits final skewness transformers and StandardScaler on the entire trainval set.
    7.  Applies these final transformers to the test set.
    8.  If `lookback_window` is provided, creates windowed versions of the fully
        processed trainval and test sets.

    Args:
        data_path (str): Path to the input dataset CSV file.
        test_split_ratio (float): Proportion of the dataset to allocate to the
            final test set.
        n_cv_splits_for_tscv (int): Number of splits for TimeSeriesSplit cross-validator.
        gap_for_tscv (int): Gap between train and test sets in TimeSeriesSplit.
        lookback_window (Optional[int]): The number of past time steps to use as
            input features. If None, windowing is not performed. Defaults to None.
        prediction_horizon (int): The number of time steps ahead to predict from the
            end of the lookback window. Used if windowing is performed. Defaults to 1.
        stride (int): The step size to slide the window across the data. Used if
            windowing is performed. Defaults to 1.

    Returns:
        Dict[str, Any]: A dictionary containing processed data components:
            - 'final_skew_transformers': Dictionary of skew transformers fitted on X_trainval.
            - 'final_scaler': StandardScaler fitted on X_trainval.
            - 'y_test': Target Series for the final test set (potentially transformed).
            - 'original_target_series': The original, untransformed target Series (`y_orig`).
            - 'y_was_transformed': Boolean flag indicating if the target variable was transformed.
            - 'processed_feature_names': List of feature names after scaling/transformations on X_trainval.
            If lookback_window is specified and > 0:
            - 'lookback_window_used' (int): The lookback window value used.
            - 'prediction_horizon_used' (int): The prediction horizon value used.
            - 'stride_used' (int): The stride value used.
            - 'windowed_cv_folds' (List[Dict]): List of dicts, each containing
                windowed CV data:
                { 'X_train_w': np.ndarray, 'y_train_w': np.ndarray, 'y_idx_train_w': pd.Index,
                  'X_val_w': np.ndarray, 'y_val_w': np.ndarray, 'y_idx_val_w': pd.Index }
                (Only present if `windowed_cv_folds_list` is not empty).
            - 'windowed_trainval_data' (Dict): Windowed trainval data:
                { 'X_w': np.ndarray, 'y_w': np.ndarray, 'y_idx_w': pd.Index }
                (Only present if `windowed_trainval_data_dict` is not empty).
            - 'windowed_test_data' (Dict): Windowed test data:
                { 'X_w': np.ndarray, 'y_w': np.ndarray, 'y_idx_w': pd.Index }
                (Only present if `windowed_test_data_dict` is not empty).
    """
    df = load_and_sort_csv(data_path)

    df['Oil_Crude_Price'] = df['Oil_Crude_Price'].clip(lower=0)

    # --- Target Variable and Feature Separation ---
    X_orig = df.drop(columns=['btc_close'], axis=1)
    y_orig_series = df['btc_close'] # Renamed from y_orig to avoid confusion later

    y_was_transformed_flag = False
    if (y_orig_series > 0).all():
        y_transformed_for_modeling = np.log(y_orig_series).diff().dropna()
        # Align X with the new y
        X_aligned_for_modeling = X_orig.loc[y_transformed_for_modeling.index].copy()
        y_was_transformed_flag = True
    else:
        print("Warning: y_orig_series contains non-positive values. Log transform for y is problematic.")
        y_transformed_for_modeling = y_orig_series.copy()
        X_aligned_for_modeling = X_orig.copy()
        y_was_transformed_flag = False

    # Using aligned and potentially transformed versions for modeling
    X = X_aligned_for_modeling.copy()
    y = y_transformed_for_modeling.copy()

    print("--- Initial Data Shapes (using potentially transformed y and aligned X) ---")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    if not X.index.equals(y.index):
        raise ValueError("X and y indices do not match after transformation. Ensure alignment before proceeding.")
    if X.empty or y.empty:
        raise ValueError("X or y is empty after initial transformation/alignment. Check data and transformation steps.")
    print(f"Data period: {X.index.min()} to {X.index.max()}")

    # --- 1. Split Data: Train-Validation-Test ---
    X_trainval, X_test, y_trainval, y_test = temporal_train_test_split(
        X, y, test_size=test_split_ratio
    )

    print("\n--- 1. Initial Train/Validation vs. Test Split ---")
    print(f"X_trainval size: {len(X_trainval)}, period: {X_trainval.index.min()} to {X_trainval.index.max()}")
    print(f"y_trainval size: {len(y_trainval)}")
    print(f"X_test size: {len(X_test)}, period: {X_test.index.min()} to {X_test.index.max()}")
    print(f"y_test size: {len(y_test)}")

    if X_trainval.empty or y_trainval.empty:
        raise ValueError("X_trainval or y_trainval is empty after initial split. Check test_split_ratio and data size.")

    test_size_cv = len(X_trainval) // (n_cv_splits_for_tscv + 2)
    if test_size_cv == 0 and len(X_trainval) > 0: test_size_cv = 1

    print(f"\nUsing TimeSeriesSplit for CV with n_splits={n_cv_splits_for_tscv}, test_size_cv={test_size_cv} (samples per val fold)")
    tscv = get_splitter(method="sklearn", n_splits=n_cv_splits_for_tscv, test_size=test_size_cv, gap=gap_for_tscv)

    # --- 2. Cross-Validation Loop with Correct Transformation Handling ---
    print("\n--- 2. Cross-Validation Folds Processing ---")
    # Remove unused lists for processed folds
    # processed_X_train_folds: List[pd.DataFrame] = []
    # processed_y_train_folds: List[pd.Series] = []
    # processed_X_val_folds: List[pd.DataFrame] = []
    # processed_y_val_folds: List[pd.Series] = []
    # fold_transformers_and_scalers: List[Dict[str, Any]] = []

    # New lists for windowed CV data if lookback_window is specified
    windowed_cv_folds_list: List[Dict[str, Any]] = []

    for i, (train_idx, val_idx) in enumerate(tscv.split(X_trainval)):
        fold_num = i + 1
        print(f"\n--- Processing Fold {fold_num} ---")

        X_train_fold, y_train_fold = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
        X_val_fold, y_val_fold = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]

        print(f"  Original Fold {fold_num} Train: {len(X_train_fold)} obs ({X_train_fold.index.min()} to {X_train_fold.index.max()})")
        print(f"  Original Fold {fold_num} Val:   {len(X_val_fold)} obs ({X_val_fold.index.min()} to {X_val_fold.index.max()})")

        if X_train_fold.empty or X_val_fold.empty:
            print(f"  Skipping Fold {fold_num} due to empty train or validation set from splitter configuration.")
            continue

        # --- 2a. Fit Skewness Transformers on X_train_fold & Apply to X_train_fold --- 
        X_train_fold_skew_transformed, fold_skew_transformers = fit_and_apply_skewness_correction(
            data_to_transform=X_train_fold, # This will be copied inside the function
            fit_reference_data=X_train_fold
        )

        # --- 2b. Apply Skewness Transformers to X_val_fold --- 
        X_val_fold_skew_transformed = apply_fitted_skewness_correction(
            data_to_transform=X_val_fold, # This will be copied inside the function
            fitted_transformers=fold_skew_transformers
        )
        
        # --- 2c. Fit StandardScaler on Skew-Transformed X_train_fold & Apply to it --- 
        X_train_fold_final, fold_scaler = fit_and_apply_scaling(
            data_to_transform=X_train_fold_skew_transformed, # This will be copied
            fit_reference_data=X_train_fold_skew_transformed # Fit on the same data
        )

        # --- 2d. Apply StandardScaler to Skew-Transformed X_val_fold --- 
        X_val_fold_final = apply_fitted_scaling(
            data_to_transform=X_val_fold_skew_transformed, # This will be copied
            fitted_scaler=fold_scaler
        )

        # Remove appending to unused lists
        # processed_X_train_folds.append(X_train_fold_final)
        # processed_y_train_folds.append(y_train_fold)
        # processed_X_val_folds.append(X_val_fold_final)
        # processed_y_val_folds.append(y_val_fold)
        # fold_transformers_and_scalers.append({'skew_transformers': fold_skew_transformers, 'scaler': fold_scaler})

        # --- 2e. Apply Windowing to Fold Data (if lookback_window is specified) ---
        if lookback_window is not None and lookback_window > 0:
            print(f"  Applying windowing to Fold {fold_num} (lookback={lookback_window}, horizon={prediction_horizon})")
            try:
                X_train_fold_w, y_train_fold_w, y_idx_train_fold_w = create_windowed_dataset(
                    X_train_fold_final, y_train_fold, lookback_window, prediction_horizon, pred_len, stride
                )
                X_val_fold_w, y_val_fold_w, y_idx_val_fold_w = create_windowed_dataset(
                    X_val_fold_final, y_val_fold, lookback_window, prediction_horizon, pred_len, stride
                )
                # Only add if both train and val windowing were successful and produced non-empty arrays for X
                if X_train_fold_w.size > 0 and X_val_fold_w.size > 0:
                    windowed_cv_folds_list.append({
                        'X_train_w': X_train_fold_w,
                        'y_train_w': y_train_fold_w,
                        'y_idx_train_w': y_idx_train_fold_w,
                        'X_val_w': X_val_fold_w,
                        'y_val_w': y_val_fold_w,
                        'y_idx_val_w': y_idx_val_fold_w
                    })
                    print(f"    Fold {fold_num} Train Windowed: X_w shape {X_train_fold_w.shape}, y_w shape {y_train_fold_w.shape}")
                    print(f"    Fold {fold_num} Val Windowed:   X_w shape {X_val_fold_w.shape}, y_w shape {y_val_fold_w.shape}")
                else:
                    print(f"    Skipping windowed data for Fold {fold_num} due to empty arrays after windowing (likely insufficient data in fold for lookback/horizon).")
            except ValueError as e:
                print(f"    ERROR during windowing for Fold {fold_num}: {e}. Skipping windowing for this fold.")
                pass 

        print(f"  Fold {fold_num} Processed Train (scaled) size: {len(X_train_fold_final)}, Val (scaled) size: {len(X_val_fold_final)}")

    print("\n--- 3. Final Transformations for Test Set Preparation ---")
    print("  Fitting transformations on the ENTIRE X_trainval set.")

    # --- 3a. Fit Skewness Transformers on ALL X_trainval data & Apply to X_trainval--- 
    X_trainval_skew_transformed, final_skew_transformers = fit_and_apply_skewness_correction(
        data_to_transform=X_trainval, # Will be copied inside
        fit_reference_data=X_trainval 
    )

    # --- 3b. Fit StandardScaler on Skew-Transformed X_trainval & Apply to it --- 
    X_trainval_final_for_model, final_scaler = fit_and_apply_scaling(
        data_to_transform=X_trainval_skew_transformed, # Will be copied inside
        fit_reference_data=X_trainval_skew_transformed
    )

    print("  Applying final (X_trainval-fitted) transformations to X_test:")
    # --- 3c. Apply FINAL Skewness Transformers to X_test ---
    X_test_skew_transformed = apply_fitted_skewness_correction(
        data_to_transform=X_test, # Will be copied inside
        fitted_transformers=final_skew_transformers
    )

    # --- 3d. Apply FINAL StandardScaler to Skew-Transformed X_test ---
    X_test_final_for_evaluation = apply_fitted_scaling(
        data_to_transform=X_test_skew_transformed, # Will be copied inside
        fitted_scaler=final_scaler
    )

    print("\n--- 4. Transformation Process Complete ---")

    # --- 5. Final Windowing (if lookback_window is specified) ---
    windowed_trainval_data_dict: Optional[Dict[str, Any]] = None
    windowed_test_data_dict: Optional[Dict[str, Any]] = None

    if lookback_window is not None and lookback_window > 0:
        print(f"\n--- 5. Applying Final Windowing (lookback={lookback_window}, horizon={prediction_horizon}, stride={stride}) ---")
        try:
            print("  Windowing X_trainval_final_for_model and y_trainval...")
            X_tv_w, y_tv_w, y_idx_tv_w = create_windowed_dataset(
                X_trainval_final_for_model, y_trainval, lookback_window, prediction_horizon, stride
            )
            if X_tv_w.size > 0: # Check if windowing produced actual data
                windowed_trainval_data_dict = {'X_w': X_tv_w, 'y_w': y_tv_w, 'y_idx_w': y_idx_tv_w}
                print(f"    Trainval Windowed: X_w shape {X_tv_w.shape}, y_w shape {y_tv_w.shape}")
            else:
                print("    Trainval windowing resulted in no samples (empty X_w).")
        except ValueError as e:
            print(f"    ERROR windowing trainval data: {e}")

        try:
            print("  Windowing X_test_final_for_evaluation and y_test...")
            X_test_w, y_test_w, y_idx_test_w = create_windowed_dataset(
                X_test_final_for_evaluation, y_test, lookback_window, prediction_horizon, stride
            )
            if X_test_w.size > 0: # Check if windowing produced actual data
                windowed_test_data_dict = {'X_w': X_test_w, 'y_w': y_test_w, 'y_idx_w': y_idx_test_w}
                print(f"    Test Windowed: X_w shape {X_test_w.shape}, y_w shape {y_test_w.shape}")
            else:
                print("    Test windowing resulted in no samples (empty X_w).")
        except ValueError as e:
            print(f"    ERROR windowing test data: {e}")


    # Construct the final results dictionary
    results: Dict[str, Any] = {
        'final_skew_transformers': final_skew_transformers,
        'final_scaler': final_scaler,
        'y_test': y_test,  # Target values for the test period (potentially transformed)
        'original_target_series': y_orig_series,  # Original full target series
        'y_was_transformed': y_was_transformed_flag,
        'processed_feature_names': X_trainval_final_for_model.columns.tolist()
    }

    if lookback_window is not None and lookback_window > 0:
        results['lookback_window_used'] = lookback_window
        results['prediction_horizon_used'] = prediction_horizon
        results['stride_used'] = stride # Added stride used

        # Add windowed data if available (lists/dicts might be empty if windowing failed or yielded no samples)
        if windowed_cv_folds_list: # Check if list is populated
            results['windowed_cv_folds'] = windowed_cv_folds_list
        if windowed_trainval_data_dict:
            results['windowed_trainval_data'] = windowed_trainval_data_dict
        if windowed_test_data_dict:
            results['windowed_test_data'] = windowed_test_data_dict
    
    return results

if __name__ == "__main__":
    print("Running data processing...")
    lookback = 5
    horizon_pred = 1
    stride = 1
    processed = run_data_processing(
        data_path='data/final/dataset.csv',
        lookback_window=lookback,
        prediction_horizon=horizon_pred,
        stride=stride
    )

    print("\n--- Data Processing Results ---")
    print(f"Processed features: {processed['processed_feature_names'][:5]} ... (total: {len(processed['processed_feature_names'])})")
    print(f"y_test shape: {processed['y_test'].shape}")
    print(f"Original target series shape: {processed['original_target_series'].shape}")
    print(f"y_was_transformed: {processed['y_was_transformed']}")

    if 'windowed_trainval_data' in processed:
        Xw = processed['windowed_trainval_data']['X_w']
        yw = processed['windowed_trainval_data']['y_w']
        print(f"Windowed trainval X shape: {Xw.shape}, y shape: {yw.shape}")
    if 'windowed_test_data' in processed:
        Xw = processed['windowed_test_data']['X_w']
        yw = processed['windowed_test_data']['y_w']
        print(f"Windowed test X shape: {Xw.shape}, y shape: {yw.shape}")
    if 'windowed_cv_folds' in processed:
        print(f"Windowed CV folds: {len(processed['windowed_cv_folds'])}")
        if processed['windowed_cv_folds']:
            fold = processed['windowed_cv_folds'][0]
            print(f"  Example fold train X: {fold['X_train_w'].shape}, y: {fold['y_train_w'].shape}")
            print(f"  Example fold val X: {fold['X_val_w'].shape}, y: {fold['y_val_w'].shape}")
    print("\nAccess any result via the returned dictionary, e.g. processed['y_test'], processed['windowed_trainval_data']['X_w']")