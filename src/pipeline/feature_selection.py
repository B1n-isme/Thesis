import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.data.process import run_data_processing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def select_features_and_transform(
    X_train_w: np.ndarray, 
    y_train_w: np.ndarray, 
    X_test_w: np.ndarray,
    original_processed_feature_names: List[str],
    lookback_window: int,
    method: str = 'rf', 
    n_estimators_rf: int = 100,
    max_features_to_select: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Any]:
    """
    Selects features using a tree-based model and transforms the datasets.

    Args:
        X_train_w: 3D numpy array for training features (samples, timesteps, original_features).
        y_train_w: 1D numpy array for training target.
        X_test_w: 3D numpy array for test features (samples, timesteps, original_features).
        original_processed_feature_names: List of names for the original features
                                          (before windowing but after other processing).
        lookback_window: The lookback window size.
        method: Feature selection method ('rf', 'lgbm', 'xgb').
        n_estimators_rf: Number of estimators for Random Forest.
        max_features_to_select: Maximum number of features to select. If None, uses model's default.
        random_state: Random state for reproducibility.

    Returns:
        Tuple containing:
        - X_train_selected: 2D training features after selection.
        - X_test_selected: 2D test features after selection.
        - selected_feature_names: List of names for the selected features.
        - selector: The fitted SelectFromModel instance.
    """
    if X_train_w.ndim != 3 or X_test_w.ndim != 3:
        raise ValueError("X_train_w and X_test_w must be 3D arrays (samples, timesteps, features).")
    
    n_samples_train, n_timesteps, n_original_features = X_train_w.shape
    if n_timesteps != lookback_window:
        raise ValueError(f"X_train_w's second dimension ({n_timesteps}) does not match lookback_window ({lookback_window}).")

    # Use only the last timestep's features for selection
    X_train_last = X_train_w[:, -1, :]
    X_test_last = X_test_w[:, -1, :]

    print(f"\n--- Feature Selection using {method.upper()} ---")
    print(f"Original number of features: {X_train_last.shape[1]}")

    model: Any
    if method == 'rf':
        model = RandomForestRegressor(
            n_estimators=n_estimators_rf, 
            random_state=random_state, 
            n_jobs=-1
        )
    elif method == 'lgbm':
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed. Please install it to use 'lgbm' method.")
        model = LGBMRegressor(random_state=random_state, n_jobs=-1, verbosity=-1)
    elif method == 'xgb':
        if XGBRegressor is None:
            raise ImportError("XGBoost is not installed. Please install it to use 'xgb' method.")
        model = XGBRegressor(random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported feature selection method: {method}. Choose 'rf', 'lgbm', or 'xgb'.")

    # If y_train_w is 2D, reduce to 1D for feature selection
    if y_train_w.ndim == 2:
        y_train_for_fs = y_train_w[:, 0]
    else:
        y_train_for_fs = y_train_w

    print(f"Fitting {method.upper()} model for feature importances...")
    model.fit(X_train_last, y_train_for_fs)

    threshold_for_sfm = 'median' if max_features_to_select is None else -np.inf
    
    selector = SelectFromModel(
        model, 
        prefit=True, 
        threshold=threshold_for_sfm, 
        max_features=max_features_to_select
    )

    X_train_selected = selector.transform(X_train_last)
    X_test_selected = selector.transform(X_test_last)

    print(f"Number of features selected: {X_train_selected.shape[1]}")

    selected_mask = selector.get_support()
    selected_feature_names = [
        original_processed_feature_names[i] 
        for i, selected in enumerate(selected_mask) if selected
    ]

    return X_train_selected, X_test_selected, selected_feature_names, selector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform feature selection on windowed time series data.")
    parser.add_argument("--data_path", type=str, default='data/final/dataset.csv', help="Path to the input dataset CSV file.")
    parser.add_argument("--lookback_window", type=int, default=5, help="Lookback window size for creating sequences.")
    parser.add_argument("--prediction_horizon", type=int, default=1, help="Prediction horizon.")
    parser.add_argument("--pred_len", type=int, default=1, help="Prediction length.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for windowing.")
    
    parser.add_argument("--method", type=str, default='rf', choices=['rf', 'lgbm', 'xgb'], help="Feature selection method: 'rf' (Random Forest), 'lgbm' (LightGBM), 'xgb' (XGBoost).")
    parser.add_argument("--n_estimators_rf", type=int, default=100, help="Number of estimators for Random Forest (if method='rf').")
    parser.add_argument("--max_features", type=int, default=None, help="Maximum number of features to select. If None, uses a median importance threshold.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")

    args = parser.parse_args()

    print("--- Starting Data Processing --- ")
    processed_data = run_data_processing(
        data_path=args.data_path,
        test_split_ratio=0.2, # Standard test split
        n_cv_splits_for_tscv=3, # Standard CV splits, not directly used here but processed_data expects it
        lookback_window=args.lookback_window,
        prediction_horizon=args.prediction_horizon,
        pred_len=args.pred_len,
        stride=args.stride
    )

    if 'windowed_trainval_data' not in processed_data or 'windowed_test_data' not in processed_data:
        print("Error: Windowed data not found in processed_data. Ensure lookback_window is set and processing was successful.")
        exit(1)
    
    if 'processed_feature_names' not in processed_data:
        print("Error: 'processed_feature_names' not found in processed_data. Make sure src.data.process is updated.")
        exit(1)

    windowed_trainval = processed_data['windowed_trainval_data']
    windowed_test = processed_data['windowed_test_data']
    original_features = processed_data['processed_feature_names']
    lookback_used = processed_data.get('lookback_window_used', args.lookback_window) # Use returned if available

    if not windowed_trainval or not windowed_test:
        print("Error: Windowed trainval or test data is empty.")
        exit(1)
    
    X_train_w = windowed_trainval.get('X_w')
    y_train_w = windowed_trainval.get('y_w')
    X_test_w = windowed_test.get('X_w')
    # y_test_w is not directly used for selection but would be in windowed_test.get('y_w')

    if X_train_w is None or y_train_w is None or X_test_w is None:
        print("Error: X_w or y_w missing from windowed data dictionaries.")
        exit(1)
    
    if X_train_w.size == 0 or X_test_w.size == 0:
        print("Error: Windowed training or testing feature data is empty. Check lookback window and data size.")
        exit(1)

    try:
        X_train_selected, X_test_selected, selected_names, selector_model = select_features_and_transform(
            X_train_w=X_train_w,
            y_train_w=y_train_w,
            X_test_w=X_test_w,
            original_processed_feature_names=original_features,
            lookback_window=lookback_used,
            method=args.method,
            n_estimators_rf=args.n_estimators_rf,
            max_features_to_select=args.max_features,
            random_state=args.random_state
        )

        print("\n--- Feature Selection Results ---")
        print(f"Method used: {args.method.upper()}")
        print(f"Number of features selected: {len(selected_names)}")
        print("Selected feature names:")
        for i, name in enumerate(selected_names):
            print(f"  {i+1}. {name}")
        
        print(f"\nShape of training data after selection: {X_train_selected.shape}")
        print(f"Shape of test data after selection: {X_test_selected.shape}")
        
        # Here you could save the selector_model, X_train_selected, X_test_selected, or selected_names
        # For example:
        # import joblib
        # joblib.dump(selector_model, f'feature_selector_{args.method}.joblib')
        # np.save(f'X_train_selected_{args.method}.npy', X_train_selected)
        # np.save(f'X_test_selected_{args.method}.npy', X_test_selected)
        # with open(f'selected_features_{args.method}.txt', 'w') as f:
        #     for name in selected_names:
        #         f.write(f"{name}\n")
        print("\nFeature selection process complete.")

    except Exception as e:
        print(f"An error occurred during feature selection: {e}")
        import traceback
        traceback.print_exc() 

# python src/pipeline/feature_selection.py --lookback_window 10 --method rf --max_features 20