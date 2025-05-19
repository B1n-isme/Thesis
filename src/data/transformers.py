import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from typing import Tuple, Dict, Any, Optional, Union

def fit_and_apply_skewness_correction(
    data_to_transform: pd.DataFrame,
    fit_reference_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Union[PowerTransformer, str]]]:
    """
    Fits PowerTransformers on reference data and applies them to the target data.

    Identifies columns needing skewness correction in `fit_reference_data`,
    fits PowerTransformer (Box-Cox or Yeo-Johnson) accordingly, and then
    applies these fitted transformers to `data_to_transform`.

    Args:
        data_to_transform (pd.DataFrame): The DataFrame to apply transformations to.
                                          It is expected to be a copy if mutation
                                          of the original is to be avoided.
        fit_reference_data (pd.DataFrame): The DataFrame on which to fit the
                                           PowerTransformers.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Union[PowerTransformer, str]]]:
        - DataFrame with skewness correction applied to relevant columns.
        - Dictionary of fitted PowerTransformer objects or 'passthrough'/'error_passthrough' strings.
    """
    fitted_transformers: Dict[str, Union[PowerTransformer, str]] = {}
    transformed_data = data_to_transform.copy() # Work on a copy

    print(f"  Fitting Skewness Transformers on reference data (shape {fit_reference_data.shape}):")
    for col in fit_reference_data.columns:
        skewness = fit_reference_data[col].skew()
        transformer_to_store: Union[PowerTransformer, str] = 'passthrough'

        if -0.5 <= skewness <= 0.5:
            # print(f"    Column '{col}': Symmetric (skew={skewness:.2f}), skipping power transformation.")
            transformer_to_store = 'passthrough'
        else:
            method_choice = 'yeo-johnson'
            if skewness > 0.5 and (fit_reference_data[col] > 0).all():
                method_choice = 'box-cox'
                # print(f"    Column '{col}': Positive skew ({skewness:.2f}), all positive. Fitting Box-Cox.")
            # else:
                # print(f"    Column '{col}': Skew={skewness:.2f}, min_val={fit_reference_data[col].min():.2f}. Fitting Yeo-Johnson.")
            
            pt = PowerTransformer(method=method_choice, standardize=False)
            try:
                # Fit on the reference data column
                pt.fit(fit_reference_data[[col]])
                transformer_to_store = pt
                # print(f"      Fitted {method_choice} for '{col}'.")
            except ValueError as e:
                print(f"      ERROR fitting PowerTransformer for '{col}' on reference data: {e}. Storing as 'error_passthrough'.")
                transformer_to_store = 'error_passthrough'
        
        fitted_transformers[col] = transformer_to_store

    # Now apply the fitted transformers to the data_to_transform
    print(f"  Applying Skewness Transformers to target data (shape {data_to_transform.shape}):")
    for col in data_to_transform.columns:
        transformer = fitted_transformers.get(col)
        if isinstance(transformer, PowerTransformer):
            # print(f"    Column '{col}': Applying fitted {transformer.method}")
            col_data_target = data_to_transform[[col]]
            try:
                # Handle Box-Cox non-positive values in the data being transformed
                if transformer.method == 'box-cox' and (col_data_target[col] <= 0).any():
                    print(f"      WARNING: Target Column '{col}' for Box-Cox has non-positive values. Clipping to 1e-9 before transform.")
                    data_for_transform = col_data_target.copy()
                    data_for_transform[col] = np.maximum(data_for_transform[col], 1e-9)
                    transformed_data[col] = transformer.transform(data_for_transform).flatten()
                else:
                    transformed_data[col] = transformer.transform(col_data_target).flatten()
                
                # new_skew = pd.Series(transformed_data[col]).skew()
                # print(f"      Column '{col}' new skew: {new_skew:.4f}")

            except ValueError as e:
                print(f"      ERROR applying PowerTransformer for '{col}' to target data: {e}. Data for this column remains unchanged.")
                # transformed_data[col] will retain its original values from data_to_transform.copy()
        # elif transformer == 'passthrough' or transformer == 'error_passthrough':
            # print(f"    Column '{col}': Passthrough or error during fit. Data unchanged.")

    return transformed_data, fitted_transformers


def apply_fitted_skewness_correction(
    data_to_transform: pd.DataFrame,
    fitted_transformers: Dict[str, Union[PowerTransformer, str]]
) -> pd.DataFrame:
    """
    Applies pre-fitted PowerTransformers to the target data.

    Args:
        data_to_transform (pd.DataFrame): The DataFrame to apply transformations to.
                                          It is expected to be a copy.
        fitted_transformers (Dict[str, Union[PowerTransformer, str]]): 
            Dictionary of pre-fitted PowerTransformer objects or 'passthrough'/'error_passthrough' strings.

    Returns:
        pd.DataFrame: DataFrame with skewness correction applied.
    """
    transformed_data = data_to_transform.copy() # Work on a copy
    print(f"  Applying existing Skewness Transformers to data (shape {data_to_transform.shape}):")

    for col in data_to_transform.columns:
        transformer = fitted_transformers.get(col)
        if isinstance(transformer, PowerTransformer):
            # print(f"    Column '{col}': Applying pre-fitted {transformer.method}")
            col_data_target = data_to_transform[[col]]
            try:
                if transformer.method == 'box-cox' and (col_data_target[col] <= 0).any():
                    print(f"      WARNING: Column '{col}' for Box-Cox has non-positive values. Clipping to 1e-9 before transform.")
                    data_for_transform = col_data_target.copy()
                    data_for_transform[col] = np.maximum(data_for_transform[col], 1e-9)
                    transformed_data[col] = transformer.transform(data_for_transform).flatten()
                else:
                    transformed_data[col] = transformer.transform(col_data_target).flatten()
                # new_skew = pd.Series(transformed_data[col]).skew()
                # print(f"      Column '{col}' new skew: {new_skew:.4f}")
            except ValueError as e:
                print(f"      ERROR applying pre-fitted PowerTransformer for '{col}': {e}. Data for this column remains unchanged.")
        # elif transformer == 'passthrough' or transformer == 'error_passthrough':
            # print(f"    Column '{col}': Passthrough or error during fit. Data unchanged.")
            
    return transformed_data


def fit_and_apply_scaling(
    data_to_transform: pd.DataFrame,
    fit_reference_data: pd.DataFrame
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Fits a StandardScaler on reference data and applies it to the target data.

    Args:
        data_to_transform (pd.DataFrame): The DataFrame to scale.
        fit_reference_data (pd.DataFrame): The DataFrame on which to fit the StandardScaler.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]:
        - Scaled DataFrame.
        - Fitted StandardScaler object.
    """
    scaler = StandardScaler()
    print(f"  Fitting StandardScaler on reference data (shape {fit_reference_data.shape}).")
    scaler.fit(fit_reference_data)
    
    print(f"  Applying StandardScaler to target data (shape {data_to_transform.shape}).")
    scaled_data_np = scaler.transform(data_to_transform)
    scaled_df = pd.DataFrame(scaled_data_np, index=data_to_transform.index, columns=data_to_transform.columns)
    
    # print(f"    Target data scaled mean (approx 0 for ref cols): {scaled_df.mean().mean():.2f}")
    # print(f"    Target data scaled std (approx 1 for ref cols): {scaled_df.std().mean():.2f}")
    return scaled_df, scaler

def apply_fitted_scaling(
    data_to_transform: pd.DataFrame,
    fitted_scaler: StandardScaler
) -> pd.DataFrame:
    """
    Applies a pre-fitted StandardScaler to the target data.

    Args:
        data_to_transform (pd.DataFrame): The DataFrame to scale.
        fitted_scaler (StandardScaler): The pre-fitted StandardScaler object.

    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    print(f"  Applying existing StandardScaler to data (shape {data_to_transform.shape}).")
    scaled_data_np = fitted_scaler.transform(data_to_transform)
    scaled_df = pd.DataFrame(scaled_data_np, index=data_to_transform.index, columns=data_to_transform.columns)
    # print(f"    Data scaled mean: {scaled_df.mean().mean():.2f}")
    # print(f"    Data scaled std: {scaled_df.std().mean():.2f}")
    return scaled_df 