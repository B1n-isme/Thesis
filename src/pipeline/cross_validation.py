"""
Cross-validation module for neural forecasting models.
"""
import polars as pl
from neuralforecast import NeuralForecast
from src.config.forecasting_config import (
    FREQUENCY, LOCAL_SCALER_TYPE, CV_N_WINDOWS, 
    CV_STEP_SIZE, FORECAST_HORIZON
)
from src.utils.forecasting_utils import calculate_evaluation_metrics
from src.models.model_loader import load_best_model_config


def run_cross_validation(model_instance, df_development, n_windows=None, step_size=None):
    """Run cross-validation with the best model configuration."""
    if n_windows is None:
        n_windows = CV_N_WINDOWS
    if step_size is None:
        step_size = CV_STEP_SIZE
    
    print("\nStarting Cross-Validation with the best configuration...")
    
    # Create NeuralForecast object for Cross-Validation
    nf_cv = NeuralForecast(
        models=[model_instance],
        freq=FREQUENCY,
        local_scaler_type=LOCAL_SCALER_TYPE
    )
    
    # Run cross-validation
    # test_size in cross_validation refers to the validation horizon for each fold (should be h)
    # n_windows is the number of CV folds. step_size controls overlap.
    cv_results_df = nf_cv.cross_validation(
        df=df_development,
        n_windows=n_windows,
        step_size=step_size,
    )
    
    return cv_results_df, nf_cv


def evaluate_cv_results(cv_results_df):
    """Evaluate cross-validation results and calculate metrics."""
    # Convert pandas DataFrame to Polars DataFrame
    df_pl = pl.from_pandas(cv_results_df)
    
    # Define columns to exclude
    exclude_cols = ['unique_id', 'ds', 'cutoff', 'y']
    
    # Get the model columns dynamically
    model_cols = [col for col in df_pl.columns if col not in exclude_cols]
    
    # Calculate metrics for each model
    results = calculate_evaluation_metrics(df_pl, model_cols, exclude_cols)
    
    return results


def run_complete_cv_pipeline(df_development, model_name='AutoNHITS', csv_path=None):
    """Run the complete cross-validation pipeline."""
    # Load best model configuration
    model_instance, best_params, final_loss_object = load_best_model_config(model_name, csv_path)
    
    if model_instance is None:
        print("Failed to load model configuration. Aborting cross-validation.")
        return None, None, None
    
    # Run cross-validation
    cv_results_df, nf_cv = run_cross_validation(model_instance, df_development)
    
    # Evaluate results
    cv_metrics = evaluate_cv_results(cv_results_df)
    
    return cv_results_df, cv_metrics, nf_cv


if __name__ == "__main__":
    from src.data.data_preparation import prepare_forecasting_data
    
    # Prepare data
    _, df_development, _, _ = prepare_forecasting_data()
    
    # Run CV pipeline
    cv_results, metrics, nf_cv = run_complete_cv_pipeline(df_development)
    
    if cv_results is not None:
        print(f"\nCross-validation completed successfully!")
        print(f"CV results shape: {cv_results.shape}")
        print(f"Metrics: {metrics}")
    else:
        print("Cross-validation failed.") 