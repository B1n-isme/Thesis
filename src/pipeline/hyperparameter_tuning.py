"""
Hyperparameter tuning module for neural forecasting models.
"""
import os
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from models.model_definition import get_auto_models
from src.config.forecasting_config import (
    FORECAST_HORIZON, LEVELS, FREQUENCY, LOCAL_SCALER_TYPE,
    NUM_SAMPLES_PER_MODEL, TUNING_RESULTS_DIR
)


def run_hyperparameter_optimization(df_development, horizon=None, loss_fn=None, num_samples=None):
    """Run hyperparameter optimization using AutoModels."""
    if horizon is None:
        horizon = FORECAST_HORIZON
    if loss_fn is None:
        loss_fn = MAE()
    if num_samples is None:
        num_samples = NUM_SAMPLES_PER_MODEL
    
    print("\nStarting Hyperparameter Optimization with AutoModels...")
    
    # Get auto models for HPO
    automodels = get_auto_models(
        horizon=horizon,
        loss_fn=loss_fn,
        num_samples_per_model=num_samples
    )
    
    # Create NeuralForecast instance for HPO
    nf_hpo = NeuralForecast(
        models=automodels,
        freq=FREQUENCY,
        local_scaler_type=LOCAL_SCALER_TYPE
    )
    
    # Fit the models (performs HPO)
    nf_hpo.fit(df_development)
    
    return nf_hpo


def extract_best_configurations(nf_hpo):
    """Extract best configurations from HPO results."""
    all_best_configs = []
    
    for model in nf_hpo.models:
        # Check if the model is an Auto model and has results
        if hasattr(model, 'results') and model.results is not None:
            model_name = model.__class__.__name__
            print(f"Processing results for {model_name}...")
            
            # Get the DataFrame of all trials for this model
            results_df = model.results.get_dataframe()
            
            if not results_df.empty:
                # Find the row with the lowest 'valid_loss'
                best_run = results_df.loc[results_df['loss'].idxmin()]
                
                # Extract the 'config/' columns to get the hyperparameters
                best_params = {
                    col.replace('config/', ''): best_run[col]
                    for col in results_df.columns if col.startswith('config/')
                }
                
                # Add model name and best loss to the dictionary
                best_params['model_name'] = model_name
                best_params['best_valid_loss'] = best_run['loss']
                best_params['training_iteration'] = best_run['training_iteration']
                
                # Append to the list
                all_best_configs.append(best_params)
                print(f"Best config for {model_name}: {best_params}")
            else:
                print(f"No tuning results found for {model_name}.")
        else:
            print(f"Model {model.__class__.__name__} is not an Auto model or has no results.")
    
    return all_best_configs


def save_best_configurations(all_best_configs, output_dir=None):
    """Save best configurations to CSV file."""
    if output_dir is None:
        output_dir = TUNING_RESULTS_DIR
    
    if all_best_configs:
        best_configs_df = pd.DataFrame(all_best_configs)
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = os.path.join(output_dir, 'best_hyperparameters.csv')
        
        # Save to CSV
        best_configs_df.to_csv(csv_filename, index=False)
        print(f"\nBest hyperparameters saved to {csv_filename}")
        print("\nContent of best_hyperparameters.csv:")
        print(best_configs_df)
        
        return csv_filename
    else:
        print("No best configurations were found for any model.")
        return None


def run_complete_hpo_pipeline(df_development, horizon=None, loss_fn=None, num_samples=None):
    """Run the complete hyperparameter optimization pipeline."""
    # Run HPO
    nf_hpo = run_hyperparameter_optimization(
        df_development, 
        horizon=horizon, 
        loss_fn=loss_fn, 
        num_samples=num_samples
    )
    
    # Extract best configurations
    all_best_configs = extract_best_configurations(nf_hpo)
    
    # Save best configurations
    csv_filename = save_best_configurations(all_best_configs)
    
    return nf_hpo, csv_filename


if __name__ == "__main__":
    from src.data.data_preparation import prepare_forecasting_data
    
    # Prepare data
    _, df_development, _, _ = prepare_forecasting_data()
    
    # Run HPO
    nf_hpo, csv_file = run_complete_hpo_pipeline(df_development)
    print(f"HPO completed. Results saved to: {csv_file}") 