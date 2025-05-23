"""
Model configuration loader for neural forecasting models.
"""
import json
import pandas as pd
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
from src.config.forecasting_config import (
    BEST_HYPERPARAMETERS_CSV, LOSS_MAP, 
    EXCLUDE_HYPERPARAMETER_KEYS, JSON_PARSEABLE_KEYS
)


def load_best_hyperparameters(csv_path=None):
    """Load best hyperparameters from CSV file."""
    if csv_path is None:
        csv_path = BEST_HYPERPARAMETERS_CSV
    
    try:
        loaded_best_configs_df = pd.read_csv(csv_path)
        print(f"\nLoaded best configs from {csv_path}:")
        print(loaded_best_configs_df)
        return loaded_best_configs_df
    
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please ensure the tuning script ran and saved the CSV.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading hyperparameters: {e}")
        return None


def parse_hyperparameters(best_row, exclude_keys=None, json_keys=None):
    """Parse hyperparameters from the best configuration row."""
    if exclude_keys is None:
        exclude_keys = EXCLUDE_HYPERPARAMETER_KEYS
    if json_keys is None:
        json_keys = JSON_PARSEABLE_KEYS
    
    best_params = {}
    
    for key, value in best_row.to_dict().items():
        if key not in exclude_keys:
            # Attempt to parse values that are expected to be lists/nested lists
            if key in json_keys:
                try:
                    best_params[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    print(f"Warning: Could not parse '{key}' value '{value}' as JSON. Keeping as original.")
                    best_params[key] = value
            else:
                best_params[key] = value
    
    return best_params


def get_loss_object(best_row, loss_map=None):
    """Get loss object from the loss string in the configuration."""
    if loss_map is None:
        loss_map = LOSS_MAP
    
    loss_string_from_csv = best_row.get('loss', 'MAE()')
    final_loss_object = loss_map.get(loss_string_from_csv, MAE())
    
    return final_loss_object


def create_model_from_config(model_name, best_params, loss_object):
    """Create a model instance from the best configuration."""
    if model_name == 'AutoNHITS' or model_name == 'NHITS':
        return NHITS(loss=loss_object, **best_params)
    else:
        raise ValueError(f"Model {model_name} not supported yet. Add it to the create_model_from_config function.")


def load_best_model_config(model_name='AutoNHITS', csv_path=None):
    """Load the best model configuration for a specific model."""
    # Load hyperparameters from CSV
    loaded_best_configs_df = load_best_hyperparameters(csv_path)
    
    if loaded_best_configs_df is None:
        return None, None, None
    
    try:
        # Get the best configuration for the specified model
        best_row = loaded_best_configs_df[loaded_best_configs_df['model_name'] == model_name].iloc[0]
        print(f"\nBest {model_name} learning rate from CSV: {best_row['learning_rate']}")
        
        # Parse hyperparameters
        best_params = parse_hyperparameters(best_row)
        print(f"\n{model_name} best params as dict (for re-initialization):")
        print(best_params)
        
        # Get loss object
        final_loss_object = get_loss_object(best_row)
        
        # Create model instance
        model_instance = create_model_from_config(model_name, best_params, final_loss_object)
        print(f"\n{model_name} model initialized with best parameters: {model_instance}")
        
        return model_instance, best_params, final_loss_object
        
    except IndexError:
        print(f"Error: '{model_name}' not found in the loaded CSV. Check the model name in your best_hyperparameters.csv.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while creating model: {e}")
        return None, None, None


if __name__ == "__main__":
    # Test loading best model configuration
    model, params, loss = load_best_model_config('AutoNHITS')
    if model is not None:
        print("\nSuccessfully loaded model configuration for NHITS")
        print(f"Model: {model}")
        print(f"Loss: {loss}")
    else:
        print("Failed to load model configuration") 