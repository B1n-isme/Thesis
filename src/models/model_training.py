"""
Model training module for neural forecasting models.
"""
from neuralforecast import NeuralForecast
from src.config.forecasting_config import FREQUENCY, LOCAL_SCALER_TYPE
from src.models.model_loader import load_best_model_config


def train_final_model(model_instance, df_development, val_size=0):
    """Train the final model on the entire development set."""
    print("\nStarting Final Model Training on the entire development set...")
    
    # Create NeuralForecast instance for final training
    nf_final_train = NeuralForecast(
        models=[model_instance],
        freq=FREQUENCY,
        local_scaler_type=LOCAL_SCALER_TYPE
    )
    
    # Train on the full development set
    # val_size=0 ensures no further splitting here
    nf_final_train.fit(df_development, val_size=val_size)
    
    print("Final model training complete.")
    return nf_final_train


def create_and_train_final_model(df_development, model_name='AutoNHITS', csv_path=None):
    """Create model from best config and train it on development data."""
    # Load best model configuration
    model_instance, best_params, final_loss_object = load_best_model_config(model_name, csv_path)
    
    if model_instance is None:
        print("Failed to load model configuration. Aborting training.")
        return None, None, None, None
    
    # Train the final model
    nf_final_train = train_final_model(model_instance, df_development)
    
    return nf_final_train, model_instance, best_params, final_loss_object


if __name__ == "__main__":
    from src.data.data_preparation import prepare_forecasting_data
    
    # Prepare data
    _, df_development, _, _ = prepare_forecasting_data()
    
    # Create and train final model
    nf_final, model, params, loss = create_and_train_final_model(df_development)
    
    if nf_final is not None:
        print(f"\nFinal model training completed successfully!")
        print(f"Model: {model}")
        print(f"Loss function: {loss}")
    else:
        print("Final model training failed.") 