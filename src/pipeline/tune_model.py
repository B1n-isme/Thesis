import os
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint # For faster HPO trials
from typing import List, Dict, Any, Tuple

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
from src.data.process import run_data_processing

# --- Define a Simple LightningModule for Tabular/Windowed Data ---
class LitTabularForecaster(L.LightningModule):
    """
    A simple PyTorch Lightning model for tabular/windowed time series forecasting.
    Uses a Multi-Layer Perceptron (MLP) architecture.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters() # Saves input_size, hidden_size, output_size, learning_rate
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _common_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        # x is already flattened by the DataLoader preparation
        y_hat = self(x)
        if y_hat.ndim == 2 and y_hat.shape[1] == 1: # Ensure y_hat is (batch_size,)
            y_hat = y_hat.squeeze(1)
        if y.ndim == 1 and y_hat.ndim == 0: # Handle batch_size = 1 case for y_hat
             y_hat = y_hat.unsqueeze(0)
        loss = F.mse_loss(y_hat, y)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, logger=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def objective(trial: optuna.Trial,
              windowed_cv_folds: List[Dict[str, Any]],
              current_lookback_window: int, # Renamed from lookback_window to avoid confusion with global
              n_features: int,
              hpo_max_epochs: int = 10 # Max epochs for HPO trials
              ) -> float:
    """
    Optuna objective function for hyperparameter tuning of LitTabularForecaster.

    Args:
        trial (optuna.Trial): Optuna trial object.
        windowed_cv_folds (List[Dict[str, Any]]): List of windowed CV fold data.
        current_lookback_window (int): Lookback window size.
        n_features (int): Number of features in the input data.
        hpo_max_epochs (int): Max epochs to train each model during HPO.

    Returns:
        float: Average validation MSE across CV folds.
    """
    # Suggest hyperparameters for LitTabularForecaster
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    # You could add more (e.g., num_layers, dropout) by modifying LitTabularForecaster

    input_size = current_lookback_window * n_features
    fold_val_mses: List[float] = []

    if not windowed_cv_folds:
        print("Objective function received no windowed_cv_folds. Returning infinity.")
        return float('inf')

    for i, fold_data in enumerate(windowed_cv_folds):
        X_train_w = fold_data['X_train_w']
        y_train_w = fold_data['y_train_w']
        X_val_w = fold_data['X_val_w']
        y_val_w = fold_data['y_val_w']

        if X_train_w.size == 0 or X_val_w.size == 0:
            print(f"Skipping fold {i+1} in trial {trial.number} due to empty windowed data.")
            continue

        # Reshape X data: (n_samples, lookback_window, n_features) -> (n_samples, lookback_window * n_features)
        X_train_flat = X_train_w.reshape(X_train_w.shape[0], -1)
        X_val_flat = X_val_w.reshape(X_val_w.shape[0], -1)

        # Convert to Tensors
        X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_w, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_flat, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_w, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Use a smaller batch size for HPO if datasets are small
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128]) 

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=os.cpu_count())

        model = LitTabularForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate
        )

        # Minimal trainer for HPO
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=False, mode="min")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # or "train_loss" if you don't have validation
            save_top_k=1,
            mode="min"
        )
        trainer = L.Trainer(
            max_epochs=hpo_max_epochs,
            accelerator="auto",
            devices="auto",
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=False, # Disable TensorBoard logging for HPO trials
            enable_progress_bar=True,
            enable_model_summary=False,
            deterministic=True # For Optuna consistency, can slow down
        )

        try:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            # Get the best validation loss for this fold from the trainer or model
            # Ensure 'val_loss' is logged by validation_step
            current_fold_val_mse = trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()
            if current_fold_val_mse == float('inf') and model.current_epoch > 0:
                 # Sometimes callback_metrics might not have it if training ended abruptly without validation
                 # Try to get it directly from a logged value if available or last checkpoint
                 print(f"Warning: val_loss not in callback_metrics for fold {i+1}, trial {trial.number}. Check training.")

            fold_val_mses.append(current_fold_val_mse)
        except Exception as e:
            print(f"Error during training/evaluation for fold {i+1}, trial {trial.number}: {e}")
            # Penalize this trial significantly
            fold_val_mses.append(float('inf')) # Or return float('inf') immediately to prune
            # return float('inf') # Prune trial if one fold fails badly
            break # Stop processing more folds for this trial if one fails

    if not fold_val_mses:
        print(f"No folds were successfully processed for trial {trial.number}. Returning high error.")
        return float('inf')

    average_mse = np.mean(fold_val_mses)
    print(f"Trial {trial.number}: LR={learning_rate:.6f}, Hidden={hidden_size}, Batch={batch_size} Avg MSE={average_mse:.6f}")
    return average_mse

# Global fixed settings for the main HPO execution
LOOKBACK_WINDOW = 7
PREDICTION_HORIZON = 7
STRIDE = 2
N_CV_SPLITS = 3
HPO_N_TRIALS = 15        # Reduced for quicker execution example
MODEL_MAX_EPOCHS = 10    # For final model training
HPO_MAX_EPOCHS_PER_TRIAL = 5 # For HPO trials

def main():
    """
    Main function to run data processing and Optuna HPO with PyTorch Lightning.
    """
    L.seed_everything(42, workers=True) # For reproducibility

    print("Starting data processing...")
    # --- Parameters for data processing ---
    # **Crucial**: Define lookback_window for windowed data needed by LitTabularForecaster
    # LOOKBACK_WINDOW = trial.suggest_int('lookback_window', 5, 30) # Example range, can be fixed for model architecture
    # PREDICTION_HORIZON = 1 # Assuming we predict 1 step ahead
    # N_CV_SPLITS = 3 # Keep CV splits manageable for HPO
    # HPO_N_TRIALS = 25 # Number of Optuna trials
    # MODEL_MAX_EPOCHS = 15 # Max epochs for final model training after HPO
    # HPO_MAX_EPOCHS_PER_TRIAL = 7 # Max epochs for each HPO trial to speed things up

    data_artifacts = run_data_processing(
        lookback_window=LOOKBACK_WINDOW,
        prediction_horizon=PREDICTION_HORIZON,
        stride=STRIDE,
        n_cv_splits_for_tscv=N_CV_SPLITS
        # Add other params for run_data_processing if needed (data_path, etc.)
    )
    print("Data processing complete.")

    windowed_cv_folds = data_artifacts.get('windowed_cv_folds')
    if not windowed_cv_folds:
        print("No windowed CV folds found in data_artifacts. Ensure 'lookback_window' was set.")
        print("Cannot proceed with HPO without windowed data for LitTabularForecaster. Exiting.")
        return

    # Determine n_features from the data (assuming all folds have same feature count)
    # Need to handle case where windowed_cv_folds might be empty or first fold has no data
    if not windowed_cv_folds or windowed_cv_folds[0]['X_train_w'].size == 0:
        print("Windowed CV folds are empty or the first fold contains no training data.")
        print("Cannot determine n_features. Exiting HPO.")
        return 
    # Shape of X_train_w is (n_samples, lookback_window, n_features)
    n_features = windowed_cv_folds[0]['X_train_w'].shape[2]
    print(f"Determined n_features: {n_features} from windowed data.")

    print(f"\nStarting Optuna hyperparameter tuning for LitTabularForecaster...")
    print(f"Number of CV folds to use: {len(windowed_cv_folds)}")
    print(f"Lookback window: {LOOKBACK_WINDOW}, Prediction horizon: {PREDICTION_HORIZON}, Stride: {STRIDE}")

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial_obj: objective(
            trial_obj, 
            windowed_cv_folds, 
            LOOKBACK_WINDOW, # Pass the fixed lookback
            n_features, 
            hpo_max_epochs=HPO_MAX_EPOCHS_PER_TRIAL
        ),
        n_trials=HPO_N_TRIALS
    )

    print("\nHyperparameter tuning complete.")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best MSE: {study.best_value:.6f}")
    print("Best hyperparameters:")
    best_hparams = study.best_params
    for key, value in best_hparams.items():
        print(f"  {key}: {value}")

    # --- Train final model with best hyperparameters ---
    print("\nTraining final model with best hyperparameters on full X_trainval_final_for_model...")
    
    final_lookback = best_hparams.get('lookback_window', LOOKBACK_WINDOW) # Use HPO'd or fixed
    # Re-run data processing if lookback was part of HPO and you want to use the *best* lookback
    # For simplicity, if LOOKBACK_WINDOW was fixed above, we can use existing data_artifacts
    # If lookback was tuned, you might need: 
    # data_artifacts_final = run_data_processing(lookback_window=final_lookback, ...)
    # X_trainval_final_w = data_artifacts_final['windowed_trainval_data']['X_w'] 
    # y_trainval_final_w = data_artifacts_final['windowed_trainval_data']['y_w']
    # X_test_final_w = data_artifacts_final['windowed_test_data']['X_w']
    # y_test_final_w = data_artifacts_final['windowed_test_data']['y_w']

    if 'windowed_trainval_data' not in data_artifacts or not data_artifacts['windowed_trainval_data']['X_w'].size > 0:
        print("Final windowed training data not available or empty. Skipping final model training.")
        return

    X_trainval_w = data_artifacts['windowed_trainval_data']['X_w']
    y_trainval_w = data_artifacts['windowed_trainval_data']['y_w']

    X_trainval_flat = X_trainval_w.reshape(X_trainval_w.shape[0], -1)
    X_trainval_tensor = torch.tensor(X_trainval_flat, dtype=torch.float32)
    y_trainval_tensor = torch.tensor(y_trainval_w, dtype=torch.float32)
    
    final_train_dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor)
    final_train_loader = DataLoader(final_train_dataset, batch_size=best_hparams.get('batch_size', 64), shuffle=True, num_workers=0)

    final_input_size = final_lookback * n_features # n_features should be consistent
    final_model = LitTabularForecaster(
        input_size=final_input_size,
        hidden_size=best_hparams['hidden_size'],
        learning_rate=best_hparams['learning_rate']
    )

    # Configure a more complete trainer for the final model
    # You can add ModelCheckpoint and TensorBoardLogger here if desired
    final_trainer = L.Trainer(
        max_epochs=MODEL_MAX_EPOCHS, 
        accelerator="auto",
        devices="auto",
        logger=True, # Enable default logger (e.g. TensorBoard) for final training
        enable_progress_bar=True,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)] # Use val_loss if you have a val split for final train
        # If no validation set for final training, remove val_loss monitoring or monitor train_loss
        # For true final training, you'd typically not use a val_loader with trainer.fit
        # Or you'd split X_trainval_final_for_model into train/val for this final fit step.
        # For simplicity here, let's assume we're fitting on all of it and monitoring train_loss
        # To monitor train_loss in EarlyStopping, it must be logged by training_step
    )
    
    print(f"Fitting final model with input_size={final_input_size}, hidden_size={best_hparams['hidden_size']}, lr={best_hparams['learning_rate']}...")
    # Note: For final training, you usually fit on all available training data.
    # If you want validation during this final fit, you need a validation dataloader.
    # For now, fitting without a validation loader. Lightning will use train_loss if val_loader is None.
    # If you want to use EarlyStopping on val_loss, provide a val_loader for the final fit.
    # This example will use train_loss for EarlyStopping if no val_loader is passed.
    # For a robust final training, you might re-split 'X_trainval_final_for_model' from 'data_artifacts'
    # into a new train/val set for this stage if you want proper early stopping on validation data.
    
    # Simpler: Fit on all trainval_w, monitor train_loss (requires train_loss in EarlyStopping)
    early_stop_final_train = EarlyStopping(monitor="train_loss", patience=5, verbose=True, mode="min")
    final_trainer_train_loss_monitor = L.Trainer(
        max_epochs=MODEL_MAX_EPOCHS, accelerator="auto", devices="auto", logger=True,
        callbacks=[early_stop_final_train], enable_progress_bar=True
    )
    final_trainer_train_loss_monitor.fit(final_model, train_dataloaders=final_train_loader)
    print("Final model training complete.")

    # Evaluate on the test set
    if 'windowed_test_data' in data_artifacts and data_artifacts['windowed_test_data']['X_w'].size > 0:
        X_test_w = data_artifacts['windowed_test_data']['X_w']
        y_test_w = data_artifacts['windowed_test_data']['y_w']
        
        X_test_flat = X_test_w.reshape(X_test_w.shape[0], -1)
        X_test_tensor = torch.tensor(X_test_flat, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_w, dtype=torch.float32)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        # Use batch_size from HPO or a default
        test_loader = DataLoader(test_dataset, batch_size=best_hparams.get('batch_size', 64), num_workers=os.cpu_count())

        print("Evaluating final model on the test set...")
        test_results = final_trainer_train_loss_monitor.test(final_model, dataloaders=test_loader)
        print("Test results:", test_results)
    else:
        print("Windowed test data not available or empty. Skipping final evaluation.")

if __name__ == "__main__":
    main() 