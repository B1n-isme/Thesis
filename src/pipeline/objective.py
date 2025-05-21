import os
import optuna
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from typing import List, Dict, Any
from src.models.lit_tabular_forecaster import LitTabularForecaster, TabularWindowedDataModule
from src.models.architectures import MLP, LSTMModel

def objective(trial: optuna.Trial,
              windowed_cv_folds: List[Dict[str, Any]],
              current_lookback_window: int,
              n_features: int,
              hpo_max_epochs: int = 10
              ) -> float:
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
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

        # Create model based on best hyperparameters
        # model = MLP(
        #     seq_len=current_lookback_window,
        #     n_features=n_features,
        #     hidden_size=hidden_size,
        #     pred_len=y_train_w.shape[1],
        #     output_size=1
        # )
        model = LSTMModel(
            input_size=n_features,
            hidden_size=hidden_size,
            pred_len=y_train_w.shape[1],
            output_size=1,
            num_layers=1,
            dropout=0.0
        )
        final_model = LitTabularForecaster(
            model=model,
            learning_rate=learning_rate
        )

        datamodule = TabularWindowedDataModule(
            X_train=X_train_w, y_train=y_train_w,
            X_val=X_val_w, y_val=y_val_w,
            n_features=n_features,
            batch_size=batch_size,
            num_workers=os.cpu_count()
        )
        datamodule.setup()

        early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=False, mode="min")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min"
        )
        
        trainer = L.Trainer(
            max_epochs=hpo_max_epochs,
            accelerator="auto",
            devices="auto",
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            deterministic=True
        )

        try:
            trainer.fit(final_model, datamodule=datamodule)
            current_fold_val_mse = trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()
            if current_fold_val_mse == float('inf') and model.current_epoch > 0:
                print(f"Warning: val_loss not in callback_metrics for fold {i+1}, trial {trial.number}. Check training.")
            fold_val_mses.append(current_fold_val_mse)
        except Exception as e:
            print(f"Error during training/evaluation for fold {i+1}, trial {trial.number}: {e}")
            fold_val_mses.append(float('inf'))
            break

    if not fold_val_mses:
        print(f"No folds were successfully processed for trial {trial.number}. Returning high error.")
        return float('inf')

    average_mse = np.mean(fold_val_mses)
    print(f"Trial {trial.number}: LR={learning_rate:.6f}, Hidden={hidden_size}, Batch={batch_size} Avg MSE={average_mse:.6f}")
    return average_mse 