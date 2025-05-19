import optuna
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from src.models import LitTabularForecaster

def objective(trial: optuna.Trial, windowed_cv_folds: list, lookback_window: int, n_features: int, hpo_max_epochs: int) -> float:
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    input_size = lookback_window * n_features
    fold_val_mses = []

    if not windowed_cv_folds:
        return float('inf')

    for i, fold_data in enumerate(windowed_cv_folds):
        try:
            # Data preparation
            X_train_flat = fold_data['X_train_w'].reshape(fold_data['X_train_w'].shape[0], -1)
            X_val_flat = fold_data['X_val_w'].reshape(fold_data['X_val_w'].shape[0], -1)
            
            train_dataset = TensorDataset(torch.tensor(X_train_flat), torch.tensor(fold_data['y_train_w']))
            val_dataset = TensorDataset(torch.tensor(X_val_flat), torch.tensor(fold_data['y_val_w']))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Model & Trainer
            model = LitTabularForecaster(input_size, hidden_size, learning_rate=learning_rate)
            early_stop = EarlyStopping(monitor="val_loss", patience=3)
            checkpoint = ModelCheckpoint(monitor="val_loss")
            
            trainer = L.Trainer(
                max_epochs=hpo_max_epochs,
                accelerator="auto",
                callbacks=[early_stop, checkpoint],
                logger=False,
                enable_progress_bar=False
            )
            
            trainer.fit(model, train_loader, val_loader)
            fold_val_mses.append(trainer.callback_metrics["val_loss"].item())
            
        except Exception as e:
            print(f"Error in fold {i+1}: {str(e)}")
            fold_val_mses.append(float('inf'))

    return np.mean(fold_val_mses)