import os
import torch
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from src.data.lightningdatamodule import WindowedTimeSeriesDataModule
import optuna
from torch.utils.data import DataLoader, TensorDataset
from src.hpo.hpo import objective
from models.LitTabular import LitTabularForecaster
from src.data.process import run_data_processing

# Configuration
LOOKBACK_WINDOW = 7
PREDICTION_HORIZON = 7
STRIDE = 2
N_CV_SPLITS = 3
HPO_N_TRIALS = 15
MODEL_MAX_EPOCHS = 10
HPO_MAX_EPOCHS = 5

def main():
    L.seed_everything(42)
    
    # Data Processing
    data_artifacts = run_data_processing(
        lookback_window=LOOKBACK_WINDOW,
        prediction_horizon=PREDICTION_HORIZON,
        stride=STRIDE,
        n_cv_splits_for_tscv=N_CV_SPLITS
    )
    
    windowed_cv_folds = data_artifacts.get('windowed_cv_folds', [])
    if not windowed_cv_folds:
        raise ValueError("No windowed data available")

    n_features = windowed_cv_folds[0]['X_train_w'].shape[2]
    
    # Hyperparameter Optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(
        trial, windowed_cv_folds, LOOKBACK_WINDOW, n_features, HPO_MAX_EPOCHS
    ), n_trials=HPO_N_TRIALS)

    # Final Model Training
    best_params = study.best_params

    # Extract trainval data
    X_trainval = data_artifacts['windowed_trainval_data']['X_w']
    y_trainval = data_artifacts['windowed_trainval_data']['y_w']
    
    # Create data module
    datamodule = WindowedTimeSeriesDataModule(
        train_data=(X_trainval, y_trainval),
        test_data=(
            data_artifacts['windowed_test_data']['X_w'],
            data_artifacts['windowed_test_data']['y_w']
        ),
        batch_size=best_params.get('batch_size', 64),
        num_workers=os.cpu_count(),
        lookback_window=LOOKBACK_WINDOW,
        n_features=n_features
    )
    
    final_model = LitTabularForecaster(
        input_size=LOOKBACK_WINDOW * n_features,
        hidden_size=best_params['hidden_size'],
        learning_rate=best_params['learning_rate']
    )
    
    trainer = L.Trainer(
        max_epochs=MODEL_MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=True,
        enable_progress_bar=True,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
    )
    trainer.fit(final_model, datamodule=datamodule)

    trainer.test(final_model, datamodule=datamodule)

if __name__ == "__main__":
    main()