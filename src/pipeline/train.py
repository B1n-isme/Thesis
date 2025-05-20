import os
import lightning as L
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from src.data.process import run_data_processing
from src.models.lit_tabular_forecaster import LitTabularForecaster, TabularWindowedDataModule
from src.pipeline.objective import objective
from src.pipeline.feature_selection import select_features_and_transform
from src.models.architectures import MLP, LSTMModel
import numpy as np

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

LOOKBACK_WINDOW = 7
PREDICTION_HORIZON = 7
STRIDE = 2
N_CV_SPLITS = 3
HPO_N_TRIALS = 15
MODEL_MAX_EPOCHS = 10
HPO_MAX_EPOCHS_PER_TRIAL = 5

FEATURE_SELECTION_METHOD = 'rf'
N_ESTIMATORS_RF = 100
MAX_FEATURES_TO_SELECT = None
RANDOM_STATE = 42
PERFORM_FEATURE_SELECTION = True  # Set to False to disable feature selection

def main():
    L.seed_everything(42, workers=True)
    print("Starting data processing...")
    data_artifacts = run_data_processing(
        lookback_window=LOOKBACK_WINDOW,
        prediction_horizon=PREDICTION_HORIZON,
        stride=STRIDE,
        n_cv_splits_for_tscv=N_CV_SPLITS
    )
    print("Data processing complete.")

    windowed_cv_folds = data_artifacts.get('windowed_cv_folds')
    if not windowed_cv_folds:
        print("No windowed CV folds found in data_artifacts. Ensure 'lookback_window' was set.")
        print("Cannot proceed with HPO without windowed data for LitTabularForecaster. Exiting.")
        return

    if not windowed_cv_folds or windowed_cv_folds[0]['X_train_w'].size == 0:
        print("Windowed CV folds are empty or the first fold contains no training data.")
        print("Cannot determine n_features. Exiting HPO.")
        return
    n_features = windowed_cv_folds[0]['X_train_w'].shape[2]
    print(f"Determined n_features: {n_features} from windowed data.")

    processed_feature_names = data_artifacts.get('processed_feature_names')
    windowed_trainval = data_artifacts.get('windowed_trainval_data')
    windowed_test = data_artifacts.get('windowed_test_data')
    lookback_used = data_artifacts.get('lookback_window_used', LOOKBACK_WINDOW)
    if not (windowed_trainval and windowed_test and processed_feature_names):
        print("Error: Required data for feature selection not found.")
        return
    X_train_w = windowed_trainval.get('X_w')
    y_train_w = windowed_trainval.get('y_w')
    X_test_w = windowed_test.get('X_w')
    if X_train_w is None or y_train_w is None or X_test_w is None:
        print("Error: X_w or y_w missing from windowed data dictionaries.")
        return

    if PERFORM_FEATURE_SELECTION:
        print("\nStarting feature selection...")
        _, _, selected_names, selector_model = select_features_and_transform(
            X_train_w=X_train_w,
            y_train_w=y_train_w,
            X_test_w=X_test_w,
            original_processed_feature_names=processed_feature_names,
            lookback_window=lookback_used,
            method=FEATURE_SELECTION_METHOD,
            n_estimators_rf=N_ESTIMATORS_RF,
            max_features_to_select=MAX_FEATURES_TO_SELECT,
            random_state=RANDOM_STATE
        )
        print(f"Selected features: {selected_names}")
        selected_indices = [processed_feature_names.index(name) for name in selected_names]
        windowed_trainval['X_w'] = X_train_w[:, :, selected_indices]
        windowed_test['X_w'] = X_test_w[:, :, selected_indices]
        for fold in windowed_cv_folds:
            fold['X_train_w'] = fold['X_train_w'][:, :, selected_indices]
            fold['X_val_w'] = fold['X_val_w'][:, :, selected_indices]
        n_selected_features = len(selected_names)
        print(f"Feature selection enabled. Using {n_selected_features} features.")
    else:
        print("\nFeature selection is DISABLED. Using all features.")
        n_selected_features = n_features
        # No filtering, use all features as in original windowed data

    print(f"\nStarting Optuna hyperparameter tuning for LitTabularForecaster...")
    print(f"Number of CV folds to use: {len(windowed_cv_folds)}")
    print(f"Lookback window: {LOOKBACK_WINDOW}, Prediction horizon: {PREDICTION_HORIZON}, Stride: {STRIDE}")

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial_obj: objective(
            trial_obj,
            windowed_cv_folds,
            LOOKBACK_WINDOW,
            n_selected_features,
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

    print("\nTraining final model with best hyperparameters on selected features...")
    # After feature selection and before datamodule creation
    # Split windowed_trainval['X_w'] and y_train_w into train/val sets (e.g., 80/20 split)
    X_all = windowed_trainval['X_w']
    y_all = y_train_w
    n_samples = X_all.shape[0]
    val_ratio = 0.2
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val
    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]

    batch_size = best_hparams.get('batch_size', 64)
    datamodule = TabularWindowedDataModule(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        lookback_window=LOOKBACK_WINDOW,
        n_features=n_selected_features,
        batch_size=batch_size,
        num_workers=os.cpu_count()
    )
    datamodule.setup()

    final_input_size = LOOKBACK_WINDOW * n_selected_features

    # Create model based on best hyperparameters
    model = MLP(
        input_size=final_input_size,
        hidden_size=best_hparams['hidden_size'],
        output_size=1
    )
    final_model = LitTabularForecaster(
        model=model,
        learning_rate=best_hparams['learning_rate']
    )

    final_trainer = L.Trainer(
        max_epochs=MODEL_MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=True,
        enable_progress_bar=True,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min")]
    )
    print(f"Fitting final model with input_size={final_input_size}, hidden_size={best_hparams['hidden_size']}, lr={best_hparams['learning_rate']}...")
    final_trainer.fit(final_model, datamodule=datamodule)
    print("Final model training complete.")

    if windowed_test['X_w'] is not None and windowed_test['X_w'].shape[0] > 0:
        y_test_w = windowed_test.get('y_w')
        test_datamodule = TabularWindowedDataModule(
            X_train=None, y_train=None,
            X_val=None, y_val=None,
            X_test=windowed_test['X_w'], y_test=y_test_w,
            lookback_window=LOOKBACK_WINDOW,
            n_features=n_selected_features,
            batch_size=batch_size,
            num_workers=os.cpu_count()
        )
        test_datamodule.setup()
        print("Evaluating final model on the test set...")
        test_results = final_trainer.test(final_model, datamodule=test_datamodule)
        print("Test results:", test_results)
    else:
        print("Windowed test data not available or empty. Skipping final evaluation.")

if __name__ == "__main__":
    main() 