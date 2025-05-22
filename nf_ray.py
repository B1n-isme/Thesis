import logging
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from utilsforecast.plotting import plot_series

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, BiTCN
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MQLoss, DistributionLoss
from utilsforecast.losses import mse, mae, rmse
from utilsforecast.evaluation import evaluate
from neuralforecast.utils import PredictionIntervals
from utilsforecast.plotting import plot_series

import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch

# For reproducibility
import torch
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

print(f"Pipeline execution started at: {datetime.now()} (Ho Chi Minh City Time)")

ray.init(num_cpus=os.cpu_count())

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
torch.set_float32_matmul_precision('high')

#### 1: Data Loading and Preparation
df = pd.read_parquet('data/final/dataset.parquet')
df = df.rename(columns={'Date': 'ds', 'btc_close': 'y'})
df['unique_id'] = 'Bitcoin'
df['ds'] = pd.to_datetime(df['ds'])
df.reset_index(drop=True, inplace=True)


all_cols = df.columns.tolist()
hist_exog_list = [col for col in all_cols if col not in ['ds', 'unique_id', 'y']]

horizon = 7
print(f"Forecast horizon (h) set to: {horizon} days")
levels = [80, 90]

# Split into development and final holdout test set
test_length = horizon * 2

if len(df) <= test_length:
    raise ValueError("Not enough data to create a test set of the desired length. Decrease test_length or get more data.")

df_development = df.iloc[:-test_length].copy()
df_final_holdout_test = df.iloc[-test_length:].copy()

print(f"\nTotal data shape: {df.shape}")
print(f"Development set shape: {df_development.shape}")
print(f"Final holdout test set shape: {df_final_holdout_test.shape}")
print(f"  Development set covers: {df_development['ds'].min()} to {df_development['ds'].max()}")
print(f"  Final holdout test set covers: {df_final_holdout_test['ds'].min()} to {df_final_holdout_test['ds'].max()}")



#### 2: Hyperarameter Tuning with AutoModels
nhits_default_config = AutoNHITS.get_default_config(h = horizon, backend="ray")      

config_nhits = {
       "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
       "max_steps": tune.choice([1000]),                                         # Number of SGD steps
       "input_size": tune.choice([5 * horizon]),                                 # input_size = multiplier * horizon
       "batch_size": tune.choice([7]),                                           # Number of series in windows
       "windows_batch_size": tune.choice([256]),                                 # Number of windows in batch
       "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),               # MaxPool's Kernelsize
       "n_freq_downsample": tune.choice([[168, 24, 1], [24, 12, 1], [1, 1, 1]]), # Interpolation expressivity ratios
       "activation": tune.choice(['ReLU']),                                      # Type of non-linear activation
       "n_blocks":  tune.choice([[1, 1, 1]]),                                    # Blocks per each 3 stacks
       "mlp_units":  tune.choice([[[512, 512], [512, 512], [512, 512]]]),        # 2 512-Layers per block for each stack
       "interpolation_mode": tune.choice(['linear']),                            # Type of multi-step interpolation
       "val_check_steps": tune.choice([100]),                                    # Compute validation every 100 epochs
       "random_seed": tune.randint(1, 10),
    }

# loss=MQLoss(level=levels), # probabilistic forecasting
# loss=DistributionLoss("Normal", level=[90]), # Uncertainty quantification

automodels = [
        # NHITS(h = horizon,
        #         loss=MAE(),
        #         max_steps=100,
        #         input_size = 5*horizon,
        #         hist_exog_list = hist_exog_list,
        #         scaler_type = 'standard'),
        AutoNHITS(h=1,
                        loss=MAE(),
                        config=config_nhits,
                        search_alg=HyperOptSearch(),
                        backend='ray',
                        num_samples=20)
]

nf_hpo = NeuralForecast(models=automodels, 
                    freq='D', 
                    local_scaler_type='standard')

print("\nStarting Hyperparameter Optimization with AutoNHITS...")
# Fit the AutoNHITS model. This performs HPO.
# AutoNHITS uses an internal validation split from df_development.
nf_hpo.fit(df_development)

# Get the best hyperparameters found by AutoNHITS
best_config = automodels.get_best_config()
print(f"\nBest hyperparameters found by AutoNHITS:\n{best_config}")

#### 3: Cross-Validation on Best Configuration
# Instantiate the normal NHITS model with the best configuration
# Ensure all necessary parameters from best_config are passed.
# If 'scaler_type' was tuned, it will be in best_config. If not, and you want standard window scaling:
if 'scaler_type' not in best_config:
    best_config['scaler_type'] = 'standard' # Add if not tuned but desired

# Ensure 'input_size' and other core params are in best_config from HPO
# If HPO didn't tune 'h', ensure it's correctly set
best_config['h'] = horizon # Ensure horizon is correctly set

model_for_cv = NHITS(**best_config)

# The NeuralForecast object for Cross-Validation
# Again, using local_scaler_type='standard' for global per-series scaling
nf_cv = NeuralForecast(
    models=[model_for_cv],
    freq='D',
    local_scaler_type='standard'
)

print("\nStarting Cross-Validation with the best configuration...")
# `test_size` in cross_validation refers to the validation horizon for each fold (should be `h`)
# `n_windows` is the number of CV folds. `step_size` controls overlap.
cv_results_df = nf_cv.cross_validation(
    df=df_development,
    n_windows=5,      # Number of CV folds
    step_size=horizon,      # How much to move the window for the next fold
    test_size=horizon       # Validation horizon for each fold (must be <= h of the model)
)

# Evaluate CV results
cv_model_name = model_for_cv.alias # Gets the model name (e.g., 'NHITS')
cv_actuals = cv_results_df['y']
cv_preds = cv_results_df[cv_model_name]

cv_mae = mae(cv_actuals, cv_preds)
cv_rmse = rmse(cv_actuals, cv_preds)

print(f"\nCross-Validation Results for {cv_model_name} with best config:")
print(f"  Mean MAE across folds: {cv_mae:.4f}")
print(f"  Mean RMSE across folds: {cv_rmse:.4f}")
# You can also group by 'cutoff' or 'unique_id' for more detailed analysis



#### 4: Final Model Training 
# Instantiate the final model with the best configuration
# (same as model_for_cv or re-instantiate for clarity)
final_model_instance = NHITS(**best_config)

nf_final_train = NeuralForecast(
    models=[final_model_instance],
    freq='D',
    local_scaler_type='standard' # Consistent global scaling
)

print("\nStarting Final Model Training on the entire development set...")
# Train on the full development set. val_size=0 ensures no further splitting here.
nf_final_train.fit(df_development, val_size=0)
print("Final model training complete.")


#### 5: Prediction on the final holdout test set
print("\nMaking predictions on the final holdout test set...")

# NeuralForecast's predict method can take df_final_holdout_test directly.
# It will use the historical part of each series in df_final_holdout_test
# to generate the initial input window, and then predict 'h' steps.
# Ensure df_final_holdout_test has enough history for input_size if predicting this way,
# OR, more commonly, predict 'h' steps from the end of df_development for each series.

# To predict for the exact timestamps in df_final_holdout_test,
# we can provide a future dataframe `futr_df`.
# For this to work cleanly, `futr_df` should contain unique_id and ds for the forecast horizon.
# Let's ensure we are forecasting for the periods covered by df_final_holdout_test.

# Option 1: Predict h steps from the end of the training data (df_development)
# This is the most straightforward if df_final_holdout_test immediately follows df_development.
predictions_on_test = nf_final_train.predict()

# Align predictions with the test set.
# This requires careful handling of unique_id and ds.
# The `predictions_on_test` will have 'ds' values h steps after the end of training for each series.
# We need to merge this with `df_final_holdout_test`.

# Option 2: A more robust way to ensure predictions align with specific future timestamps in the test set
# is to use the `futr_df` argument if df_final_holdout_test contains future exogenous regressors,
# or if you want to explicitly define the forecast timestamps.
# If no future regressors, `predict()` forecasts `h` steps from the end of the training data for each series.
# If `df_final_holdout_test` contains only the `y` values for the test period,
# we need to align based on `ds`.

# Let's refine prediction alignment:
# The `predict()` method forecasts `h` steps from the last timestamp in the training data for each `unique_id`.
# We need to ensure these forecasted `ds` values match our `df_final_holdout_test`.
# The `df_final_holdout_test` was created to be `test_length_per_series` long, and `h` is the forecast horizon.
# If `test_length_per_series` is exactly `h`, then `predict()` should align well.
# If `test_length_per_series` > `h`, `predict()` will give the first `h` steps into that period.

# For this example, assuming `test_length_per_series` was set up to align with forecasting `h` steps.
# If `predict()` output doesn't perfectly align or you need more control, consider `predict(futr_df=...)`.
# Let's merge based on 'unique_id' and 'ds'.
final_evaluation_df = pd.merge(
    df_final_holdout_test,
    predictions_on_test,
    on=['unique_id', 'ds'],
    how='left' # Use left to keep all test points; predictions might be shorter if h < test_length_per_series
)
final_evaluation_df.dropna(inplace=True) # If some predictions couldn't be made or aligned.

# The column name for predictions will be the model's alias (e.g., 'NHITS')
final_model_alias = final_model_instance.alias


#### 6: Evaluation & Final Results
if not final_evaluation_df.empty and final_model_alias in final_evaluation_df.columns:
    test_actuals = final_evaluation_df['y']
    test_preds = final_evaluation_df[final_model_alias]

    final_mae = mae(test_actuals, test_preds)
    final_rmse = rmse(test_actuals, test_preds)

    print(f"\nFinal Evaluation on Holdout Test Set for {final_model_alias}:")
    print(f"  Test MAE: {final_mae:.4f}")
    print(f"  Test RMSE: {final_rmse:.4f}")
else:
    print("\nCould not perform final evaluation. Check prediction alignment or prediction column name.")

print(f"\nPipeline execution finished at: {datetime.now()} (Ho Chi Minh City Time)")


# Key Considerations and Best Practices:

# Data Leakage: The primary defense here is the initial split into df_development and df_final_holdout_test. All scalers (internal to NeuralForecast) and HPO procedures learn only from df_development or its internal folds.
# Scaling:
# local_scaler_type='standard' in NeuralForecast(...) applies global standard scaling per series.
# Models like NHITS can also have their own scaler_type (e.g., NHITS(scaler_type='standard', ...)), which applies window-level scaling. AutoNHITS will likely tune this internal model scaler_type. Ensure the final best_config reflects this.
# AutoModel Computational Cost: AutoNHITS (and similar) can be computationally intensive due to many HPO trials. Adjust num_samples based on your resources.
# cross_validation Parameters: Carefully choose n_windows, step_size, and test_size (which should be h) for meaningful CV results.
# Prediction Alignment: Ensuring your predictions correctly align with your test set's timestamps (ds values) is crucial for accurate evaluation. Using futr_df in predict offers more explicit control if needed.
# Frequency (freq): Ensure the freq parameter in NeuralForecast(...) matches your data's actual time series frequency (e.g., 'D' for daily, 'H' for hourly, 'M' for monthly).
# Exogenous Variables: If you have exogenous variables, you'll need to include them in df_development, df_final_holdout_test, and handle them correctly in predict (often via futr_df). This pipeline focuses on univariate forecasting.





















# set val_size & test_size
# Y_hat_df = nf.cross_validation(df, n_windows=4, step_size=horizon//2, verbose=1, refit=True) 

# nf.fit(df=train_df, val_size=horizon*2, prediction_intervals=PredictionIntervals())

# Y_hat_insample = nf.predict_insample(step_size=horizon)

# results = nf.models[1].results.trials_dataframe()
# results.drop(columns='user_attrs_ALL_PARAMS')

# evaluation_df = evaluate(Y_hat_df.drop(columns='cutoff'), [mse, mae, rmse])
# evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
# print(evaluation_df.head())

# summary_df = evaluation_df.groupby(['metric', 'best_model']).size().sort_values().to_frame()
# summary_df = summary_df.reset_index()
# summary_df.columns = ['metric', 'model', 'nr. of unique_ids']
# print(summary_df)

# Y_hat_df1 = nf.predict(df= test_df, level = [90]) # generate conformal intervals
# Y_hat_df1 = Y_hat_df1.reset_index()
# print(Y_hat_df1.head())

# nf.save(path='./checkpoints/test_run/',
#         model_index=None, # None for all models
#         overwrite=True,
#         save_dataset=True)

# nf2 = NeuralForecast.load(path='./checkpoints/test_run/')
# Y_hat_df2 = nf2.predict()
# Y_hat_df2.head()