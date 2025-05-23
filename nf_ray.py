import logging
import os
import json
import torch
import pandas as pd
import polars as pl
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


from model_definition import get_auto_models


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


print(f"Pipeline execution started at: {datetime.now()} (Ho Chi Minh City Time)")


ray.init(
    address='local',
    num_cpus=os.cpu_count(),
    num_gpus=torch.cuda.device_count()
)


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


# Define a mapping from string names (how they might appear in the CSV) to actual loss objects
LOSS_MAP = {
    'MAE': MAE(),
    'MAE()': MAE(),
    'MQLoss': MQLoss(),
    'MQLoss()': MQLoss(),
    'RMSE': RMSE(),
    'RMSE()': RMSE(),
    # Add other losses if you use them and their string representation
}




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




# #### 2: Hyperarameter Tuning with AutoModels


# # loss=MQLoss(level=levels), # probabilistic forecasting
# # loss=DistributionLoss("Normal", level=[90]), # Uncertainty quantification


# # Pass the required parameters to the function
# automodels = get_auto_models(
#     horizon=horizon,
#     loss_fn=MAE(),
#     num_samples_per_model=1 # Tune each model with 5 trials
# )


# nf_hpo = NeuralForecast(models=automodels,
#                     freq='D',
#                     local_scaler_type='standard')


# print("\nStarting Hyperparameter Optimization with AutoNHITS...")
# # Fit the AutoNHITS model. This performs HPO.
# # AutoNHITS uses an internal validation split from df_development.
# nf_hpo.fit(df_development)


# # Initialize a list to store best configs for all models
# all_best_configs = []


# for model in nf_hpo.models:
#     # Check if the model is an Auto model and has results
#     if hasattr(model, 'results') and model.results is not None:
#         model_name = model.__class__.__name__
#         print(f"Processing results for {model_name}...")


#         # Get the DataFrame of all trials for this model
#         results_df = model.results.get_dataframe()


#         if not results_df.empty:
#             # Find the row with the lowest 'valid_loss'
#             # Assuming 'valid_loss' is the metric to minimize
#             best_run = results_df.loc[results_df['loss'].idxmin()]


#             # Extract the 'config/' columns to get the hyperparameters
#             best_params = {
#                 col.replace('config/', ''): best_run[col]
#                 for col in results_df.columns if col.startswith('config/')
#             }


#             # Add model name and best loss to the dictionary
#             best_params['model_name'] = model_name
#             best_params['best_valid_loss'] = best_run['loss']
#             best_params['training_iteration'] = best_run['training_iteration'] # Useful for understanding convergence


#             # Append to the list
#             all_best_configs.append(best_params)
#             print(f"Best config for {model_name}: {best_params}")
#         else:
#             print(f"No tuning results found for {model_name}.")
#     else:
#         print(f"Model {model.__class__.__name__} is not an Auto model or has no results.")


# # Convert the list of best configs into a single DataFrame
# if all_best_configs:
#     best_configs_df = pd.DataFrame(all_best_configs)


#     # Define the output CSV path
#     output_dir = 'tuning_results'
#     os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
#     csv_filename = os.path.join(output_dir, 'best_hyperparameters.csv')


#     # Save to CSV
#     best_configs_df.to_csv(csv_filename, index=False)
#     print(f"\nBest hyperparameters saved to {csv_filename}")
#     print("\nContent of best_hyperparameters.csv:")
#     print(best_configs_df)
# else:
#     print("No best configurations were found for any model.")






#### 3: Cross-Validation on Best Configuration
# Load the best hyperparameters from the CSV ---
csv_filename = 'tuning_results/best_hyperparameters.csv' # Adjust path if needed


try:
    loaded_best_configs_df = pd.read_csv(csv_filename)
    print(f"\nLoaded best configs from {csv_filename}:")
    print(loaded_best_configs_df)


    # Example: Get the best N-HiTS config
    nhits_best_row = loaded_best_configs_df[loaded_best_configs_df['model_name'] == 'AutoNHITS'].iloc[0]
    print(f"\nBest N-HiTS learning rate from CSV: {nhits_best_row['learning_rate']}")


    # --- 3. Convert the row of best parameters into a dictionary ---
    # Exclude metadata columns not needed for model initialization
    # These are specific to our best_hyperparameters.csv structure
    # 'loss' is also excluded here because we'll handle it separately via LOSS_MAP
    exclude_keys = ['model_name', 'loss', 'valid_loss', 'best_valid_loss', 'training_iteration']
    nhits_best_params = {}
    for key, value in nhits_best_row.to_dict().items():
        if key not in exclude_keys:
            # Attempt to parse values that are expected to be lists/nested lists
            if key in ['n_pool_kernel_size', 'n_freq_downsample', 'n_blocks', 'mlp_units']:
                try:
                    nhits_best_params[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    print(f"Warning: Could not parse '{key}' value '{value}' as JSON. Keeping as original.")
                    nhits_best_params[key] = value
            else:
                nhits_best_params[key] = value


    print("\nN-HiTS best params as dict (for re-initialization):")
    print(nhits_best_params)


    # --- 4. Special Handling for 'loss' parameter ---
    # We explicitly retrieve the loss string from the row and map it to the object
    loss_string_from_csv = nhits_best_row.get('loss', 'MAE()') # Default to MAE() if 'loss' column isn't found
    final_loss_object = LOSS_MAP.get(loss_string_from_csv, MAE()) # Map string to object, default to MAE()


    # --- 5. Initialize the NHITS model with the best parameters ---
    # Pass 'h' and 'loss' explicitly, then unpack the rest of the parameters
    model_for_cv = NHITS(loss=final_loss_object, **nhits_best_params)


    print(f"\nNHITS model initialized with best parameters: {model_for_cv}")


except FileNotFoundError:
    print(f"Error: {csv_filename} not found. Please ensure the tuning script ran and saved the CSV.")
except IndexError:
    print("Error: 'AutoNHITS' not found in the loaded CSV. Check the model name in your `best_hyperparameters.csv`.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")




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
)


# Convert your pandas DataFrame to a Polars DataFrame
df_pl = pl.from_pandas(cv_results_df)


# Define columns to exclude
exclude_cols = ['unique_id', 'ds', 'cutoff', 'y']


# Get the model columns dynamically
model_cols = [col for col in df_pl.columns if col not in exclude_cols]


# Loop through each model column
for model in model_cols:
    # Calculate MSE and MAE using utilsforecast
    mse_val = mse(df=df_pl, models=[model], target_col='y').to_pandas()[model].values[0]
    mae_val = mae(df=df_pl, models=[model], target_col='y').to_pandas()[model].values[0]
    rmse_val = np.sqrt(mse_val)

    print(f"\nEvaluation for model: {model}")
    print(f"  MSE : {mse_val:.4f}")
    print(f"  MAE : {mae_val:.4f}")
    print(f"  RMSE: {rmse_val:.4f}")




# You can also group by 'cutoff' or 'unique_id' for more detailed analysis






#### 4: Final Model Training
# Instantiate the final model with the best configuration
# (same as model_for_cv or re-instantiate for clarity)
final_model_instance = NHITS(loss=final_loss_object, **nhits_best_params)


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

predictions_on_test = nf_final_train.predict(df=df_final_holdout_test)

print(predictions_on_test.shape)







# # For this example, assuming `test_length_per_series` was set up to align with forecasting `h` steps.
# # If `predict()` output doesn't perfectly align or you need more control, consider `predict(futr_df=...)`.
# # Let's merge based on 'unique_id' and 'ds'.
# final_evaluation_df = pd.merge(
#     df_final_holdout_test,
#     predictions_on_test,
#     on=['unique_id', 'ds'],
#     how='left' # Use left to keep all test points; predictions might be shorter if h < test_length_per_series
# )
# final_evaluation_df.dropna(inplace=True) # If some predictions couldn't be made or aligned.


# print(final_evaluation_df.columns)


# #### 6: Evaluation & Final Results


# test_actuals = final_evaluation_df['y']
# test_preds = final_evaluation_df['NHITS']


# final_mae = mae(test_actuals, test_preds)
# final_rmse = rmse(test_actuals, test_preds)


# print(f"\nFinal Evaluation on Holdout Test Set for {'NHITS'}:")
# print(f"  Test MAE: {final_mae:.4f}")
# print(f"  Test RMSE: {final_rmse:.4f}")


# print(f"\nPipeline execution finished at: {datetime.now()} (Ho Chi Minh City Time)")




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

