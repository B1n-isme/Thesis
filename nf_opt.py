import logging

import torch
import pandas as pd
from utilsforecast.plotting import plot_series

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, BiTCN
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MQLoss, DistributionLoss
from utilsforecast.evaluation import evaluate

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
torch.set_float32_matmul_precision('high')

# Load data
df = pd.read_csv('data/final/dataset.csv')
df = df.rename(columns={'Date': 'ds', 'btc_close': 'y'})
df['unique_id'] = 'Bitcoin'
df['ds'] = pd.to_datetime(df['ds'])

all_cols = df.columns.tolist()
exog_vars = [col for col in all_cols if col not in ['ds', 'unique_id', 'y']]

horizon = 1
levels = [80, 90]
nhits_default_config = AutoNHITS.get_default_config(h = horizon, backend="optuna")  
def convert_to_tuple(selected_value):
    if isinstance(selected_value, str):
        return tuple(map(int, selected_value.split(',')))
    elif isinstance(selected_value, (list, tuple)):
        return tuple(selected_value)
    else:
        raise ValueError("Unsupported type for kernel size conversion")

def config_nhits(trial):
    n_pool_kernel_size = trial.suggest_categorical(  # MaxPooling Kernel size
        "n_pool_kernel_size",
        ["1,1,1,1,1", "2,2,2,2,2", "4,4,4,4,4", "8,4,2,1,1"]
    )
    n_freq_downsample = trial.suggest_categorical(   # Interpolation expressivity ratios
        "n_freq_downsample",
        ["8,4,2,1,1", "1,1,1,1,1"]
    )
    return {
        "input_size": trial.suggest_categorical(          # Length of input window
            "input_size", (48, 48*2, 48*3)                
        ),                                                
        "start_padding_enabled": True,                                          
        "n_blocks": 5 * [1],                              # Length of input window
        "mlp_units": 5 * [[64, 64]],                      # Length of input window
        "n_pool_kernel_size": convert_to_tuple(n_pool_kernel_size),
        "n_freq_downsample": convert_to_tuple(n_freq_downsample),
        "learning_rate": trial.suggest_float(             # Initial Learning rate
            "learning_rate",
            low=1e-4,
            high=1e-2,
            log=True,
        ),            
        "scaler_type": None,                              # Scaler type
        "max_steps": 1000,                                # Max number of training iterations
        "batch_size": trial.suggest_categorical(          # Number of series in batch
            "batch_size",
            (1, 4, 10),
        ),                   
        "windows_batch_size": trial.suggest_categorical(  # Number of windows in batch
            "windows_batch_size",
            (128, 256, 512),
        ),      
        "random_seed": trial.suggest_int(                 # Random seed   
            "random_seed",
            low=1,
            high=20,
        ),                      
    }


nhits_default_config = AutoNHITS.get_default_config(h = horizon, backend="ray")      


models = [NHITS(h = horizon,
                loss=MAE(),
                max_steps=100,
                input_size = 5*horizon,
                hist_exog_list = exog_vars,
                scaler_type = 'standard'),
        AutoNHITS(h=1,
                #   loss=MQLoss(level=levels), # probabilistic forecasting
                loss=DistributionLoss("Normal", level=[90]), # Uncertainty quantification
                  config=config_nhits,
                  search_alg=optuna.samplers.TPESampler(),
                  backend='ray',
                  num_samples=20)
]

nf = NeuralForecast(models=models, 
                    freq='D', 
                    local_scaler_type='standard')

# set val_size & test_size
Y_hat_df = nf.cross_validation(df, n_windows=4, step_size=horizon//2, verbose=1, refit=True) 

# nf.fit(df=df, val_size=horizon*2)

# Y_hat_insample = nf.predict_insample(step_size=horizon)

results = nf.models[0].results.trials_dataframe()
results.drop(columns='user_attrs_ALL_PARAMS')

evaluation_df = evaluate(Y_hat_df.drop(columns='cutoff'), metrics=[MSE, MAE, RMSE])
evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
evaluation_df.head()

summary_df = evaluation_df.groupby(['metric', 'best_model']).size().sort_values().to_frame()
summary_df = summary_df.reset_index()
summary_df.columns = ['metric', 'model', 'nr. of unique_ids']
summary_df

Y_hat_df1 = nf.predict(df= test_df, level = [90]) # generate conformal intervals
Y_hat_df1 = Y_hat_df1.reset_index()
Y_hat_df1.head()

nf.save(path='./checkpoints/test_run/',
        model_index=None, # None for all models
        overwrite=True,
        save_dataset=True)

nf2 = NeuralForecast.load(path='./checkpoints/test_run/')
Y_hat_df2 = nf2.predict()
Y_hat_df2.head()