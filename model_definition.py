from neuralforecast.auto import AutoNHITS, AutoNBEATS
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MQLoss, DistributionLoss


import ray
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
import torch


torch.set_float32_matmul_precision('high')


from config import get_nbeats_config, get_nhits_config


# Import the functions that provide the configs
from config import get_nhits_config, get_nbeats_config


def get_auto_models(horizon: int, loss_fn=MAE(), num_samples_per_model: int = 5):
    """
    Defines and returns a list of Auto models for NeuralForecast.


    Args:
        horizon (int): The forecast horizon (h) for the models.
        loss_fn (object): The loss function to use for the models. Defaults to MAE.
        num_samples_per_model (int): The number of hyperparameter combinations
                                      to try for each model during tuning.


    Returns:
        list: A list of configured Auto models.
    """
    # Get the specific configs using the provided horizon
    nhits_config = get_nhits_config(horizon)
    # nbeats_config = get_nbeats_config(horizon)


    # Define your Auto models
    auto_nhits = AutoNHITS(
        h=horizon,
        loss=loss_fn,
        config=nhits_config,
        num_samples=num_samples_per_model,
        backend='ray',
        # Optional: specific resources per trial if needed, e.g., cpus=2, gpus=0.5
    )


    # auto_nbeats = AutoNBEATS(
    #     h=horizon,
    #     loss=loss_fn,
    #     config=nbeats_config,
    #     num_samples=num_samples_per_model,
    #     backend='ray',
    # )


    # Add other Auto models here if you have them
    # auto_tft = AutoTFT(
    #     h=horizon,
    #     loss=loss_fn,
    #     config=get_tft_config(horizon), # Assuming you have get_tft_config in config.py
    #     num_samples=num_samples_per_model,
    #     backend='ray',
    # )


    return [auto_nhits] # Add auto_tft if uncommented

