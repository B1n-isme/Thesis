from ray import tune
import torch


torch.set_float32_matmul_precision('high')


# You can define a function to dynamically generate configs based on horizon
def get_nhits_config(h):
    return{
        "input_size": tune.choice([h * 1, h*2]),         # Minimal input window
        "step_size": h,              # Single-step prediction
        "max_steps": 1000,                           # Small number of steps
        "val_check_steps": 10,
        "learning_rate": tune.loguniform(1e-3, 1e-2),     # Narrowed range for faster convergence
        "batch_size": 32,                           # Small batch size
        "windows_batch_size": 128,                  # Smaller makes training stable
    }
    # return {
    #     # Computed input size: h * multiplier, where multiplier ∈ [1, 2, 3, 4, 5]
    #     "input_size": tune.choice([h * 1, h * 2, h * 3, h * 4, h * 5]),


    #     # Step size can be 1 or equal to h
    #     "step_size": tune.choice([1, h]),


    #     # Pooling kernel sizes for NHITS stack
    #     "n_pool_kernel_size": tune.choice([
    #         [2, 2, 1],
    #         [1, 1, 1],
    #         [2, 2, 2],
    #         [4, 4, 4],
    #         [8, 4, 1],
    #         [16, 8, 1]
    #     ]),


    #     # Frequency downsampling per stack level
    #     "n_freq_downsample": tune.choice([
    #         [168, 24, 1],
    #         [24, 12, 1],
    #         [180, 60, 1],
    #         [60, 8, 1],
    #         [40, 20, 1],
    #         [1, 1, 1]
    #     ]),


    #     # Log-uniform learning rate sampling
    #     "learning_rate": tune.loguniform(1e-4, 1e-1),


    #     # Scaler type for preprocessing
    #     "scaler_type": tune.choice([None, "robust", "standard"]),


    #     # Training steps (sampled in steps of 100)
    #     "max_steps": tune.quniform(lower=500, upper=1500, q=100),


    #     # Mini-batch size for training
    #     "batch_size": tune.choice([32, 64, 128, 256]),


    #     # Number of windows per training batch
    #     "windows_batch_size": tune.choice([128, 256, 512, 1024]),


    #     # Random seed (integer between 1 and 19)
    #     "random_seed": tune.randint(lower=1, upper=20),
    # }


def get_nbeats_config(h):
    return {
        # Computed input size: h * multiplier, where multiplier ∈ [1, 2, 3, 4, 5]
        "input_size": tune.choice([h * 1, h * 2, h * 3, h * 4, h * 5]),


        # Step size can be 1 or equal to h
        "step_size": tune.choice([1, h]),


        # Log-uniform distribution for learning rate
        "learning_rate": tune.loguniform(1e-4, 1e-1),


        # Data scaling method
        "scaler_type": tune.choice([None, "robust", "standard"]),


        # Max training steps: fixed choices only
        "max_steps": tune.choice([500, 1000]),


        # Mini-batch size for training
        "batch_size": tune.choice([32, 64, 128, 256]),


        # Number of training windows in a batch
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),


        # Loss function placeholder (set externally)
        "loss": None,


        # Seed for reproducibility
        "random_seed": tune.randint(1, 20),
    }

