'''
PyTorch Lightning Detailed Example: Autoencoder on MNIST

This script demonstrates a comprehensive example of using PyTorch Lightning 
to train an autoencoder model on the MNIST dataset. It covers:
1.  Defining a `LightningModule` (the model, training, validation, test steps, optimizers).
2.  Defining a `LightningDataModule` (data preparation, loading, and splitting).
3.  Using Callbacks (`ModelCheckpoint` for saving the best model, `EarlyStopping` to prevent overfitting).
4.  Using a Logger (`TensorBoardLogger` for experiment tracking).
5.  Configuring and using the `Trainer` for training and testing.
6.  Basic inference and manual model saving/loading (commented out).
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision as tv
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# --------------------------------
# Step 1.0: Define Encoder, Decoder, and Autoencoder as nn.Module
# --------------------------------
class Encoder(nn.Module):
    """
    Encoder network for the Autoencoder.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # Bottleneck layer with 3 features
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """
    Decoder network for the Autoencoder.
    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module):
    """
    Combines Encoder and Decoder into a single Autoencoder model.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        """
        Defines the forward pass for the autoencoder: encode then decode.
        """
        # x is expected to be already flattened by LitAutoEncoder._shared_step or LitAutoEncoder.forward
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction

# --------------------------------
# Step 1.1: Define a LightningModule
# --------------------------------
class LitAutoEncoder(L.LightningModule):
    """
    A LightningModule for an Autoencoder.
    It includes the logic for training, validation, and testing.
    Now accepts an auto_encoder module.
    """
    def __init__(self, auto_encoder: AutoEncoder, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['auto_encoder']) # Don't save the nn.Module itself
        self.auto_encoder = auto_encoder
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Defines the forward pass of the model, used for inference/prediction.
        For an autoencoder, this typically means encoding the input.
        """
        x = x.view(x.size(0), -1) # Flatten the image
        embedding = self.auto_encoder.encoder(x)
        return embedding

    def _shared_step(self, batch):
        """Helper function for common logic in training, validation, and test steps."""
        x, _ = batch
        x_flat = x.view(x.size(0), -1)
        x_hat = self.auto_encoder(x_flat) # Pass through the full AutoEncoder (enc -> dec)
        loss = F.mse_loss(x_hat, x_flat)
        return loss, x_hat, x

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self._shared_step(batch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss, _, _ = self._shared_step(batch)
        self.log("test_loss", test_loss, on_epoch=True, logger=True, sync_dist=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.auto_encoder.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

# ------------------------------------
# Step 2: Define Data and DataLoaders
# ------------------------------------
class MNISTDataModule(L.LightningDataModule):
    '''
    A LightningDataModule for the MNIST dataset.
    It handles downloading, transforming, and creating DataLoaders.
    '''
    def __init__(self, data_dir: str = "./MNIST_data", batch_size: int = 256, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Determine num_workers based on CPU availability, ensuring it's at least 0
        self.num_workers = max(0, num_workers)
        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization, often used for classification
        ])

    def prepare_data(self):
        '''Downloads the dataset (if not already present). Called once per node.'''
        tv.datasets.MNIST(self.data_dir, train=True, download=True)
        tv.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        '''
        Assigns train/val/test datasets. Called on every GPU/TPU in distributed training.
        'stage' can be 'fit', 'validate', 'test', or 'predict'.
        '''
        if stage == "fit" or stage is None:
            mnist_full = tv.datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            # Split training data into training and validation sets
            self.mnist_train, self.mnist_val = data.random_split(mnist_full, [55000, 5000],
                                                                  generator=torch.Generator().manual_seed(42))

        if stage == "test" or stage is None:
            self.mnist_test = tv.datasets.MNIST(self.data_dir, train=False, transform=self.transform)
        
        # if stage == "predict" or stage is None: # For prediction with new data
            # self.mnist_predict = tv.datasets.MNIST(self.data_dir, train=False, transform=self.transform) # Or your custom dataset

    def train_dataloader(self):
        '''Returns the DataLoader for the training set.'''
        return data.DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, 
                               shuffle=True, persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        '''Returns the DataLoader for the validation set.'''
        return data.DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers,
                               persistent_workers=True if self.num_workers > 0 else False)

    def test_dataloader(self):
        '''Returns the DataLoader for the test set.'''
        return data.DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers,
                               persistent_workers=True if self.num_workers > 0 else False)
    
    # def predict_dataloader(self):
        # '''Returns the DataLoader for prediction.'''
        # return data.DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers,
                               # persistent_workers=True if self.num_workers > 0 else False)


def main():
    # For reproducibility
    L.seed_everything(42, workers=True)

    # ----------------------------------------------------
    # Step 3: Initialize Model and DataModule
    # ----------------------------------------------------
    # Initialize the AutoEncoder nn.Module
    core_autoencoder_model = AutoEncoder()
    # Pass the nn.Module to the LightningModule
    autoencoder_model = LitAutoEncoder(auto_encoder=core_autoencoder_model, learning_rate=1e-3)
    # Use a reasonable number of workers, e.g., half of CPU cores, or 0 for main process loading
    num_workers = os.cpu_count() if os.cpu_count() else 0
    mnist_data_module = MNISTDataModule(data_dir="./MNIST_data", batch_size=256, num_workers=num_workers)

    # ----------------------------------------------------
    # Step 4: Configure Callbacks and Logger
    # ----------------------------------------------------
    # ModelCheckpoint: Save the best model based on 'val_loss'
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_detailed_example/",
        filename="best-autoencoder-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,       # Save only the best model
        verbose=True,
        monitor="val_loss", # Metric to monitor
        mode="min"          # 'min' for loss (lower is better), 'max' for accuracy
    )

    # EarlyStopping: Stop training if 'val_loss' doesn't improve for a number of epochs
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,   # Minimum change in the monitored quantity to qualify as an improvement
        patience=10,        # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode="min"
    )

    # TensorBoard Logger for visualizing training progress
    tensorboard_logger = TensorBoardLogger(
        save_dir="lightning_logs_detailed_example/",
        name="autoencoder_mnist"
    )

    # ----------------------------------------------------
    # Step 5: Initialize and Configure the Trainer
    # ----------------------------------------------------
    trainer = L.Trainer(
        accelerator="auto",  # Automatically choose accelerator (cpu, gpu, tpu, mps)
        devices="auto",      # Automatically choose number of devices or specify (e.g., [0, 1] for 2 GPUs)
        max_epochs=50,       # Maximum number of epochs to train
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed", # For mixed precision training (if supported by hardware and beneficial)
        strategy="ddp",     # For distributed training (e.g., "ddp", "fsdp")
        # fast_dev_run=True,  # Run a quick sanity check: 1 batch for train, val, test
        log_every_n_steps=50, # How often to log within an epoch
        deterministic=True    # Ensures reproducibility. Might impact performance slightly.
    )

    # ----------------------------------------------------
    # Step 6: Train the model
    # ----------------------------------------------------
    print("Starting model training...")
    trainer.fit(model=autoencoder_model, datamodule=mnist_data_module)
    print("Training finished.")

    # ----------------------------------------------------
    # Step 7: Test the model
    # ----------------------------------------------------
    # The trainer.test() method will automatically load the best checkpoint 
    # if ModelCheckpoint was used during fitting and monitor is set.
    # Alternatively, you can specify ckpt_path="best" or a specific path.
    print("Starting model testing using the best checkpoint...")
    test_results = trainer.test(model=autoencoder_model, datamodule=mnist_data_module, ckpt_path="best")
    print("Testing results:", test_results)

    # ----------------------------------------------------
    # Step 8: Make predictions (Inference) - Optional
    # ----------------------------------------------------
    # print("\nStarting inference example...")
    # # Load the best model checkpoint manually if needed (e.g., in a separate script)
    # best_model_path = checkpoint_callback.best_model_path
    # if best_model_path:
    #     print(f"Loading best model from: {best_model_path}")
    #     # When loading, you'd need to instantiate the AutoEncoder first, then LitAutoEncoder
    #     core_model_for_loading = AutoEncoder() # Instantiate your core model
    #     loaded_model = LitAutoEncoder.load_from_checkpoint(best_model_path, auto_encoder=core_model_for_loading)
    #     loaded_model.eval() # Set model to evaluation mode
    #
    #     # Get a batch of data for prediction
    #     # Ensure DataModule is set up for the stage you need (e.g., test or a new "predict" stage)
    #     mnist_data_module.setup(stage="test") # or a custom predict stage
    #     predict_loader = mnist_data_module.test_dataloader() # or predict_dataloader()
    #     one_batch = next(iter(predict_loader))
    #     images, _ = one_batch
        
    #     # Move images to the same device as the model
    #     # images = images.to(loaded_model.device)

    #     with torch.no_grad(): # Disable gradient calculations for inference
    #         embeddings = loaded_model(images) # Get embeddings
    #         reconstructions = loaded_model.decoder(embeddings) # Get reconstructions
        
    #     # Reshape reconstructions back to image format if they were flattened
    #     reconstructions = reconstructions.view(images.size(0), 1, 28, 28) 

    #     print(f"Input images shape: {images.shape}")
    #     print(f"Embeddings shape: {embeddings.shape}")
    #     print(f"Reconstructed images shape: {reconstructions.shape}")
        
    #     # Here you could add code to visualize the original vs reconstructed images
    #     # import matplotlib.pyplot as plt
    #     # fig, axes = plt.subplots(2, 5, figsize=(10,4))
    #     # for i in range(5):
    #     #     axes[0,i].imshow(images[i].squeeze().cpu(), cmap='gray')
    #     #     axes[0,i].set_title("Original")
    #     #     axes[0,i].axis('off')
    #     #     axes[1,i].imshow(reconstructions[i].squeeze().cpu(), cmap='gray')
    #     #     axes[1,i].set_title("Reconstructed")
    #     #     axes[1,i].axis('off')
    #     # plt.tight_layout()
    #     # plt.savefig("reconstructions_example.png")
    #     # print("Saved example reconstructions to reconstructions_example.png")
    # else:
    #     print("No best model path found from checkpoint callback.")

    # ---------------------------------------------------------------------
    # Step 9: Save and Load model manually (alternative to checkpointing)
    # ---------------------------------------------------------------------
    # print("\nManual save/load example...")
    # # Save model weights and hyperparameters
    # manual_save_path = "my_autoencoder_manually_saved.ckpt"
    # trainer.save_checkpoint(manual_save_path)
    # print(f"Model manually saved to: {manual_save_path}")

    # # Load model from a checkpoint
    # # When loading manually, instantiate the core model first
    # core_model_for_manual_loading = AutoEncoder() # Instantiate your core model
    # loaded_model_manual = LitAutoEncoder.load_from_checkpoint(manual_save_path, auto_encoder=core_model_for_manual_loading)
    # # To load hyperparameters that were saved with self.save_hyperparameters()
    # # print("Loaded hyperparameters:", loaded_model_manual.hparams)
    # loaded_model_manual.eval()
    # print("Model manually loaded and set to eval mode.")

    print(f"\nPyTorch Lightning detailed example execution complete.")
    print(f"To view logs, run in your terminal: tensorboard --logdir {tensorboard_logger.save_dir}")

if __name__ == '__main__':
    main()
