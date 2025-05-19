import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28)
        )
        
    def forward(self, x):
        # Inference path
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # Flatten target for loss
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)  # Flatten input
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# 2. Data Module
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        
    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        
    def train_dataloader(self):
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=True
        )
        return DataLoader(train_dataset, batch_size=32, num_workers=7)
    
    def val_dataloader(self):
        val_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=True
        )
        return DataLoader(val_dataset, batch_size=32, num_workers=7)

# 3. Training with advanced features
def main():
    model = LitAutoEncoder()
    data = MNISTDataModule()
    trainer = pl.Trainer(
        accelerator="auto",
        devices=-1,
        max_epochs=10,
        logger=pl.loggers.TensorBoardLogger("lightning_logs/"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="best_model",
                monitor="val_loss"
            ),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        ],
        precision="16-mixed",
        strategy="ddp"
    )
    trainer.fit(model, datamodule=data)
    scripted_model = model.to_torchscript()
    torch.jit.save(scripted_model, "model.pt")
    
if __name__ == "__main__":
    main()