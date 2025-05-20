import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Callable

class LitTabularForecaster(L.LightningModule):
    """
    Generic LightningModule for tabular/windowed time series forecasting.
    Accepts any nn.Module as the model.
    """
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        loss_fn: Callable = F.mse_loss,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _common_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)  # shape: (batch_size, pred_len) for multi-step

        # Optional: Ensure both have same dtype (e.g., float32)
        y = y.to(y_hat.dtype)

        # No need to squeeze unless predicting single step and expecting 1D output
        if y_hat.shape[1] == 1:
            y_hat = y_hat.squeeze(1)  # optional, for compatibility with 1D metrics

        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        # testing with other optimizers and schedulers
        # optimizer2 = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class TabularWindowedDataModule(L.LightningDataModule):
    def __init__(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None,
                 lookback_window=1, n_features=1, batch_size=64, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.lookback_window = lookback_window
        self.n_features = n_features
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        # Reshape X arrays: (n_samples, lookback_window, n_features) -> (n_samples, lookback_window * n_features)
        if self.X_train is not None:
            self.X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
            self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            self.X_train_tensor = torch.tensor(self.X_train_flat, dtype=torch.float32)
            self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        else:
            self.train_dataset = None
        if self.X_val is not None:
            self.X_val_flat = self.X_val.reshape(self.X_val.shape[0], -1)
            self.y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32)
            self.X_val_tensor = torch.tensor(self.X_val_flat, dtype=torch.float32)
            self.val_dataset = TensorDataset(self.X_val_tensor, self.y_val_tensor)
        else:
            self.val_dataset = None
        if self.X_test is not None:
            self.X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
            self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)
            self.X_test_tensor = torch.tensor(self.X_test_flat, dtype=torch.float32)
            self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) if self.train_dataset else None

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) if self.val_dataset else None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) if self.test_dataset else None

    def configure_optimizers(self):
        # Optional: for hyperparameter tuning compatibility
        return None 