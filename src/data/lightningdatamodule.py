import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from typing import Tuple, Optional

class WindowedTimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        lookback_window: int = 7,
        n_features: int = 1,
        pin_memory: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Store raw data
        self.train_X, self.train_y = train_data
        self.val_X, self.val_y = val_data if val_data else (None, None)
        self.test_X, self.test_y = test_data if test_data else (None, None)

    def _prepare_tensor_dataset(self, X: np.ndarray, y: np.ndarray) -> TensorDataset:
        """Convert numpy arrays to TensorDataset with flattened input."""
        if X.size == 0 or y.size == 0:
            return None
            
        input_size = self.hparams.lookback_window * self.hparams.n_features
        X_flat = X.reshape(X.shape[0], input_size)
        
        return TensorDataset(
            torch.tensor(X_flat, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

    def setup(self, stage: str = None):
        """Create datasets for train/val/test stages."""
        # Train dataset
        self.train_dataset = self._prepare_tensor_dataset(self.train_X, self.train_y)
        
        # Validation dataset (optional)
        if self.val_X is not None and self.val_y is not None:
            self.val_dataset = self._prepare_tensor_dataset(self.val_X, self.val_y)
        
        # Test dataset (optional)
        if self.test_X is not None and self.test_y is not None:
            self.test_dataset = self._prepare_tensor_dataset(self.test_X, self.test_y)

    def train_dataloader(self) -> DataLoader:
        if not self.train_dataset:
            raise ValueError("Train dataset is empty or not initialized.")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        if not self.val_dataset:
            return None
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        if not self.test_dataset:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )