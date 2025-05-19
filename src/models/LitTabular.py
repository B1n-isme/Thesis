import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class LitTabularForecaster(L.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def _common_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x).squeeze(1)
        return F.mse_loss(y_hat, y)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }