# src/models/architectures.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size, pred_len=1, output_size=1):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_len = pred_len

        # Total input size after flattening: batch_size x (seq_len * n_features)
        input_dim = seq_len * n_features

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, pred_len * output_size)  # Predict `pred_len` steps

    def forward(self, x):
        batch_size = x.shape[0]
        # Flatten input: (batch_size, seq_len, n_features) -> (batch_size, seq_len * n_features)
        x = x.view(batch_size, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Shape: (batch_size, pred_len * output_size)

        # Reshape to: (batch_size, pred_len)
        x = x.view(batch_size, self.pred_len)
        return x
    
# input: (batch_size, seq_len, input_size)
# output: (batch_size, pred_len, output_size)

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        pred_len: int = 1,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.pred_len = pred_len
        self.output_size = output_size

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Final linear layer to map from hidden state to desired output
        self.fc = nn.Linear(hidden_size, pred_len * output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Pass through LSTM
        out, (h_n, _) = self.lstm(x)  # out: (batch_size, seq_len, hidden_size)

        # Use the last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_size)

        # Project to pred_len × output_size
        predictions = self.fc(last_hidden)  # (batch_size, pred_len × output_size)

        # Reshape to (batch_size, pred_len, output_size)
        predictions = predictions.view(batch_size, self.pred_len, self.output_size)

        return predictions

# Add Transformer, SARIMA wrappers, etc. as needed