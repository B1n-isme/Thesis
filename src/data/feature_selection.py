import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import optuna

def load_and_preprocess_data(filepath, target_col, seq_len=30, train_split=0.85):
    df = pd.read_csv(filepath)
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    data = df[numeric].values.astype(np.float32)
    
    split = int(train_split * len(data))
    train_data, val_data = data[:split], data[split:]
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_val = scaler.transform(val_data)
    scaled = np.vstack([scaled_train, scaled_val])
    
    X = create_sequences(scaled, seq_len)
    return X, df, target_col

def create_sequences(arr, seq_len):
    return np.stack([arr[i:i+seq_len] for i in range(len(arr) - seq_len + 1)])

class SeqDataset(Dataset):
    def __init__(self, sequences):
        self.seq = torch.from_numpy(sequences.astype(np.float32))
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, i):
        return self.seq[i], self.seq[i]

class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features, latent_dim, nhead=8, num_layers=2, dropout=0.1, embed_dim=32):
        super().__init__()
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"
        self.input_proj = nn.Linear(n_features, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.encoder_linear = nn.Linear(embed_dim, latent_dim)
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.decoder_linear = nn.Linear(latent_dim, n_features)

    def forward(self, x):
        x_proj = self.input_proj(x)  # (batch, seq, embed_dim)
        memory = self.encoder(x_proj)  # (batch, seq, embed_dim)
        latent = self.encoder_linear(memory)  # (batch, seq, latent_dim)
        tgt = torch.zeros_like(latent)  # (batch, seq, latent_dim)
        out = self.decoder(tgt, latent)  # (batch, seq, latent_dim)
        recon = self.decoder_linear(out)  # (batch, seq, n_features)
        return recon, latent

def train_autoencoder(trial, n_features, train_loader, val_loader, device, latent_dim_choices):
    latent_dim = trial.suggest_categorical('latent_dim', latent_dim_choices)
    model = TransformerAutoencoder(
        n_features=n_features,
        latent_dim=latent_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    for epoch in range(30):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return best_val_loss

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, _ = model(x)
        loss = loss_fn(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, _ = model(x)
            total_loss += loss_fn(recon, x).item()
    return total_loss / len(loader)

def main():
    # Load and preprocess data
    data_path = 'data/final/dataset.csv'

    X, df, target_col = load_and_preprocess_data(data_path, 'btc_price')
    n_features = X.shape[2]
    nhead = 8  # must match TransformerAutoencoder default
    latent_dim_choices = [i for i in range(nhead, 51, nhead)]  # [8, 16, 24, 32, 40, 48]
    
    # Create data loaders
    split_seq = int(0.85 * len(X))
    train_ds = SeqDataset(X[:split_seq])
    val_ds = SeqDataset(X[split_seq:])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial):
        return train_autoencoder(
            trial,
            n_features=n_features,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            latent_dim_choices=latent_dim_choices
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

if __name__ == "__main__":
    main()

# Best trial:
#   Value: 0.5408926039934159
#   Params: {'latent_dim': 48}

# Best trial:
#   Value: 1.0243791533367974
#   Params: {'latent_dim': 40}