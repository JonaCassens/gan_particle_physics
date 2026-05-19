import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class ParticleDataset(Dataset):
    """PyTorch Dataset for particle physics data."""
    def __init__(self, dataframe):
        self.data = torch.FloatTensor(dataframe.values)
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)
        # Normalize
        self.data = (self.data - self.mean) / (self.std + 1e-8)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Generator(nn.Module):
    """Generator network: noise -> synthetic samples."""
    def __init__(self, latent_dim=100, output_dim=6, hidden_dims=[128, 256, 128]):
        super(Generator, self).__init__()
        
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """Discriminator network: sample -> real/fake probability."""
    def __init__(self, input_dim=6, hidden_dims=[128, 64]):
        super(Discriminator, self).__init__()
        
        layers = []
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class GAN:
    """Wasserstein GAN with Gradient Penalty for particle physics data."""
    def __init__(self, input_dim=6, latent_dim=100, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim
        
        self.generator = Generator(latent_dim, input_dim).to(device)
        self.discriminator = Discriminator(input_dim).to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_step(self, real_data):
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(z).detach()
        fake_output = self.discriminator(fake_data)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate(self, n_samples, mean, std):
        """Generate synthetic samples and denormalize."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            synthetic = self.generator(z).cpu().numpy()
        
        # Denormalize
        synthetic = synthetic * std.numpy() + mean.numpy()
        return synthetic

def train_gan(dataframe, epochs=100, batch_size=512, latent_dim=100, device='cuda'):
    """Train GAN on particle physics data."""
    dataset = ParticleDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    gan = GAN(input_dim=dataframe.shape[1], latent_dim=latent_dim, device=device)
    
    print(f"Training GAN for {epochs} epochs on {device}...")
    
    history = {"epoch": [], "d_loss": [], "g_loss": []}

    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)
            d_loss, g_loss = gan.train_step(batch)
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            num_batches += 1

        if num_batches > 0:
            history["epoch"].append(epoch + 1)
            history["d_loss"].append(epoch_d_loss / num_batches)
            history["g_loss"].append(epoch_g_loss / num_batches)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] D_loss: {epoch_d_loss/num_batches:.4f}, G_loss: {epoch_g_loss/num_batches:.4f}")

    return gan, dataset.mean, dataset.std, history