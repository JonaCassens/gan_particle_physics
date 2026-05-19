import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ParticleDataset(Dataset):
    """PyTorch Dataset for particle physics data."""
    def __init__(self, dataframe):
        self.data = torch.FloatTensor(dataframe.values)
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)
        self.data = (self.data - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Generator(nn.Module):
    """Generator network: noise -> synthetic samples."""
    def __init__(self, latent_dim=100, output_dim=6, hidden_dims=[128, 256, 128]):
        super().__init__()
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

class Critic(nn.Module):
    """Critic network: sample -> real-valued score."""
    def __init__(self, input_dim=6, hidden_dims=[128, 64]):
        super().__init__()
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

class WGAN:
    """WGAN (weight clipping) for particle physics data."""
    def __init__(self, input_dim=6, latent_dim=100, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim

        self.generator = Generator(latent_dim, input_dim).to(device)
        self.critic = Critic(input_dim).to(device)

        # WGAN recommends RMSprop
        self.g_optimizer = optim.RMSprop(self.generator.parameters(), lr=5e-5)
        self.c_optimizer = optim.RMSprop(self.critic.parameters(), lr=5e-5)

    def train_step(self, real_data, clip_value=0.01, n_critic=5):
        batch_size = real_data.size(0)

        # Train Critic n_critic times
        for _ in range(n_critic):
            self.c_optimizer.zero_grad()
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(z).detach()

            real_score = self.critic(real_data)
            fake_score = self.critic(fake_data)

            c_loss = -(real_score.mean() - fake_score.mean())
            c_loss.backward()
            self.c_optimizer.step()

            # Weight clipping
            for p in self.critic.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # Train Generator
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(z)
        g_loss = -self.critic(fake_data).mean()
        g_loss.backward()
        self.g_optimizer.step()

        return c_loss.item(), g_loss.item()

    def generate(self, n_samples, mean, std):
        """Generate synthetic samples and denormalize."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            synthetic = self.generator(z).cpu().numpy()
        synthetic = synthetic * std.numpy() + mean.numpy()
        return synthetic

def train_wgan(dataframe, epochs=100, batch_size=512, latent_dim=100, device='cuda',
               n_critic=5, clip_value=0.01, num_workers=2, log_interval=10):
    """Train WGAN on particle physics data."""
    dataset = ParticleDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    wgan = WGAN(input_dim=dataframe.shape[1], latent_dim=latent_dim, device=device)

    print(f"Training WGAN for {epochs} epochs on {device}...")
    history = {"epoch": [], "d_loss": [], "g_loss": []}

    for epoch in range(epochs):
        c_losses, g_losses = [], []
        for batch in dataloader:
            batch = batch.to(device)
            c_loss, g_loss = wgan.train_step(batch, clip_value=clip_value, n_critic=n_critic)
            c_losses.append(c_loss)
            g_losses.append(g_loss)

        history["epoch"].append(epoch + 1)
        history["d_loss"].append(float(np.mean(c_losses)))
        history["g_loss"].append(float(np.mean(g_losses)))

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch [{epoch+1}/{epochs}] C_loss: {np.mean(c_losses):.4f}, G_loss: {np.mean(g_losses):.4f}")

    return wgan, dataset.mean, dataset.std, history