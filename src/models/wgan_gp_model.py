import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Any, Callable, Optional
from utils import compute_mmd_rbf

lambda_gp_var = 10.0
hidden_dims_default = [512, 1024, 1024, 512]
TRIG_FEATURES = ("sin_phi_s", "cos_phi_s", "sin_theta", "cos_theta")
ENERGY_FEATURE_CANDIDATES = ("log1p_p_mag", "p_mag")
PDG_FEATURE_CANDIDATES = ("pdg", "pdg_id", "pdgid", "pid")
PDG_MASS_GEV = {
    11: 0.00051099895,
    13: 0.1056583755,
    15: 1.77686,
    22: 0.0,
    111: 0.1349768,
    211: 0.13957039,
    310: 0.497611,
    321: 0.493677,
    130: 0.497611,
    311: 0.497611,
    2212: 0.9382720813,
    2112: 0.9395654133,
    3122: 1.115683,
}
DEFAULT_PARTICLE_MASS_GEV = PDG_MASS_GEV[13]
TRIG_CLIP_FEATURES = ("sin_phi_s", "cos_phi_s", "sin_theta", "cos_theta", "phi_p")
POSITIVE_CLIP_FEATURES = ("log1p_r",)  # <-- fix
BOUNDED_CLIP_FEATURES = TRIG_CLIP_FEATURES + POSITIVE_CLIP_FEATURES
UNIT_CIRCLE_PAIRS = (
    ("sin_phi_s", "cos_phi_s"),
    ("sin_theta", "cos_theta"),
)

def _available_unit_circle_pairs(
    feature_index: dict[str, int],
) -> list[tuple[str, str]]:
    """Return only the unit-circle pairs whose both features exist in feature_index."""
    return [(s, c) for (s, c) in UNIT_CIRCLE_PAIRS if s in feature_index and c in feature_index]

def _apply_generation_bounds(samples: np.ndarray, clip_feature_indices: dict[str, int]) -> np.ndarray:
    """Project selected sin/cos pairs to unit circle, then clip configured feature bounds."""
    # 1) Unit-circle projection (generation-time)
    eps = 1e-8
    for sin_name, cos_name in UNIT_CIRCLE_PAIRS:
        sin_idx = clip_feature_indices.get(sin_name)
        cos_idx = clip_feature_indices.get(cos_name)
        if sin_idx is None or cos_idx is None:
            continue

        s = samples[:, sin_idx]
        c = samples[:, cos_idx]
        r = np.sqrt(s * s + c * c)
        r = np.where(r < eps, 1.0, r)  # avoid divide-by-zero
        samples[:, sin_idx] = s / r
        samples[:, cos_idx] = c / r

    # 2) Hard bounds
    clip_bounds = {
        "sin_phi_s": (-1.0, 1.0),
        "cos_phi_s": (-1.0, 1.0),
        "sin_theta": (-1.0, 1.0),
        "cos_theta": (-1.0, 1.0),
        "phi_p": (-np.pi, np.pi),
        "log1p_r": (0.0, np.log1p(350.5)),
        "p_mag": (0.0, np.inf),
    }
    for feature_name in BOUNDED_CLIP_FEATURES:
        feature_idx = clip_feature_indices.get(feature_name)
        if feature_idx is None:
            continue
        lo, hi = clip_bounds[feature_name]
        samples[:, feature_idx] = np.clip(samples[:, feature_idx], lo, hi)

    return samples


def _compute_generator_constraints(
    fake_data: torch.Tensor,
    constraints: Optional[list[Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, float]]]]]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Aggregate modular generator constraints into one scalar loss."""
    total_loss = fake_data.new_tensor(0.0)
    metrics: dict[str, float] = {}
    if not constraints:
        return total_loss, metrics

    for constraint_fn in constraints:
        loss_value, constraint_metrics = constraint_fn(fake_data)
        total_loss = total_loss + loss_value
        if constraint_metrics:
            metrics.update(constraint_metrics)

    return total_loss, metrics


def _make_trig_constraint(
    feature_index: dict[str, int],
    weight: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, float]]]:
    """Build trig unit-circle constraint from UNIT_CIRCLE_PAIRS and available features."""
    weight_value = float(weight)
    mean = mean.detach().clone()
    std = std.detach().clone()

    valid_pairs: list[tuple[str, str, int, int]] = []
    for sin_name, cos_name in UNIT_CIRCLE_PAIRS:
        if sin_name in feature_index and cos_name in feature_index:
            valid_pairs.append((sin_name, cos_name, feature_index[sin_name], feature_index[cos_name]))

    if not valid_pairs:
        # No-op constraint if no configured trig pair exists in this dataset
        def _noop(fake_data: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
            return fake_data.new_tensor(0.0), {}
        return _noop

    def _constraint(fake_data: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        m = mean.to(fake_data.device)
        s = std.to(fake_data.device)

        total_penalty = fake_data.new_tensor(0.0)
        metrics: dict[str, float] = {}

        for sin_name, cos_name, sin_idx, cos_idx in valid_pairs:
            sin_v = fake_data[:, sin_idx] * (s[sin_idx] + 1e-8) + m[sin_idx]
            cos_v = fake_data[:, cos_idx] * (s[cos_idx] + 1e-8) + m[cos_idx]
            pair_penalty = ((sin_v * sin_v + cos_v * cos_v - 1.0) ** 2).mean()
            total_penalty = total_penalty + pair_penalty
            metrics[f"trig_{sin_name}_{cos_name}_penalty"] = float(pair_penalty.detach().item())

        loss = weight_value * total_penalty
        metrics["trig_penalty"] = float(total_penalty.detach().item())
        return loss, metrics

    return _constraint


def _resolve_mass_from_dataframe(dataframe) -> tuple[float, str]:
    for column_name in PDG_FEATURE_CANDIDATES:
        if column_name not in dataframe.columns:
            continue

        series = dataframe[column_name].dropna()
        if len(series) == 0:
            continue

        try:
            pdg_mode = int(np.abs(series.astype(np.int64)).mode().iloc[0])
        except Exception:
            continue

        mass_value = PDG_MASS_GEV.get(pdg_mode)
        if mass_value is not None:
            return float(mass_value), f"pdg:{pdg_mode}"

    return float(DEFAULT_PARTICLE_MASS_GEV), "default:muon"


def _make_energy_constraint(
    feature_index: dict[str, int],
    weight: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_mean_energy: float,
    mass_gev: float,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, float]]]:
    """Build formula-based energy constraint using E = sqrt(p^2 + m^2)."""
    weight_value = float(weight)
    mean = mean.detach().clone()
    std = std.detach().clone()
    target_mean_energy_value = float(target_mean_energy)
    mass_value = float(mass_gev)

    momentum_feature_name = None
    for candidate in ENERGY_FEATURE_CANDIDATES:
        if candidate in feature_index:
            momentum_feature_name = candidate
            break

    if momentum_feature_name is None:
        def _noop(fake_data: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
            return fake_data.new_tensor(0.0), {}
        return _noop

    momentum_idx = feature_index[momentum_feature_name]
    uses_log_momentum = momentum_feature_name == "log1p_p_mag"

    def _constraint(fake_data: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        m = mean.to(fake_data.device)
        s = std.to(fake_data.device)

        momentum = fake_data[:, momentum_idx] * (s[momentum_idx] + 1e-8) + m[momentum_idx]
        if uses_log_momentum:
            momentum = torch.expm1(momentum)
        momentum = torch.clamp(momentum, min=0.0)

        mass_sq = fake_data.new_tensor(mass_value * mass_value)
        energy = torch.sqrt(momentum * momentum + mass_sq)

        target_energy_tensor = fake_data.new_tensor(target_mean_energy_value)
        penalty = (energy.mean() - target_energy_tensor) ** 2
        loss = weight_value * penalty

        metrics = {
            "energy_penalty": float(penalty.detach().item()),
            "energy_mean_fake": float(energy.mean().detach().item()),
            "energy_mean_target": target_mean_energy_value,
        }
        return loss, metrics

    return _constraint

class ParticleDataset(Dataset):
    """PyTorch Dataset for particle physics data (robust-scaled)."""
    def __init__(self, dataframe):
        self.data = torch.FloatTensor(dataframe.values)

        # Robust scaling: center by median, scale by IQR (Q3 - Q1)
        self.center = self.data.median(dim=0).values
        q1 = torch.as_tensor(dataframe.quantile(0.25, numeric_only=False).to_numpy(), dtype=self.data.dtype)
        q3 = torch.as_tensor(dataframe.quantile(0.75, numeric_only=False).to_numpy(), dtype=self.data.dtype)
        self.scale = q3 - q1

        # Avoid divide-by-zero for near-constant columns
        self.scale = torch.where(self.scale.abs() < 1e-8, torch.ones_like(self.scale), self.scale)

        # Keep backward-compatible names used elsewhere in the file
        self.mean = self.center
        self.std = self.scale

        self.data = (self.data - self.center) / (self.scale + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Generator(nn.Module):
    def __init__(self, latent_dim=128, output_dim=8, hidden_dims = hidden_dims_default):
        super().__init__()
        layers = []
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim)) 
            layers.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim=9, hidden_dims = hidden_dims_default):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class WGAN_GP:
    """WGAN-GP for particle physics data with stable training."""
    def __init__(self, input_dim=9, latent_dim=128, device='cuda',
                 lr_g=5e-5, lr_c=5e-5, use_rmsprop=False,
                 generator_constraints=None):
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.generator_constraints = list(generator_constraints) if generator_constraints else []
        self.clip_feature_indices: dict[str, int] = {}

        self.generator = Generator(latent_dim, input_dim).to(device)
        self.critic = Critic(input_dim).to(device)

        # Use Adam with custom betas for WGAN-GP
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
        self.c_optimizer = optim.Adam(self.critic.parameters(), lr=lr_c, betas=(0.0, 0.9))

    def _gradient_penalty(self, real_data, fake_data, lambda_gp=None):
        """Compute gradient penalty for Lipschitz constraint."""
        if lambda_gp is None:
            lambda_gp = lambda_gp_var
        batch_size = real_data.size(0)
        eps = torch.rand(batch_size, 1, device=self.device)
        eps = eps.expand_as(real_data)

        interpolated = eps * real_data + (1 - eps) * fake_data
        interpolated.requires_grad_(True)

        interp_score = self.critic(interpolated)
        grads = torch.autograd.grad(
            outputs=interp_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interp_score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        grads = grads.view(batch_size, -1)
        grad_norm = grads.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return lambda_gp * gp

    def train_step(self, real_data, n_critic=5, lambda_gp=None):
        """One training step: update critic n_critic times, then generator once."""
        if lambda_gp is None:
            lambda_gp = lambda_gp_var
        batch_size = real_data.size(0)

        # Train Critic n_critic times
        critic_losses = []
        for _ in range(n_critic):
            self.c_optimizer.zero_grad()
            
            # Generate fake data
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(z).detach()  # Detach to prevent backprop to generator

            # Critic scores
            real_score = self.critic(real_data)
            fake_score = self.critic(fake_data)

            # Wasserstein loss + gradient penalty
            gp = self._gradient_penalty(real_data, fake_data, lambda_gp)
            c_loss = -(real_score.mean() - fake_score.mean()) + gp
            
            c_loss.backward()
            self.c_optimizer.step()
            critic_losses.append(c_loss.item())

        avg_c_loss = float(np.mean(critic_losses))

        # Train Generator once
        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(z)
        g_adv_loss = -self.critic(fake_data).mean()
        g_constraint_loss, constraint_metrics = _compute_generator_constraints(
            fake_data,
            self.generator_constraints
        )
        g_loss = g_adv_loss + g_constraint_loss
        g_loss.backward()
        self.g_optimizer.step()

        step_metrics = {
            "g_adv_loss": float(g_adv_loss.detach().item()),
            "g_constraint_loss": float(g_constraint_loss.detach().item())
        }
        step_metrics.update(constraint_metrics)

        return avg_c_loss, g_loss.item(), step_metrics

    def compute_validation_wasserstein(self, val_data, mean, std, n_samples=10000):
        """Compute mean Wasserstein distance across all variables on validation set (normalized)."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            synthetic = self.generator(z).cpu().numpy()
        
        # Sample from validation set and normalize using train mean/std
        val_indices = np.random.choice(len(val_data), min(n_samples, len(val_data)), replace=False)
        val_samples = val_data.iloc[val_indices].values.astype(np.float32)
        val_samples = (val_samples - mean.numpy()) / (std.numpy() + 1e-8)
        
        # Compute Wasserstein distance per variable
        wasserstein_dists = []
        for i in range(self.input_dim):
            wd = wasserstein_distance(val_samples[:, i], synthetic[:, i])
            wasserstein_dists.append(wd)
        
        mean_wd = float(np.mean(wasserstein_dists))
        return mean_wd, wasserstein_dists

    def compute_validation_mmd(self, val_data, mean, std, n_samples=10000,
                               sigma="median", sigma_scale=1.0,
                               chunk_size=1024, unbiased=True, seed=42):
        """Compute full multivariate MMD between normalized val samples and generated samples."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            synthetic = self.generator(z).cpu().numpy()

        val_indices = np.random.choice(len(val_data), min(n_samples, len(val_data)), replace=False)
        val_samples = val_data.iloc[val_indices].values.astype(np.float32)
        val_samples = (val_samples - mean.numpy()) / (std.numpy() + 1e-8)

        mmd_metrics = compute_mmd_rbf(
            val_samples,
            synthetic,
            sigma=sigma,
            sigma_scale=sigma_scale,
            max_samples=n_samples,
            chunk_size=chunk_size,
            unbiased=unbiased,
            seed=seed
        )
        return mmd_metrics

    def generate(self, n_samples, mean, std, batch_size=32768):
        """Generate samples in GPU-safe batches, then de-normalize on CPU."""
        self.generator.eval()
        out_chunks = []

        mean_np = mean.detach().cpu().numpy() if torch.is_tensor(mean) else np.asarray(mean)
        std_np = std.detach().cpu().numpy() if torch.is_tensor(std) else np.asarray(std)

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                bs = min(batch_size, n_samples - start)
                z = torch.randn(bs, self.latent_dim, device=self.device)
                chunk = self.generator(z).detach().cpu().numpy()
                chunk = chunk * (std_np + 1e-8) + mean_np
                chunk = _apply_generation_bounds(chunk, self.clip_feature_indices)
                out_chunks.append(chunk)

        return np.concatenate(out_chunks, axis=0)

def _clone_state_dict_to_cpu(module: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

def train_wgan_gp(dataframe, selected_pdg: int, epochs=100, batch_size=512, latent_dim=128, device='cuda',
                  n_critic=5, lambda_gp=10.0, num_workers=1, log_interval=10,
                  patience=25, min_delta=0.0001, val_split=0.2,
                  mmd_sigma="median", mmd_sigma_scale=1.0,
                  mmd_chunk_size=1024, mmd_unbiased=True, mmd_seed=42,
                  lr_g=5e-5, lr_c=5e-5, trig_constraint_weight=0.0,
                  energy_constraint_weight=0.01,
                  energy_mass_mode="pdg_lookup",
                  energy_missing_policy="warn_disable",
                  mmd_eval_interval=1,
                  mmd_eval_samples=20000):
    """Train WGAN-GP on particle physics data with early stopping based on validation MMD."""
    if selected_pdg is None:
        raise ValueError("train_wgan_gp requires selected_pdg (from --pdg in main.py).")

    # Split into train and validation
    n_train = int(len(dataframe) * (1 - val_split))
    train_df = dataframe.iloc[:n_train].copy()
    val_df = dataframe.iloc[n_train:].copy()

    dataset = ParticleDataset(train_df)

    # Reverted to original/simple DataLoader behavior
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    feature_index = {name: idx for idx, name in enumerate(dataframe.columns)}
    generator_constraints = []
    if float(trig_constraint_weight) > 0.0:
        available_pairs = _available_unit_circle_pairs(feature_index)
        if not available_pairs:
            print(f"⚠️ Trig constraint disabled; no complete unit-circle pairs found in columns={list(dataframe.columns)}")
        else:
            print(f"→ Trig constraint enabled for pairs: {available_pairs}")
            generator_constraints.append(
                _make_trig_constraint(
                    feature_index,
                    weight=trig_constraint_weight,
                    mean=dataset.mean,
                    std=dataset.std,
                )
            )

    mass_source = None
    if float(energy_constraint_weight) > 0.0:
        momentum_feature_name = None
        for candidate in ENERGY_FEATURE_CANDIDATES:
            if candidate in feature_index:
                momentum_feature_name = candidate
                break

        if momentum_feature_name is None:
            message = (
                "Energy constraint disabled; missing momentum feature. "
                f"Expected one of: {list(ENERGY_FEATURE_CANDIDATES)}"
            )
            if energy_missing_policy == "warn_disable":
                print(f"⚠️ {message}")
            else:
                raise ValueError(message)
        else:
            if energy_mass_mode == "pdg_lookup":
                pdg_abs = int(np.abs(int(selected_pdg)))
                resolved_mass = PDG_MASS_GEV.get(pdg_abs)
                if resolved_mass is None:
                    mass_gev = float(DEFAULT_PARTICLE_MASS_GEV)
                    mass_source = f"cli_pdg_default:{pdg_abs}"
                else:
                    mass_gev = float(resolved_mass)
                    mass_source = f"cli_pdg:{pdg_abs}"
            else:
                raise ValueError(
                    f"Unsupported energy_mass_mode={energy_mass_mode}; expected 'pdg_lookup'"
                )

            if mass_source.startswith("cli_pdg_default:"):
                print(
                    "⚠️ Energy constraint mass fallback to muon mass "
                    f"({DEFAULT_PARTICLE_MASS_GEV:.6f} GeV); unsupported CLI PDG={selected_pdg}"
                )

            momentum_values = train_df[momentum_feature_name].values.astype(np.float32)
            if momentum_feature_name == "log1p_p_mag":
                momentum_values = np.expm1(momentum_values)
            momentum_values = np.clip(momentum_values, a_min=0.0, a_max=None)

            target_mean_energy = float(np.mean(np.sqrt(momentum_values * momentum_values + mass_gev * mass_gev)))
            print(
                f"→ Energy constraint enabled: feature={momentum_feature_name}, "
                f"mass_source={mass_source}, target_mean_E={target_mean_energy:.6f}"
            )

            generator_constraints.append(
                _make_energy_constraint(
                    feature_index,
                    weight=energy_constraint_weight,
                    mean=dataset.mean,
                    std=dataset.std,
                    target_mean_energy=target_mean_energy,
                    mass_gev=mass_gev,
                )
            )

    missing_required_clip_features = [
        name for name in POSITIVE_CLIP_FEATURES
        if name not in feature_index
    ]
    if missing_required_clip_features:
        raise ValueError(
            f"Missing required generation clipping columns: {missing_required_clip_features}"
        )

    wgan_gp = WGAN_GP(
        input_dim=dataframe.shape[1],
        latent_dim=latent_dim,
        device=device,
        lr_g=lr_g,
        lr_c=lr_c,
        generator_constraints=generator_constraints,
    )

    # Simple local LR scheduler: decay both optimizers every few epochs.
    lr_decay_every = 5
    lr_decay_gamma = 0.9
    g_scheduler = optim.lr_scheduler.StepLR(
        wgan_gp.g_optimizer,
        step_size=lr_decay_every,
        gamma=lr_decay_gamma,
    )
    c_scheduler = optim.lr_scheduler.StepLR(
        wgan_gp.c_optimizer,
        step_size=lr_decay_every,
        gamma=lr_decay_gamma,
    )

    wgan_gp.clip_feature_indices = {
        name: feature_index[name]
        for name in BOUNDED_CLIP_FEATURES
        if name in feature_index
    }

    print(f"Training WGAN-GP for {epochs} epochs on {device}...")
    print(f"Train: {len(train_df)} samples, Validation: {len(val_df)} samples")
    history: dict[str, Any] = {
        "epoch": [],
        "d_loss": [],
        "g_loss": [],
        "g_constraint_loss": [],
        "trig_penalty": [],
        "energy_penalty": [],
        "train_wasserstein": [],
        "val_wasserstein": [],
        "train_mmd": [],
        "val_mmd": []
    }
    
    best_val_wd = float('inf')
    best_val_mmd = float('inf')
    best_epoch = None
    best_gen_state = None
    best_critic_state = None

    patience_counter = 0
    warmup_epochs = 15

    for epoch in range(epochs):
        c_losses, g_losses = [], []
        g_constraint_losses = []
        trig_penalties = []
        energy_penalties = []
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            c_loss, g_loss, step_metrics = wgan_gp.train_step(
                batch,
                n_critic=n_critic,
                lambda_gp=lambda_gp
            )
            c_losses.append(c_loss)
            g_losses.append(g_loss)
            g_constraint_losses.append(float(step_metrics.get("g_constraint_loss", 0.0)))
            if "trig_penalty" in step_metrics:
                trig_penalties.append(float(step_metrics["trig_penalty"]))
            if "energy_penalty" in step_metrics:
                energy_penalties.append(float(step_metrics["energy_penalty"]))

        avg_c_loss = float(np.mean(c_losses))
        avg_g_loss = float(np.mean(g_losses))
        avg_g_constraint_loss = float(np.mean(g_constraint_losses)) if g_constraint_losses else 0.0
        avg_trig_penalty = float(np.mean(trig_penalties)) if trig_penalties else 0.0
        avg_energy_penalty = float(np.mean(energy_penalties)) if energy_penalties else 0.0

        # Keep WD every epoch
        train_wd, train_wd_per_var = wgan_gp.compute_validation_wasserstein(
            train_df, dataset.mean, dataset.std, n_samples=5000
        )
        val_wd, val_wd_per_var = wgan_gp.compute_validation_wasserstein(
            val_df, dataset.mean, dataset.std, n_samples=5000
        )

        # MMD only every few epochs, but with larger sample count
        do_mmd_eval = (
            (epoch == 0)
            or ((epoch + 1) % int(mmd_eval_interval) == 0)
            or (epoch + 1 == epochs)
        )
        if do_mmd_eval:
            train_mmd_metrics = wgan_gp.compute_validation_mmd(
                train_df,
                dataset.mean,
                dataset.std,
                n_samples=int(mmd_eval_samples),
                sigma=mmd_sigma,
                sigma_scale=mmd_sigma_scale,
                chunk_size=mmd_chunk_size,
                unbiased=mmd_unbiased,
                seed=mmd_seed + epoch
            )
            val_mmd_metrics = wgan_gp.compute_validation_mmd(
                val_df,
                dataset.mean,
                dataset.std,
                n_samples=int(mmd_eval_samples),
                sigma=mmd_sigma,
                sigma_scale=mmd_sigma_scale,
                chunk_size=mmd_chunk_size,
                unbiased=mmd_unbiased,
                seed=mmd_seed + 10000 + epoch
            )
        else:
            train_mmd_metrics = {
                "mmd": history["train_mmd"][-1] if history["train_mmd"] else float("nan")
            }
            val_mmd_metrics = {
                "mmd": history["val_mmd"][-1] if history["val_mmd"] else float("nan")
            }

        history["epoch"].append(epoch + 1)
        history["d_loss"].append(avg_c_loss)
        history["g_loss"].append(avg_g_loss)
        history["g_constraint_loss"].append(avg_g_constraint_loss)
        history["trig_penalty"].append(avg_trig_penalty)
        history["energy_penalty"].append(avg_energy_penalty)
        history["train_wasserstein"].append(train_wd)
        history["val_wasserstein"].append(val_wd)
        history["train_mmd"].append(train_mmd_metrics["mmd"])
        history["val_mmd"].append(val_mmd_metrics["mmd"])

        prev_g_lr = float(wgan_gp.g_optimizer.param_groups[0]["lr"])
        prev_c_lr = float(wgan_gp.c_optimizer.param_groups[0]["lr"])
        g_scheduler.step()
        c_scheduler.step()
        curr_g_lr = float(wgan_gp.g_optimizer.param_groups[0]["lr"])
        curr_c_lr = float(wgan_gp.c_optimizer.param_groups[0]["lr"])

        if curr_g_lr < prev_g_lr:
            print(
                f"→ Reduced Generator LR: {prev_g_lr:.2e} -> {curr_g_lr:.2e} "
                f"(epoch {epoch+1}, every {lr_decay_every} epochs, gamma={lr_decay_gamma})"
            )
        if curr_c_lr < prev_c_lr:
            print(
                f"→ Reduced Critic LR: {prev_c_lr:.2e} -> {curr_c_lr:.2e} "
                f"(epoch {epoch+1}, every {lr_decay_every} epochs, gamma={lr_decay_gamma})"
            )

        if (epoch + 1) % log_interval == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] C_loss: {avg_c_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
                f"G_const: {avg_g_constraint_loss:.4f}, "
                f"Train WD: {train_wd:.4f}, Val WD: {val_wd:.4f}, "
                f"Train MMD: {train_mmd_metrics['mmd']:.4f}, Val MMD: {val_mmd_metrics['mmd']:.4f}"
            )

        current_val_mmd = val_mmd_metrics["mmd"]
        current_val_wd = val_wd

        # Checkpoint on val WD improvement
        if current_val_wd < best_val_wd - min_delta:
            print(f"→ New best Val WD: {current_val_wd:.6f} (Val MMD: {current_val_mmd:.4f})")
            best_val_wd = current_val_wd
            best_val_mmd = val_mmd_metrics["mmd"]
            best_epoch = epoch + 1
            best_gen_state = _clone_state_dict_to_cpu(wgan_gp.generator)
            best_critic_state = _clone_state_dict_to_cpu(wgan_gp.critic)
            patience_counter = 0
        elif epoch >= warmup_epochs:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}: Val WD no improvement for {patience} epochs")
                break

    # Restore best weights before returning
    if best_gen_state is not None and best_critic_state is not None:
        wgan_gp.generator.load_state_dict(best_gen_state)
        wgan_gp.critic.load_state_dict(best_critic_state)
        print(f"✅ Restored best checkpoint from epoch {best_epoch} (Val WD={best_val_wd:.6f}, Val MMD={best_val_mmd:.4f})")

    history["best_epoch"] = best_epoch
    history["best_val_wasserstein"] = best_val_wd
    history["best_val_mmd"] = best_val_mmd
    history["mmd_config"] = {
        "sigma": mmd_sigma,
        "sigma_scale": float(mmd_sigma_scale),
        "chunk_size": int(mmd_chunk_size),
        "unbiased": bool(mmd_unbiased),
        "seed": int(mmd_seed)
    }
    history["constraint_config"] = {
        "trig_constraint_weight": float(trig_constraint_weight),
        "energy_constraint_weight": float(energy_constraint_weight),
        "energy_mass_mode": str(energy_mass_mode),
        "energy_missing_policy": str(energy_missing_policy),
        "selected_pdg": int(selected_pdg),
        "mass_source": str(mass_source),
    }
    history["lr_scheduler_config"] = {
        "type": "StepLR",
        "decay_every": int(lr_decay_every),
        "gamma": float(lr_decay_gamma)
    }

    return wgan_gp, dataset.mean, dataset.std, history