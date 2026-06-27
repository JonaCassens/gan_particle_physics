"""Conditional WGAN-GP (cWGAN-GP) conditioned on PDG particle code.

Mirrors all conventions from wgan_gp_model.py. The PDG code is passed as an
integer label (not a raw float) and mapped to a learned dense embedding that is
concatenated to the latent vector (Generator) or the feature vector (Critic)
before the MLP trunk.

Public API
----------
train_cwgan_gp(dataframe, ...) -> (CWGAN_GP, mean, std, history)
    Drop-in replacement for train_wgan_gp; expects a "pdg" column.

CWGAN_GP.generate(n_samples, pdg_code, mean, std, ...) -> np.ndarray
    pdg_code may be a single int (broadcast) or an array of ints (per-sample).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import wasserstein_distance
from typing import Any, Callable, Optional, Union
from utils import compute_mmd_rbf

# ---------------------------------------------------------------------------
# Constants — identical to wgan_gp_model.py
# ---------------------------------------------------------------------------

lambda_gp_var = 10.0
hidden_dims_default = [512, 1024, 1024, 512]

TRIG_FEATURES = ("sin_phi_s", "cos_phi_s", "sin_theta", "cos_theta")
ENERGY_FEATURE_CANDIDATES = ("log1p_p_mag", "p_mag")
PDG_FEATURE_CANDIDATES = ("pdg", "pdg_id", "pdgid", "pid")

PDG_MASS_GEV = {
    11: 0.00051099895,
    13: 0.1056583755,
    211: 0.13957039,
    321: 0.493677,
    2212: 0.9382720813,
}
DEFAULT_PARTICLE_MASS_GEV = PDG_MASS_GEV[13]

TRIG_CLIP_FEATURES = ("sin_phi_s", "cos_phi_s", "sin_theta", "cos_theta", "phi_p")
POSITIVE_CLIP_FEATURES = ()  # ("r", "p_mag")
BOUNDED_CLIP_FEATURES = TRIG_CLIP_FEATURES + POSITIVE_CLIP_FEATURES

UNIT_CIRCLE_PAIRS = (
    ("sin_phi_s", "cos_phi_s"),
    ("sin_theta", "cos_theta"),
)

# ---------------------------------------------------------------------------
# Conditioning vocabulary
# ---------------------------------------------------------------------------

# Canonical PDG codes supported as conditioning labels.
# Any unseen code at inference time will map to the "unknown" index (last entry).
PDG_VOCAB: list[int] = [11, -11, 13, -13, 211, -211, 321, -321, 2212, -2212, 22, 2112, -2112]
PDG_UNKNOWN_IDX: int = len(PDG_VOCAB)  # extra slot for unseen codes

PDG_EMBED_DIM: int = 8


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve requested device with safe CPU fallback when CUDA is unavailable."""
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _build_pdg_vocab(codes: np.ndarray) -> tuple[list[int], dict[int, int], int]:
    """Build a vocabulary from observed PDG codes, adding an unknown slot."""
    unique_codes = sorted(set(int(c) for c in codes))
    vocab = unique_codes
    pdg_to_idx = {code: i for i, code in enumerate(vocab)}
    unknown_idx = len(vocab)
    return vocab, pdg_to_idx, unknown_idx


def _encode_pdg(
    pdg_codes: Union[int, np.ndarray, list],
    pdg_to_idx: dict[int, int],
    unknown_idx: int,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Convert PDG integer code(s) to a LongTensor of embedding indices."""
    if isinstance(pdg_codes, (int, np.integer)):
        arr = np.array([int(pdg_codes)], dtype=np.int64)
    else:
        arr = np.asarray(pdg_codes, dtype=np.int64)
    idx = np.array([pdg_to_idx.get(int(c), unknown_idx) for c in arr], dtype=np.int64)
    return torch.from_numpy(idx).to(_resolve_device(device))


# ---------------------------------------------------------------------------
# Shared helpers — identical to wgan_gp_model.py
# ---------------------------------------------------------------------------

def _available_unit_circle_pairs(
    feature_index: dict[str, int],
) -> list[tuple[str, str]]:
    return [(s, c) for (s, c) in UNIT_CIRCLE_PAIRS if s in feature_index and c in feature_index]


def _apply_generation_bounds(
    samples: np.ndarray, clip_feature_indices: dict[str, int]
) -> np.ndarray:
    """Project sin/cos pairs to unit circle then clip hard bounds."""
    eps = 1e-8
    for sin_name, cos_name in UNIT_CIRCLE_PAIRS:
        sin_idx = clip_feature_indices.get(sin_name)
        cos_idx = clip_feature_indices.get(cos_name)
        if sin_idx is None or cos_idx is None:
            continue
        s = samples[:, sin_idx]
        c = samples[:, cos_idx]
        r = np.sqrt(s * s + c * c)
        r = np.where(r < eps, 1.0, r)
        samples[:, sin_idx] = s / r
        samples[:, cos_idx] = c / r

    clip_bounds = {
        "sin_phi_s": (-1.0, 1.0),
        "cos_phi_s": (-1.0, 1.0),
        "sin_theta": (-1.0, 1.0),
        "cos_theta": (-1.0, 1.0),
        "phi_p": (-np.pi, np.pi),
        "r": (0.0, np.inf),
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
    constraints,
    **context,
) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss = fake_data.new_tensor(0.0)
    metrics: dict[str, float] = {}
    if not constraints:
        return total_loss, metrics
    for constraint_fn in constraints:
        loss_value, constraint_metrics = constraint_fn(fake_data, **context)
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
    weight_value = float(weight)
    mean = mean.detach().clone()
    std = std.detach().clone()

    valid_pairs: list[tuple[str, str, int, int]] = []
    for sin_name, cos_name in UNIT_CIRCLE_PAIRS:
        if sin_name in feature_index and cos_name in feature_index:
            valid_pairs.append(
                (sin_name, cos_name, feature_index[sin_name], feature_index[cos_name])
            )

    if not valid_pairs:
        def _noop(fake_data: torch.Tensor, **_) -> tuple[torch.Tensor, dict[str, float]]:
            return fake_data.new_tensor(0.0), {}
        return _noop

    def _constraint(fake_data: torch.Tensor, **_) -> tuple[torch.Tensor, dict[str, float]]:
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


def _make_energy_constraint_conditional(
    feature_index: dict[str, int],
    weight: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    mass_by_idx: np.ndarray,      # shape: vocab_size
    target_e_by_idx: np.ndarray,  # shape: vocab_size
):
    weight_value = float(weight)
    mean = mean.detach().clone()
    std = std.detach().clone()

    momentum_feature_name = next((c for c in ENERGY_FEATURE_CANDIDATES if c in feature_index), None)
    if momentum_feature_name is None:
        def _noop(fake_data: torch.Tensor, **_) -> tuple[torch.Tensor, dict[str, float]]:
            return fake_data.new_tensor(0.0), {}
        return _noop

    momentum_idx = feature_index[momentum_feature_name]
    uses_log_momentum = momentum_feature_name == "log1p_p_mag"

    def _constraint(fake_data: torch.Tensor, pdg_idx: Optional[torch.Tensor] = None, **_):
        if pdg_idx is None:
            return fake_data.new_tensor(0.0), {}

        m = mean.to(fake_data.device)
        s = std.to(fake_data.device)

        p = fake_data[:, momentum_idx] * (s[momentum_idx] + 1e-8) + m[momentum_idx]
        if uses_log_momentum:
            p = torch.expm1(p)
        p = torch.clamp(p, min=0.0)

        mass_lut = torch.as_tensor(mass_by_idx, dtype=fake_data.dtype, device=fake_data.device)
        target_lut = torch.as_tensor(target_e_by_idx, dtype=fake_data.dtype, device=fake_data.device)

        mass_t = mass_lut[pdg_idx]
        e = torch.sqrt(p * p + mass_t * mass_t)

        # Per-PDG mean-matching (prevents per-sample collapse)
        unique_idx = torch.unique(pdg_idx)
        if unique_idx.numel() == 0:
            return fake_data.new_tensor(0.0), {}

        class_losses = []
        class_fake_means = []
        class_target_means = []
        for cls in unique_idx:
            mask = (pdg_idx == cls)
            e_cls_mean = e[mask].mean()
            t_cls = target_lut[cls]
            class_losses.append((e_cls_mean - t_cls) ** 2)
            class_fake_means.append(e_cls_mean)
            class_target_means.append(t_cls)

        penalty = torch.stack(class_losses).mean()
        fake_mean = torch.stack(class_fake_means).mean()
        target_mean = torch.stack(class_target_means).mean()

        return weight_value * penalty, {
            "energy_penalty": float(penalty.detach().item()),
            "energy_mean_fake": float(fake_mean.detach().item()),
            "energy_mean_target": float(target_mean.detach().item()),
        }

    return _constraint


def _clone_state_dict_to_cpu(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


# ---------------------------------------------------------------------------
# Dataset — identical to wgan_gp_model.py
# ---------------------------------------------------------------------------

class ParticleDataset(Dataset):
    """Robust-scaled (median / IQR) Dataset for particle physics data.

    The ``pdg`` column is kept separately as integer labels and is NOT included
    in the scaled feature tensor.
    """

    def __init__(self, feature_df, pdg_series: Optional[np.ndarray] = None):
        self.data = torch.FloatTensor(feature_df.values)

        self.center = self.data.median(dim=0).values
        q1 = torch.as_tensor(feature_df.quantile(0.25, numeric_only=False).to_numpy(), dtype=self.data.dtype)
        q3 = torch.as_tensor(feature_df.quantile(0.75, numeric_only=False).to_numpy(), dtype=self.data.dtype)
        self.scale = q3 - q1
        self.scale = torch.where(
            self.scale.abs() < 1e-8, torch.ones_like(self.scale), self.scale
        )

        # Backward-compatible aliases
        self.mean = self.center
        self.std = self.scale

        self.data = (self.data - self.center) / (self.scale + 1e-8)

        if pdg_series is not None:
            self.pdg = torch.from_numpy(np.asarray(pdg_series, dtype=np.int64))
        else:
            self.pdg = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.pdg is not None:
            return self.data[idx], self.pdg[idx]
        return self.data[idx], torch.tensor(-1, dtype=torch.int64)


# ---------------------------------------------------------------------------
# Conditional Generator + Critic
# ---------------------------------------------------------------------------

class CGenerator(nn.Module):
    """Generator conditioned on a PDG embedding.

    Input: latent z (latent_dim,) concatenated with PDG embedding (embed_dim,).
    Output: feature vector (output_dim,).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        output_dim: int = 8,
        hidden_dims: list[int] = hidden_dims_default,
        vocab_size: int = len(PDG_VOCAB) + 1,
        embed_dim: int = PDG_EMBED_DIM,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = []
        input_dim = latent_dim + embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, pdg_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(pdg_idx)          # (B, embed_dim)
        x = torch.cat([z, emb], dim=1)        # (B, latent_dim + embed_dim)
        return self.model(x)


class CCritic(nn.Module):
    """Critic conditioned on a PDG embedding.

    Input: feature vector (input_dim,) concatenated with PDG embedding (embed_dim,).
    Output: scalar score.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] = hidden_dims_default,
        vocab_size: int = len(PDG_VOCAB) + 1,
        embed_dim: int = PDG_EMBED_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = []
        feat_dim = input_dim + embed_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(feat_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            feat_dim = hidden_dim
        layers.append(nn.Linear(feat_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, pdg_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(pdg_idx)          # (B, embed_dim)
        inp = torch.cat([x, emb], dim=1)       # (B, input_dim + embed_dim)
        return self.model(inp)


# ---------------------------------------------------------------------------
# CWGAN_GP class
# ---------------------------------------------------------------------------

class CWGAN_GP:
    """Conditional WGAN-GP for particle physics data, conditioned on PDG code."""

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 128,
        device: Union[str, torch.device] = "cuda",
        lr_g: float = 5e-5,
        lr_c: float = 5e-5,
        vocab: Optional[list[int]] = None,
        pdg_to_idx: Optional[dict[int, int]] = None,
        unknown_idx: Optional[int] = None,
        embed_dim: int = PDG_EMBED_DIM,
        generator_constraints=None,
    ):
        self.device = _resolve_device(device)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.generator_constraints = list(generator_constraints) if generator_constraints else []
        self.clip_feature_indices: dict[str, int] = {}

        self.vocab = vocab or PDG_VOCAB
        self.pdg_to_idx = pdg_to_idx or {c: i for i, c in enumerate(self.vocab)}
        self.unknown_idx = unknown_idx if unknown_idx is not None else len(self.vocab)
        vocab_size = self.unknown_idx + 1  # includes unknown slot

        self.generator = CGenerator(
            latent_dim=latent_dim,
            output_dim=input_dim,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        ).to(self.device)

        self.critic = CCritic(
            input_dim=input_dim,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        ).to(self.device)

        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(0.0, 0.9)
        )
        self.c_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr_c, betas=(0.0, 0.9)
        )

    # ------------------------------------------------------------------
    def _encode(self, pdg_codes) -> torch.Tensor:
        return _encode_pdg(pdg_codes, self.pdg_to_idx, self.unknown_idx, self.device)

    # ------------------------------------------------------------------
    def _gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        pdg_idx: torch.Tensor,
        lambda_gp: float,
    ) -> torch.Tensor:
        batch_size = real_data.size(0)
        eps = torch.rand(batch_size, 1, device=self.device).expand_as(real_data)
        interpolated = (eps * real_data + (1 - eps) * fake_data).requires_grad_(True)
        interp_score = self.critic(interpolated, pdg_idx)
        grads = torch.autograd.grad(
            outputs=interp_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interp_score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = grads.view(batch_size, -1).norm(2, dim=1)
        return lambda_gp * ((grad_norm - 1) ** 2).mean()

    # ------------------------------------------------------------------
    def train_step(
        self,
        real_data: torch.Tensor,
        pdg_idx: torch.Tensor,
        n_critic: int = 5,
        lambda_gp: Optional[float] = None,
    ) -> tuple[float, float, dict]:
        if lambda_gp is None:
            lambda_gp = lambda_gp_var
        batch_size = real_data.size(0)

        critic_losses = []
        for _ in range(n_critic):
            self.c_optimizer.zero_grad()
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(z, pdg_idx).detach()
            real_score = self.critic(real_data, pdg_idx)
            fake_score = self.critic(fake_data, pdg_idx)
            gp = self._gradient_penalty(real_data, fake_data, pdg_idx, lambda_gp)
            c_loss = -(real_score.mean() - fake_score.mean()) + gp
            c_loss.backward()
            self.c_optimizer.step()
            critic_losses.append(c_loss.item())

        avg_c_loss = float(np.mean(critic_losses))

        self.g_optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(z, pdg_idx)
        g_adv_loss = -self.critic(fake_data, pdg_idx).mean()
        g_constraint_loss, constraint_metrics = _compute_generator_constraints(
            fake_data, self.generator_constraints, pdg_idx=pdg_idx
        )
        g_loss = g_adv_loss + g_constraint_loss
        g_loss.backward()
        self.g_optimizer.step()

        step_metrics = {
            "g_adv_loss": float(g_adv_loss.detach().item()),
            "g_constraint_loss": float(g_constraint_loss.detach().item()),
        }
        step_metrics.update(constraint_metrics)
        return avg_c_loss, g_loss.item(), step_metrics

    # ------------------------------------------------------------------
    def compute_validation_wasserstein(
        self,
        val_data,
        pdg_series,
        mean: torch.Tensor,
        std: torch.Tensor,
        n_samples: int = 10000,
    ) -> tuple[float, list[float]]:
        self.generator.eval()
        val_indices = np.random.choice(len(val_data), min(n_samples, len(val_data)), replace=False)
        val_samples = val_data.iloc[val_indices].values.astype(np.float32)
        val_pdg = pdg_series.iloc[val_indices].values if hasattr(pdg_series, "iloc") else pdg_series[val_indices]
        val_samples_norm = (val_samples - mean.numpy()) / (std.numpy() + 1e-8)

        with torch.no_grad():
            pdg_idx = self._encode(val_pdg)
            z = torch.randn(len(val_indices), self.latent_dim, device=self.device)
            synthetic = self.generator(z, pdg_idx).cpu().numpy()

        wasserstein_dists = [
            wasserstein_distance(val_samples_norm[:, i], synthetic[:, i])
            for i in range(self.input_dim)
        ]
        return float(np.mean(wasserstein_dists)), wasserstein_dists

    # ------------------------------------------------------------------
    def compute_validation_mmd(
        self,
        val_data,
        pdg_series,
        mean: torch.Tensor,
        std: torch.Tensor,
        n_samples: int = 10000,
        sigma="median",
        sigma_scale: float = 1.0,
        chunk_size: int = 1024,
        unbiased: bool = True,
        seed: int = 42,
    ) -> dict:
        self.generator.eval()
        val_indices = np.random.choice(len(val_data), min(n_samples, len(val_data)), replace=False)
        val_samples = val_data.iloc[val_indices].values.astype(np.float32)
        val_pdg = pdg_series.iloc[val_indices].values if hasattr(pdg_series, "iloc") else pdg_series[val_indices]
        val_samples_norm = (val_samples - mean.numpy()) / (std.numpy() + 1e-8)

        with torch.no_grad():
            pdg_idx = self._encode(val_pdg)
            z = torch.randn(len(val_indices), self.latent_dim, device=self.device)
            synthetic = self.generator(z, pdg_idx).cpu().numpy()

        return compute_mmd_rbf(
            val_samples_norm,
            synthetic,
            sigma=sigma,
            sigma_scale=sigma_scale,
            max_samples=n_samples,
            chunk_size=chunk_size,
            unbiased=unbiased,
            seed=seed,
        )

    # ------------------------------------------------------------------
    def generate(
        self,
        n_samples: int,
        pdg_code: Union[int, np.ndarray, list],
        mean: torch.Tensor,
        std: torch.Tensor,
        batch_size: int = 32768,
    ) -> np.ndarray:
        """Generate samples conditioned on pdg_code.

        Parameters
        ----------
        pdg_code : int or array-like
            Single int → broadcast to all n_samples.
            Array → must have length n_samples (per-sample conditioning).
        """
        self.generator.eval()
        mean_np = mean.detach().cpu().numpy() if torch.is_tensor(mean) else np.asarray(mean)
        std_np = std.detach().cpu().numpy() if torch.is_tensor(std) else np.asarray(std)

        # Build per-sample PDG code array
        if isinstance(pdg_code, (int, np.integer)):
            pdg_arr = np.full(n_samples, int(pdg_code), dtype=np.int64)
        else:
            pdg_arr = np.asarray(pdg_code, dtype=np.int64)
            if len(pdg_arr) != n_samples:
                raise ValueError(
                    f"pdg_code array length {len(pdg_arr)} != n_samples {n_samples}"
                )

        out_chunks = []
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                bs = min(batch_size, n_samples - start)
                z = torch.randn(bs, self.latent_dim, device=self.device)
                pdg_idx = self._encode(pdg_arr[start : start + bs])
                chunk = self.generator(z, pdg_idx).detach().cpu().numpy()
                chunk = chunk * (std_np + 1e-8) + mean_np
                chunk = _apply_generation_bounds(chunk, self.clip_feature_indices)
                out_chunks.append(chunk)

        return np.concatenate(out_chunks, axis=0)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_cwgan_gp(
    dataframe,
    epochs: int = 100,
    batch_size: int = 512,
    latent_dim: int = 128,
    device: str = "cuda",
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    num_workers: int = 1,
    log_interval: int = 10,
    patience: int = 25,
    min_delta: float = 0.0001,
    val_split: float = 0.2,
    mmd_sigma="median",
    mmd_sigma_scale: float = 1.0,
    mmd_chunk_size: int = 1024,
    mmd_unbiased: bool = True,
    mmd_seed: int = 42,
    lr_g: float = 5e-5,
    lr_c: float = 5e-5,
    trig_constraint_weight: float = 0.0,
    energy_constraint_weight: float = 0.01,
    energy_missing_policy: str = "warn_disable",
    mmd_eval_interval: int = 1,
    mmd_eval_samples: int = 20000,
    embed_dim: int = PDG_EMBED_DIM,
):
    """Train a conditional WGAN-GP on particle physics data.

    The ``dataframe`` **must** contain a ``"pdg"`` column with integer particle
    codes.  It is used for conditioning only and is **not** included in the
    feature vectors passed to the generator / critic.

    Returns
    -------
    (cwgan_gp, mean, std, history)
        Same shape as train_wgan_gp for drop-in compatibility.
    """
    if "pdg" not in dataframe.columns:
        raise ValueError(
            "train_cwgan_gp requires a 'pdg' column in dataframe. "
            "Ensure data_loader preserves the PDG code."
        )

    resolved_device = _resolve_device(device)

    # Separate PDG labels from feature columns
    pdg_col = dataframe["pdg"].astype(np.int64)
    feature_df = dataframe.drop(columns=["pdg"]).copy()

    # Build vocabulary from observed PDG codes in the whole dataset
    vocab, pdg_to_idx, unknown_idx = _build_pdg_vocab(pdg_col.values)
    print(f"PDG vocabulary ({len(vocab)} codes): {vocab}")

    # Train / validation split
    n_train = int(len(feature_df) * (1 - val_split))
    train_feat_df = feature_df.iloc[:n_train].copy()
    val_feat_df = feature_df.iloc[n_train:].copy()
    train_pdg = pdg_col.iloc[:n_train].reset_index(drop=True)
    val_pdg = pdg_col.iloc[n_train:].reset_index(drop=True)

    dataset = ParticleDataset(train_feat_df, pdg_series=train_pdg.values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    feature_index = {name: idx for idx, name in enumerate(feature_df.columns)}

    # Build generator constraints (same pattern as wgan_gp_model.py)
    generator_constraints = []

    if float(trig_constraint_weight) > 0.0:
        available_pairs = _available_unit_circle_pairs(feature_index)
        if not available_pairs:
            print(
                f"⚠️ Trig constraint disabled; no complete unit-circle pairs found "
                f"in columns={list(feature_df.columns)}"
            )
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

    # replace old global mass block with per-PDG tensors
    if float(energy_constraint_weight) > 0.0:
        momentum_feature_name = next((c for c in ENERGY_FEATURE_CANDIDATES if c in feature_index), None)
        if momentum_feature_name is None:
            if energy_missing_policy == "warn_disable":
                print("⚠️ Energy constraint disabled; missing momentum feature.")
            else:
                raise ValueError("Energy constraint disabled; missing momentum feature.")
        else:
            vocab_size = len(vocab) + 1  # + unknown
            mass_by_idx = np.full(vocab_size, DEFAULT_PARTICLE_MASS_GEV, dtype=np.float32)
            target_e_by_idx = np.zeros(vocab_size, dtype=np.float32)

            # compute train momentum in physical space
            pvals = train_feat_df[momentum_feature_name].to_numpy(dtype=np.float32)
            if momentum_feature_name == "log1p_p_mag":
                pvals = np.expm1(pvals)
            pvals = np.clip(pvals, 0.0, None)

            train_pdg_np = train_pdg.to_numpy(dtype=np.int64)
            global_target = float(np.mean(pvals)) if len(pvals) else 0.0

            for code, idx in pdg_to_idx.items():
                mass = float(PDG_MASS_GEV.get(abs(int(code)), DEFAULT_PARTICLE_MASS_GEV))
                mass_by_idx[idx] = mass
                mask = (train_pdg_np == int(code))
                if np.any(mask):
                    pm = pvals[mask]
                    target_e_by_idx[idx] = float(np.mean(np.sqrt(pm * pm + mass * mass)))
                else:
                    target_e_by_idx[idx] = global_target

            # unknown slot
            mass_by_idx[unknown_idx] = DEFAULT_PARTICLE_MASS_GEV
            target_e_by_idx[unknown_idx] = global_target

            generator_constraints.append(
                _make_energy_constraint_conditional(
                    feature_index=feature_index,
                    weight=energy_constraint_weight,
                    mean=dataset.mean,
                    std=dataset.std,
                    mass_by_idx=mass_by_idx,
                    target_e_by_idx=target_e_by_idx,
                )
            )

    missing_required_clip_features = [
        name for name in POSITIVE_CLIP_FEATURES if name not in feature_index
    ]
    if missing_required_clip_features:
        raise ValueError(
            f"Missing required generation clipping columns: {missing_required_clip_features}"
        )

    cwgan_gp = CWGAN_GP(
        input_dim=feature_df.shape[1],
        latent_dim=latent_dim,
        device=resolved_device,
        lr_g=lr_g,
        lr_c=lr_c,
        vocab=vocab,
        pdg_to_idx=pdg_to_idx,
        unknown_idx=unknown_idx,
        embed_dim=embed_dim,
        generator_constraints=generator_constraints,
    )

    # LR scheduler — identical policy to wgan_gp_model.py
    lr_decay_every = 5
    lr_decay_gamma = 0.9
    g_scheduler = optim.lr_scheduler.StepLR(
        cwgan_gp.g_optimizer, step_size=lr_decay_every, gamma=lr_decay_gamma
    )
    c_scheduler = optim.lr_scheduler.StepLR(
        cwgan_gp.c_optimizer, step_size=lr_decay_every, gamma=lr_decay_gamma
    )

    cwgan_gp.clip_feature_indices = {
        name: feature_index[name]
        for name in BOUNDED_CLIP_FEATURES
        if name in feature_index
    }

    print(f"Training cWGAN-GP for {epochs} epochs on {resolved_device}...")
    print(f"Train: {len(train_feat_df)} samples, Validation: {len(val_feat_df)} samples")

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
        "val_mmd": [],
    }

    best_val_wd = float("inf")
    best_val_mmd = float("inf")
    best_epoch = None
    best_gen_state = None
    best_critic_state = None
    patience_counter = 0
    warmup_epochs = 15

    for epoch in range(epochs):
        c_losses, g_losses = [], []
        g_constraint_losses, trig_penalties, energy_penalties = [], [], []

        for batch_feat, batch_pdg in dataloader:
            batch_feat = batch_feat.to(resolved_device, non_blocking=True)
            pdg_idx = _encode_pdg(batch_pdg.detach().cpu().numpy(), pdg_to_idx, unknown_idx, resolved_device)

            c_loss, g_loss, step_metrics = cwgan_gp.train_step(
                batch_feat, pdg_idx, n_critic=n_critic, lambda_gp=lambda_gp
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

        train_wd, _ = cwgan_gp.compute_validation_wasserstein(
            train_feat_df, train_pdg, dataset.mean, dataset.std, n_samples=5000
        )
        val_wd, _ = cwgan_gp.compute_validation_wasserstein(
            val_feat_df, val_pdg, dataset.mean, dataset.std, n_samples=5000
        )

        do_mmd_eval = (
            (epoch == 0)
            or ((epoch + 1) % int(mmd_eval_interval) == 0)
            or (epoch + 1 == epochs)
        )
        if do_mmd_eval:
            train_mmd_metrics = cwgan_gp.compute_validation_mmd(
                train_feat_df, train_pdg, dataset.mean, dataset.std,
                n_samples=int(mmd_eval_samples),
                sigma=mmd_sigma, sigma_scale=mmd_sigma_scale,
                chunk_size=mmd_chunk_size, unbiased=mmd_unbiased,
                seed=mmd_seed + epoch,
            )
            val_mmd_metrics = cwgan_gp.compute_validation_mmd(
                val_feat_df, val_pdg, dataset.mean, dataset.std,
                n_samples=int(mmd_eval_samples),
                sigma=mmd_sigma, sigma_scale=mmd_sigma_scale,
                chunk_size=mmd_chunk_size, unbiased=mmd_unbiased,
                seed=mmd_seed + 10000 + epoch,
            )
        else:
            train_mmd_metrics = {"mmd": history["train_mmd"][-1] if history["train_mmd"] else float("nan")}
            val_mmd_metrics = {"mmd": history["val_mmd"][-1] if history["val_mmd"] else float("nan")}

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

        prev_g_lr = float(cwgan_gp.g_optimizer.param_groups[0]["lr"])
        prev_c_lr = float(cwgan_gp.c_optimizer.param_groups[0]["lr"])
        g_scheduler.step()
        c_scheduler.step()
        curr_g_lr = float(cwgan_gp.g_optimizer.param_groups[0]["lr"])
        curr_c_lr = float(cwgan_gp.c_optimizer.param_groups[0]["lr"])

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

        current_val_wd = val_wd
        current_val_mmd = val_mmd_metrics["mmd"]

        if current_val_wd < best_val_wd - min_delta:
            print(f"→ New best Val WD: {current_val_wd:.6f} (Val MMD: {current_val_mmd:.4f})")
            best_val_wd = current_val_wd
            best_val_mmd = current_val_mmd
            best_epoch = epoch + 1
            best_gen_state = _clone_state_dict_to_cpu(cwgan_gp.generator)
            best_critic_state = _clone_state_dict_to_cpu(cwgan_gp.critic)
            patience_counter = 0
        elif epoch >= warmup_epochs:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n⏹️ Early stopping at epoch {epoch+1}: "
                    f"Val WD no improvement for {patience} epochs"
                )
                break

    if best_gen_state is not None and best_critic_state is not None:
        cwgan_gp.generator.load_state_dict(best_gen_state)
        cwgan_gp.critic.load_state_dict(best_critic_state)
        print(
            f"✅ Restored best checkpoint from epoch {best_epoch} "
            f"(Val WD={best_val_wd:.6f}, Val MMD={best_val_mmd:.4f})"
        )

    history["best_epoch"] = best_epoch
    history["best_val_wasserstein"] = best_val_wd
    history["best_val_mmd"] = best_val_mmd
    history["mmd_config"] = {
        "sigma": mmd_sigma,
        "sigma_scale": float(mmd_sigma_scale),
        "chunk_size": int(mmd_chunk_size),
        "unbiased": bool(mmd_unbiased),
        "seed": int(mmd_seed),
    }
    history["constraint_config"] = {
        "trig_constraint_weight": float(trig_constraint_weight),
        "energy_constraint_weight": float(energy_constraint_weight),
        "energy_missing_policy": str(energy_missing_policy),
    }
    history["lr_scheduler_config"] = {
        "type": "StepLR",
        "decay_every": int(lr_decay_every),
        "gamma": float(lr_decay_gamma),
    }
    history["pdg_vocab"] = vocab

    return cwgan_gp, dataset.mean, dataset.std, history
