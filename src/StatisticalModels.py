"""
statistical_model.py
====================
PyTorch statistical / neural-operator model for latent force estimation.

Architecture menu
-----------------
Each architecture is implemented as a separate ``nn.Module`` and registered
in ``NeuralOperatorRegistry``.  The ``LatentForceModel`` class wraps any of
them into a common training / inference interface.

Available operators
~~~~~~~~~~~~~~~~~~~
- ``FNO``     : Fourier Neural Operator  (Li et al. 2021)
- ``DeepONet``: Deep Operator Network    (Lu et al. 2021)
- ``UNet``    : U-Net style convolution operator
- ``MLP``     : Plain MLP baseline (no inductive bias)

The class also supports:
- Uncertainty quantification via MC-Dropout or deep ensembles
- Observation likelihood conditioning  p(u_obs | f)
- Gradient-based force field update   ∇_f  ‖u(f) - u_obs‖²
"""

from __future__ import annotations

import abc
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseNeuralOperator(nn.Module, abc.ABC):
    """
    Abstract base for all Neural Operator architectures.

    Subclasses must implement ``forward(x) -> Tensor`` where
    x  ∈ R^{B × C_in × *spatial}
    out ∈ R^{B × C_out × *spatial}
    """

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Fourier Neural Operator (FNO)
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """Complex-valued spectral convolution in 2-D Fourier space."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, x: Tensor, w: Tensor) -> Tensor:
        # (B, in_c, h, w) × (in_c, out_c, h, w) → (B, out_c, h, w)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOBlock(nn.Module):
    """One FNO layer: spectral conv + pointwise residual."""

    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNO(BaseNeuralOperator):
    """
    Fourier Neural Operator for 2-D spatial fields.

    Input  : (B, in_channels,  H, W)
    Output : (B, out_channels, H, W)

    Parameters
    ----------
    in_channels, out_channels : int
    hidden_channels : int     — width of lifted feature maps
    n_layers : int            — number of FNO blocks
    modes1, modes2 : int      — number of Fourier modes retained per dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[FNOBlock(hidden_channels, modes1, modes2) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.lift(x)
        x = self.blocks(x)
        return self.project(x)


# ---------------------------------------------------------------------------
# DeepONet
# ---------------------------------------------------------------------------

class BranchNet(nn.Module):
    """Branch network: encodes the function input (sensor values)."""

    def __init__(self, n_sensors: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, u_sensors: Tensor) -> Tensor:
        return self.net(u_sensors)  # (B, latent_dim)


class TrunkNet(nn.Module):
    """Trunk network: encodes query locations."""

    def __init__(self, coord_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, coords: Tensor) -> Tensor:
        return self.net(coords)  # (N_query, latent_dim)


class DeepONet(BaseNeuralOperator):
    """
    Deep Operator Network (unstacked version).

    Forward signature differs from grid-based operators:
        u_sensors : (B, n_sensors)          — observed field at sensor locations
        coords    : (N_query, coord_dim)    — query coordinates

    Returns (B, N_query) output values.
    """

    def __init__(
        self,
        n_sensors: int = 64,
        coord_dim: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        out_channels: int = 1,
    ):
        super().__init__()
        self.branch = BranchNet(n_sensors, hidden_dim, latent_dim)
        self.trunk = TrunkNet(coord_dim, hidden_dim, latent_dim)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, u_sensors: Tensor, coords: Tensor) -> Tensor:
        b = self.branch(u_sensors)          # (B, latent_dim)
        t = self.trunk(coords)              # (N_q, latent_dim)
        # Outer product sum: b[i,:] · t[j,:] for all i,j
        out = torch.einsum("bd,nd->bn", b, t) + self.bias
        return out   # (B, N_query)


# ---------------------------------------------------------------------------
# U-Net operator (grid-based, local inductive bias)
# ---------------------------------------------------------------------------

class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(min(8, out_ch), out_ch), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(BaseNeuralOperator):
    """
    U-Net operator.  Encodes strong local/multiscale spatial inductive bias.

    Input/Output : (B, in_channels, H, W) — H and W must be divisible by 2^depth.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.depth = depth
        ch = [min(base_channels * (2 ** i), 256) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Encoder path
        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(UNetBlock(in_ch, ch[i]))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch[i]

        # Bottleneck
        self.bottleneck = UNetBlock(ch[depth - 1], ch[depth])

        # Decoder path
        for i in reversed(range(depth)):
            self.upsamples.append(nn.ConvTranspose2d(ch[i + 1], ch[i], kernel_size=2, stride=2))
            self.decoders.append(UNetBlock(ch[i] * 2, ch[i]))

        self.head = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upsample, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)


# ---------------------------------------------------------------------------
# MLP baseline (no spatial inductive bias)
# ---------------------------------------------------------------------------

class MLP(BaseNeuralOperator):
    """
    Plain MLP baseline — operates pointwise on flattened field.

    No spatial inductive bias.  Useful as a lower bound on estimation quality.
    """

    def __init__(
        self,
        n_dof: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        in_channels: int = 3,
        out_channels: int = 1,
    ):
        super().__init__()
        in_dim = n_dof * in_channels
        out_dim = n_dof * out_channels
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.n_dof = n_dof
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        return self.net(x.view(B, -1)).view(B, self.out_channels, self.n_dof)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class NeuralOperatorRegistry:
    """Simple registry mapping string names to constructor callables."""

    _registry: Dict[str, type] = {
        "fno": FNO,
        "deeponet": DeepONet,
        "unet": UNet,
        "mlp": MLP,
    }

    @classmethod
    def build(cls, name: str, **kwargs) -> BaseNeuralOperator:
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"Unknown operator '{name}'. Available: {list(cls._registry)}")
        return cls._registry[key](**kwargs)

    @classmethod
    def register(cls, name: str, klass: type) -> None:
        cls._registry[name.lower()] = klass


# ---------------------------------------------------------------------------
# Uncertainty wrapper  (MC-Dropout)
# ---------------------------------------------------------------------------

class MCDropoutWrapper(nn.Module):
    """
    Wraps any ``BaseNeuralOperator`` and enables MC-Dropout at inference.

    During ``forward``, Dropout layers are kept active regardless of
    ``model.eval()`` mode, producing stochastic outputs used to estimate
    predictive uncertainty.
    """

    def __init__(self, base_model: BaseNeuralOperator, dropout_rate: float = 0.1):
        super().__init__()
        self.model = base_model
        self.dropout_rate = dropout_rate
        self._inject_dropout(self.model, dropout_rate)

    @staticmethod
    def _inject_dropout(module: nn.Module, p: float) -> None:
        """Append Dropout after every GELU/ReLU/Tanh activation in place."""
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.GELU, nn.ReLU, nn.Tanh)):
                setattr(module, name, nn.Sequential(child, nn.Dropout(p=p)))
            else:
                MCDropoutWrapper._inject_dropout(child, p)

    def forward(self, *args, **kwargs) -> Tensor:
        self.model.train()          # keep dropout active
        return self.model(*args, **kwargs)

    def predict_with_uncertainty(
        self,
        *args,
        n_samples: int = 50,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Run ``n_samples`` stochastic forward passes and return
        ``(mean, std)`` estimates.
        """
        samples = torch.stack([self(*args, **kwargs) for _ in range(n_samples)], dim=0)
        return samples.mean(0), samples.std(0)


# ---------------------------------------------------------------------------
# Latent Force Model (main estimation class)
# ---------------------------------------------------------------------------

class LatentForceModel(nn.Module):
    """
    Wraps a Neural Operator and exposes:
    - ``estimate_force``     : predict forcing field f̂ from observed state u_obs
    - ``update_with_obs``    : gradient step to fit f̂ s.t. u(f̂) ≈ u_obs
    - ``uncertainty``        : MC-Dropout or ensemble uncertainty in f̂
    - ``log_likelihood``     : Gaussian observation log-likelihood

    Parameters
    ----------
    operator_name : str
        Architecture key in ``NeuralOperatorRegistry``.
    operator_kwargs : dict
        Keyword arguments forwarded to the operator constructor.
    obs_noise_std : float
        Assumed observation noise σ for the Gaussian likelihood.
    use_mc_dropout : bool
        Wrap the operator in ``MCDropoutWrapper`` for uncertainty estimation.
    """

    def __init__(
        self,
        operator_name: str = "fno",
        operator_kwargs: Optional[Dict] = None,
        obs_noise_std: float = 0.01,
        use_mc_dropout: bool = True,
        mc_dropout_rate: float = 0.1,
    ):
        super().__init__()
        op_kwargs = operator_kwargs or {}
        base_op = NeuralOperatorRegistry.build(operator_name, **op_kwargs)

        if use_mc_dropout:
            self.operator = MCDropoutWrapper(base_op, dropout_rate=mc_dropout_rate)
        else:
            self.operator = base_op

        self.obs_noise_std = obs_noise_std
        self.operator_name = operator_name

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Direct pass through the neural operator.

        x : (B, C_in, *spatial)  — e.g. concatenation of [u, coords, t]
        """
        return self.operator(x)

    # ------------------------------------------------------------------
    # Force estimation
    # ------------------------------------------------------------------

    def estimate_force(self, u_obs: Tensor, coords: Optional[Tensor] = None) -> Tensor:
        """
        Estimate latent forcing field from observed state u_obs.

        Parameters
        ----------
        u_obs  : (B, 1, H, W)   observed field (grid-based operators)
                 or (B, n_sensors) for DeepONet
        coords : (N_query, 2)   query coords  (DeepONet only)

        Returns
        -------
        Tensor  : estimated forcing f̂
        """
        if self.operator_name == "deeponet":
            assert coords is not None, "DeepONet requires query coordinates."
            return self.operator(u_obs, coords)
        return self.operator(u_obs)

    def estimate_force_with_uncertainty(
        self,
        u_obs: Tensor,
        coords: Optional[Tensor] = None,
        n_samples: int = 50,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return ``(mean_f, std_f)`` via MC-Dropout.

        Requires the model to have been built with ``use_mc_dropout=True``.
        """
        if not isinstance(self.operator, MCDropoutWrapper):
            raise RuntimeError("Build model with use_mc_dropout=True to use uncertainty.")
        if self.operator_name == "deeponet":
            assert coords is not None
            return self.operator.predict_with_uncertainty(u_obs, coords, n_samples=n_samples)
        return self.operator.predict_with_uncertainty(u_obs, n_samples=n_samples)

    # ------------------------------------------------------------------
    # Observation likelihood
    # ------------------------------------------------------------------

    def log_likelihood(self, u_pred: Tensor, u_obs: Tensor) -> Tensor:
        """
        Gaussian log-likelihood:  log p(u_obs | u_pred) = -‖u_obs - u_pred‖² / (2σ²) + const.

        Parameters
        ----------
        u_pred : predicted field (B, 1, H, W) or (B, N)
        u_obs  : observed field  (same shape as u_pred)

        Returns
        -------
        Tensor : scalar log-likelihood
        """
        sigma = self.obs_noise_std
        diff = u_pred - u_obs
        log_p = -0.5 * (diff / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)
        return log_p.sum()

    def mse_loss(self, u_pred: Tensor, u_obs: Tensor) -> Tensor:
        return F.mse_loss(u_pred, u_obs)

    # ------------------------------------------------------------------
    # Gradient-based force update
    # ------------------------------------------------------------------

    def observation_fit_loss(
        self,
        f_hat: Tensor,
        u_obs: Tensor,
        pde_forward: Optional[callable] = None,
    ) -> Tensor:
        """
        Compute the observation-fitting loss:

            L(f̂) = ‖u(f̂) - u_obs‖²

        where ``pde_forward(f)`` maps a forcing field to the resulting PDE
        solution u.  If ``pde_forward`` is None, the operator itself is used
        as an approximate surrogate.

        Parameters
        ----------
        f_hat       : (B, 1, H, W) current forcing estimate
        u_obs       : (B, 1, H, W) observed field
        pde_forward : callable  f → u  (optional)

        Returns
        -------
        Tensor : scalar loss
        """
        if pde_forward is not None:
            u_pred = pde_forward(f_hat)
        else:
            u_pred = f_hat   # dummy when PDE surrogate unavailable
        return self.mse_loss(u_pred, u_obs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        op = self.operator.model if isinstance(self.operator, MCDropoutWrapper) else self.operator
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"LatentForceModel\n"
            f"  operator       : {self.operator_name.upper()}\n"
            f"  trainable params: {n_params:,}\n"
            f"  mc_dropout     : {isinstance(self.operator, MCDropoutWrapper)}\n"
            f"  obs_noise_std  : {self.obs_noise_std}\n"
        )


# ---------------------------------------------------------------------------
# Deep Ensemble wrapper
# ---------------------------------------------------------------------------

class DeepEnsemble(nn.Module):
    """
    Ensemble of ``LatentForceModel`` instances for uncertainty estimation.

    Each member is trained independently from a different random seed.

    Parameters
    ----------
    n_members : int
        Number of ensemble members.
    **model_kwargs :
        Forwarded to each ``LatentForceModel``.
    """

    def __init__(self, n_members: int = 5, **model_kwargs):
        super().__init__()
        self.members = nn.ModuleList(
            [LatentForceModel(**model_kwargs) for _ in range(n_members)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Run all ensemble members and return ``(mean, std)`` over members.
        """
        preds = torch.stack([m(x) for m in self.members], dim=0)
        return preds.mean(0), preds.std(0)

    def estimate_force_with_uncertainty(
        self, u_obs: Tensor, coords: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        preds = torch.stack(
            [m.estimate_force(u_obs, coords) for m in self.members], dim=0
        )
        return preds.mean(0), preds.std(0)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class Trainer:
    """
    Minimal training loop for ``LatentForceModel`` (or ``DeepEnsemble``).

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    device : str
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.history: List[float] = []

    def train_step(self, x: Tensor, y: Tensor) -> float:
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(
        self, dataloader: torch.utils.data.DataLoader, verbose: bool = False
    ) -> float:
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            total_loss += self.train_step(batch_x, batch_y)
        mean_loss = total_loss / len(dataloader)
        self.history.append(mean_loss)
        if verbose:
            print(f"  epoch loss: {mean_loss:.4e}")
        return mean_loss

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_epochs: int = 100,
        verbose: bool = True,
    ) -> List[float]:
        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(dataloader, verbose=False)
            if verbose and epoch % max(1, n_epochs // 10) == 0:
                print(f"Epoch {epoch:4d}/{n_epochs}  loss={loss:.4e}")
        return self.history