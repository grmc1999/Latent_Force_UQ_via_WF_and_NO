"""
latent_force_estimation.py
==========================
Top-level orchestrator for the latent force estimation framework.

Ties together:
  ┌──────────────────────────────────────────────────────────────┐
  │  ArtificialForceGenerator  →  ObservationGenerator           │
  │          ↓                          ↓                        │
  │   Ground-truth f(x,t)       Noisy u_obs(x,t)                 │
  │                                     ↓                        │
  │              LatentForceEstimator (Neural Operator)           │
  │                         ↓                                    │
  │           f̂(x,t)  +  σ̂_f(x,t)  (mean + uncertainty)       │
  │                         ↓                                    │
  │              PDE residual ‖u(f̂) - u_obs‖²                   │
  └──────────────────────────────────────────────────────────────┘

Quick-start example
-------------------
>>> from latent_force_estimation import LatentForceEstimationPipeline
>>> pipe = LatentForceEstimationPipeline.from_config(cfg)
>>> pipe.generate_data(n_realisations=100, n_steps=50)
>>> pipe.train(n_epochs=200)
>>> f_mean, f_std = pipe.predict(u_obs)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.StatisticalModels import Trainer, LatentForceModel

from src.ForceGenerator import (
    BaseForceGenerator,
    GaussianBumpForce,
    SinusoidalForce,
    RandomFieldForce,
    LocalisedPulseForce,
    CompositeForce,
    ObservationGenerator,
    make_pytorch_dataset,
)
from src.Solvers import ImplicitDiffusionStepper, MeshFactory
import firedrake as fd

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Unified configuration for the latent force estimation pipeline.

    PDE settings
    ------------
    mesh_nx, mesh_ny   : mesh resolution
    dt                 : time step
    T                  : total simulation time
    pde_type           : 'diffusion' | 'advection_diffusion' | 'reaction_diffusion'
    diffusivity        : κ
    advection_velocity : [vx, vy] or None
    reaction_rate      : r (reaction-diffusion only)
    theta              : time integration parameter  (0.5 = CN)

    Force generator
    ---------------
    force_type  : 'gaussian' | 'sinusoidal' | 'random_field' | 'pulse' | 'composite'
    force_kwargs: dict forwarded to the generator constructor

    Neural Operator
    ---------------
    operator_name    : 'fno' | 'deeponet' | 'unet' | 'mlp'
    operator_kwargs  : dict forwarded to the operator constructor
    use_mc_dropout   : bool
    mc_dropout_rate  : float
    use_ensemble     : bool
    n_ensemble       : int
    obs_noise_std    : float

    Training
    --------
    n_realisations   : int
    n_steps_per_real : int
    output_every     : int
    batch_size       : int
    n_epochs         : int
    learning_rate    : float
    weight_decay     : float
    device           : 'cpu' | 'cuda'

    Grid
    ----
    grid_shape       : [H, W]  for reshaping DOF arrays to 2-D tensors
    """

    # PDE
    mesh_nx: int = 32
    mesh_ny: int = 32
    dt: float = 0.01
    T: float = 1.0
    pde_type: str = "diffusion"
    diffusivity: float = 0.01
    advection_velocity: Optional[List[float]] = None
    reaction_rate: float = 0.0
    theta: float = 0.5

    # Force generator
    force_type: str = "gaussian"
    force_kwargs: Dict = field(default_factory=dict)

    # Neural operator
    operator_name: str = "fno"
    operator_kwargs: Dict = field(default_factory=dict)
    use_mc_dropout: bool = True
    mc_dropout_rate: float = 0.1
    use_ensemble: bool = False
    n_ensemble: int = 5
    obs_noise_std: float = 0.005

    # Training
    n_realisations: int = 50
    n_steps_per_real: int = 50
    output_every: int = 1
    batch_size: int = 16
    n_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"

    # Grid shape for tensor conversion
    grid_shape: List[int] = field(default_factory=lambda: [32, 33])

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PipelineConfig":
        with open(path) as fh:
            return cls(**json.load(fh))


# ---------------------------------------------------------------------------
# Force generator factory
# ---------------------------------------------------------------------------

class ForceGeneratorFactory:
    """Build a ``BaseForceGenerator`` from a config entry."""

    _builders = {
        "gaussian":     GaussianBumpForce.random,
        "sinusoidal":   SinusoidalForce.random,
        "pulse":        LocalisedPulseForce,
    }

    @classmethod
    def build(cls, force_type: str, spatial_dim: int = 2, **kwargs) -> BaseForceGenerator:
        key = force_type.lower()
        if key == "random_field":
            # Requires coords — deferred; returns a partial factory
            return None   # handled separately in pipeline
        if key == "composite":
            generators = [
                cls.build("gaussian", spatial_dim=spatial_dim),
                cls.build("sinusoidal", spatial_dim=spatial_dim),
            ]
            return CompositeForce(generators)
        if key not in cls._builders:
            raise ValueError(f"Unknown force type: '{force_type}'. "
                             f"Available: {list(cls._builders) + ['random_field', 'composite']}")
        return cls._builders[key](spatial_dim=spatial_dim, **kwargs)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class LatentForceEstimationPipeline:
    """
    End-to-end pipeline for latent force estimation.

    Workflow
    --------
    1. ``generate_data()``  — run PDE with true forcing, collect observations
    2. ``train()``          — fit Neural Operator to (u_obs → f) pairs
    3. ``predict()``        — estimate f̂ and uncertainty σ̂_f for new observations
    4. ``evaluate()``       — compute quantitative metrics vs ground truth

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._solver = None
        self._force_gen: Optional[BaseForceGenerator] = None
        self._obs_gen: Optional[ObservationGenerator] = None
        self._model: Optional[nn.Module] = None
        self._trainer: Optional[Trainer] = None
        self._dataset: Optional[List[Dict]] = None
        self._X: Optional["torch.Tensor"] = None
        self._Y: Optional["torch.Tensor"] = None

    # ------------------------------------------------------------------
    # Solver initialisation
    # ------------------------------------------------------------------

    def _init_solver(self):
        """Lazily build the Firedrake PDE solver."""

        mesh = MeshFactory.unit_square(self.cfg.mesh_nx, self.cfg.mesh_ny)
        adv = tuple(self.cfg.advection_velocity) if self.cfg.advection_velocity else None

        self._solver = ImplicitDiffusionStepper(
            mesh=mesh,
            dt=self.cfg.dt,
            #theta=self.cfg.theta,
            diffusivity=self.cfg.diffusivity,
            #advection_velocity=adv,
            #pde_type=self.cfg.pde_type,
            #reaction_rate=self.cfg.reaction_rate,
        )

        # Zero Dirichlet on all boundaries by default # TODO
        #self._solver.set_boundary_conditions({1: fd.Constant(0.0), 2: fd.Constant(0.0),
        #                                      3: fd.Constant(0.0), 4: fd.Constant(0.0)})
        #self._solver.set_initial_condition(fd.Constant(0.0))

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_data(
        self,
        n_realisations: Optional[int] = None,
        n_steps: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> "LatentForceEstimationPipeline":
        """
        Run the PDE forward model and collect (f, u_obs) pairs.

        Parameters
        ----------
        n_realisations : int  (defaults to config value)
        n_steps        : int  (defaults to config value)
        seed           : int  random seed
        verbose        : bool print progress
        """
        n_real = n_realisations or self.cfg.n_realisations
        n_st = n_steps or self.cfg.n_steps_per_real

        self._init_solver()
        #coords = self._solver.get_dof_coordinates()

        # Build force generator
        #if self.cfg.force_type == "random_field":
        #    self._force_gen = RandomFieldForce(
        #        coords, seed=seed, **self.cfg.force_kwargs
        #    )
        #else:
        self._force_gen = ForceGeneratorFactory.build(
            self.cfg.force_type, **self.cfg.force_kwargs
        )

        self._obs_gen = ObservationGenerator(
            pde_solver=self._solver,
            force_generator=self._force_gen,
            obs_noise_std=self.cfg.obs_noise_std,
        )

        if verbose:
            print(f"[data] Generating {n_real} realisations × {n_st} steps ...")
        t0 = time.perf_counter()
        self._dataset = self._obs_gen.generate_batch(
            n_realisations=n_real,
            n_steps=n_st,
            output_every=self.cfg.output_every,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[data] Done in {elapsed:.1f}s — "
                  f"{len(self._dataset)} records, "
                  f"{self._dataset[0]['observations'].shape} obs shape")

        # Convert to PyTorch tensors
        grid = tuple(self.cfg.grid_shape)
        self._X, self._Y = make_pytorch_dataset(self._dataset, grid_shape=grid)
        if verbose:
            print(f"[data] X shape: {tuple(self._X.shape)}  Y shape: {tuple(self._Y.shape)}")

        return self

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Construct Neural Operator model according to config."""
        model_kwargs = dict(
            operator_name=self.cfg.operator_name,
            operator_kwargs=self.cfg.operator_kwargs,
            obs_noise_std=self.cfg.obs_noise_std,
            use_mc_dropout=self.cfg.use_mc_dropout,
            mc_dropout_rate=self.cfg.mc_dropout_rate,
        )
        if self.cfg.use_ensemble:
            return DeepEnsemble(n_members=self.cfg.n_ensemble, **model_kwargs)
        return LatentForceModel(**model_kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        n_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> "LatentForceEstimationPipeline":
        """
        Train the Neural Operator on the generated (u_obs, f) dataset.

        Parameters
        ----------
        n_epochs : int  (defaults to config value)
        verbose  : bool
        """
        if self._X is None:
            raise RuntimeError("Call generate_data() before train().")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training.")

        epochs = n_epochs or self.cfg.n_epochs
        device = self.cfg.device

        self._model = self._build_model()
        if verbose:
            if hasattr(self._model, "summary"):
                print(self._model.summary())
            n_p = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            print(f"[model] Trainable parameters: {n_p:,}")

        optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        dataset = TensorDataset(self._X, self._Y)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        self._trainer = Trainer(self._model, optimizer, device=device)

        if verbose:
            print(f"[train] Starting training for {epochs} epochs ...")
        t0 = time.perf_counter()
        self._trainer.fit(loader, n_epochs=epochs, verbose=verbose)
        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[train] Finished in {elapsed:.1f}s")

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        u_obs: "torch.Tensor",
        return_uncertainty: bool = True,
        n_mc_samples: int = 50,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Estimate the latent forcing field from observations.

        Parameters
        ----------
        u_obs : (B, C, H, W) observation tensor
        return_uncertainty : bool — also return σ̂_f
        n_mc_samples : int  — MC-Dropout samples for uncertainty

        Returns
        -------
        (f_mean, f_std)   where f_std is None if uncertainty not requested
        """
        if self._model is None:
            raise RuntimeError("Call train() before predict().")
        self._model.eval()
        device = self.cfg.device
        u_obs = u_obs.to(device)

        with torch.no_grad():
            if return_uncertainty:
                if isinstance(self._model, DeepEnsemble):
                    f_mean, f_std = self._model.estimate_force_with_uncertainty(u_obs)
                elif isinstance(self._model, LatentForceModel):
                    f_mean, f_std = self._model.estimate_force_with_uncertainty(
                        u_obs, n_samples=n_mc_samples
                    )
                else:
                    f_mean = self._model(u_obs)
                    f_std = None
                return f_mean, f_std
            else:
                f_mean = self._model(u_obs)
                return f_mean, None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: "torch.Tensor",
        Y_test: "torch.Tensor",
        n_mc_samples: int = 50,
    ) -> Dict[str, float]:
        """
        Compute quantitative metrics on a held-out test set.

        Returns
        -------
        dict with keys:
          'rmse'         : root-mean-square error on forcing estimate
          'relative_l2'  : relative L² error ‖f̂ - f‖ / ‖f‖
          'mean_calib'   : mean calibration (RMSE / mean σ̂_f)
          'nll'          : negative log-likelihood under Gaussian noise model
        """
        import torch.nn.functional as F

        f_pred, f_std = self.predict(X_test, return_uncertainty=True, n_mc_samples=n_mc_samples)
        Y_test = Y_test.to(self.cfg.device)

        diff = f_pred - Y_test
        rmse = diff.pow(2).mean().sqrt().item()
        rel_l2 = (diff.pow(2).sum().sqrt() / Y_test.pow(2).sum().sqrt()).item()

        metrics = {"rmse": rmse, "relative_l2": rel_l2}

        if f_std is not None:
            mean_sigma = f_std.mean().item()
            metrics["mean_uncertainty"] = mean_sigma
            metrics["mean_calib"] = rmse / (mean_sigma + 1e-12)
            # Gaussian NLL
            nll = (0.5 * ((diff / (f_std + 1e-8)) ** 2 + torch.log(f_std + 1e-8))).mean().item()
            metrics["nll"] = nll

        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """Save model weights and config to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.cfg.save(directory / "config.json")
        if self._model is not None:
            import torch
            torch.save(self._model.state_dict(), directory / "model.pt")
        print(f"[save] Saved to {directory}")

    def load(self, directory: Union[str, Path]) -> "LatentForceEstimationPipeline":
        """Load model weights from *directory*."""
        directory = Path(directory)
        import torch
        self.cfg = PipelineConfig.load(directory / "config.json")
        self._model = self._build_model()
        self._model.load_state_dict(torch.load(directory / "model.pt", map_location=self.cfg.device))
        print(f"[load] Loaded from {directory}")
        return self

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Union[PipelineConfig, Dict]) -> "LatentForceEstimationPipeline":
        if isinstance(config, dict):
            config = PipelineConfig(**config)
        return cls(config)

    @classmethod
    def from_config_file(cls, path: Union[str, Path]) -> "LatentForceEstimationPipeline":
        return cls(PipelineConfig.load(path))


# ---------------------------------------------------------------------------
# Experiment runner  (CLI-friendly)
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Runs a systematic comparison of Neural Operator inductive biases.

    Trains one pipeline per operator type on the *same* synthetic dataset
    and reports comparative metrics.

    Parameters
    ----------
    base_config : PipelineConfig
        Config template; operator_name is overridden per run.
    operator_names : list[str]
        Operators to benchmark.
    """

    def __init__(
        self,
        base_config: PipelineConfig,
        operator_names: Optional[List[str]] = None,
    ):
        self.base_config = base_config
        self.operator_names = operator_names or ["fno", "unet", "mlp"]
        self.results: Dict[str, Dict] = {}

    def run(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Run the full benchmark.

        Returns
        -------
        dict  operator_name → metric_dict
        """
        import copy

        # Generate a shared dataset once
        print("=" * 60)
        print("ExperimentRunner: generating shared dataset")
        print("=" * 60)
        ref_pipe = LatentForceEstimationPipeline.from_config(copy.deepcopy(self.base_config))
        ref_pipe.generate_data(verbose=verbose)
        X, Y = ref_pipe._X, ref_pipe._Y

        # Train/evaluate each operator
        n = len(X)
        split = int(0.8 * n)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        for op_name in self.operator_names:
            print(f"\n{'='*60}")
            print(f"  Operator: {op_name.upper()}")
            print(f"{'='*60}")
            cfg = copy.deepcopy(self.base_config)
            cfg.operator_name = op_name
            pipe = LatentForceEstimationPipeline.from_config(cfg)
            pipe._X, pipe._Y = X_train, Y_train

            t0 = time.perf_counter()
            pipe.train(verbose=verbose)
            train_time = time.perf_counter() - t0

            metrics = pipe.evaluate(X_test, Y_test)
            metrics["train_time_s"] = train_time
            self.results[op_name] = metrics

            print(f"  RMSE={metrics['rmse']:.4e}  "
                  f"RelL2={metrics['relative_l2']:.4e}  "
                  f"Train={train_time:.1f}s")

        return self.results

    def print_summary(self) -> None:
        if not self.results:
            print("No results yet — call run() first.")
            return
        print("\n" + "=" * 60)
        print(f"{'Operator':<12} {'RMSE':>10} {'RelL2':>10} {'NLL':>10} {'Time(s)':>10}")
        print("-" * 60)
        for name, m in self.results.items():
            nll = m.get("nll", float("nan"))
            print(f"{name.upper():<12} {m['rmse']:>10.4e} {m['relative_l2']:>10.4e} "
                  f"{nll:>10.4e} {m['train_time_s']:>10.1f}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Default config for quick prototyping
# ---------------------------------------------------------------------------

def default_config() -> PipelineConfig:
    """Return a minimal working config for quick experiments."""
    return PipelineConfig(
        mesh_nx=32,
        mesh_ny=32,
        dt=0.01,
        T=0.5,
        pde_type="diffusion",
        diffusivity=0.01,
        force_type="gaussian",
        force_kwargs={"n_blobs": 3, "seed": 0},
        operator_name="fno",
        operator_kwargs={
            "in_channels": 1,
            "out_channels": 1,
            "hidden_channels": 32,
            "n_layers": 4,
            "modes1": 12,
            "modes2": 12,
        },
        use_mc_dropout=True,
        obs_noise_std=0.005,
        n_realisations=50,
        n_steps_per_real=50,
        batch_size=16,
        n_epochs=100,
        learning_rate=1e-3,
        device="cuda" if TORCH_AVAILABLE and __import__("torch").cuda.is_available() else "cpu",
        grid_shape=[32, 33],   # (nx+1) × (ny+1) for a 32×32 UnitSquareMesh
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = default_config()

    runner = ExperimentRunner(
        base_config=cfg,
        operator_names=["fno", "unet", "mlp"],
    )
    runner.run(verbose=True)
    runner.print_summary()