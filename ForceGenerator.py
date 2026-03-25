"""
force_generator.py
==================
Artificial latent force term generator for synthetic experiments.

Provides a library of spatiotemporal forcing patterns that can be used
to drive the PDE solver and generate ground-truth observation datasets.

Each generator returns:
    f(x, t)  ∈  R                  — scalar forcing value at (x, t)
    obs      ∈  R^{N_dof}          — noisy observations of the PDE solution

Generator catalogue
-------------------
- ``GaussianBumpForce``    : one or more moving Gaussian blobs
- ``SinusoidalForce``      : spatiotemporal Fourier sum
- ``RandomFieldForce``     : Gaussian random field (Matérn covariance)
- ``LocalisedPulseForce``  : compact-support pulses at random times
- ``DataDrivenForce``      : replay an arbitrary f(x, t) ndarray
- ``CompositeForce``       : additive superposition of multiple generators

The ``ObservationGenerator`` class wraps any force generator and a PDE
solver to produce labelled (f, u_obs) datasets for supervised training.
"""

from __future__ import annotations

import abc
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseForceGenerator(abc.ABC):
    """
    Abstract forcing term  f : Ω × [0, T] → R.

    Subclasses implement ``__call__(coords, t)`` which returns a 1-D array
    of shape ``(N_coords,)`` giving the force value at every spatial DOF
    for the given time.
    """

    @abc.abstractmethod
    def __call__(self, coords: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate the forcing field.

        Parameters
        ----------
        coords : (N, d)  — spatial coordinates of DOFs / query points
        t      : float   — current time

        Returns
        -------
        np.ndarray  shape (N,)
        """
        ...

    def as_time_series(
        self, coords: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate f at all (coords, t) pairs.

        Returns
        -------
        np.ndarray  shape (len(times), N_coords)
        """
        return np.stack([self(coords, t) for t in times], axis=0)

    def to_firedrake_callable(self) -> Callable:
        """
        Return a Python callable ``g(x, t)`` suitable for injection
        into ``TimeDependentPDESolver.step(f_func=g)``.
        """

        gen = self

        def _callable(x, t):
            # x is a UFL SpatialCoordinate tuple; convert to numpy for eval
            # This is evaluated symbolically — use projection in practice.
            return gen  # Return generator itself; solver handles interpolation

        return _callable


# ---------------------------------------------------------------------------
# Gaussian Bump Force
# ---------------------------------------------------------------------------

class GaussianBumpForce(BaseForceGenerator):
    """
    Sum of moving Gaussian blobs:

        f(x,t) = Σ_k  A_k · exp( -‖x − c_k(t)‖² / (2σ_k²) )

    where c_k(t) = c_k^0 + v_k · t  (constant velocity trajectory).

    Parameters
    ----------
    centres : (K, d)  — initial centre positions
    velocities : (K, d) or None  — constant drift velocities
    amplitudes : (K,) or None    — peak amplitudes (default: 1.0)
    widths     : (K,) or None    — Gaussian width σ (default: 0.1)
    """

    def __init__(
        self,
        centres: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
        widths: Optional[np.ndarray] = None,
    ):
        self.centres = np.atleast_2d(centres).astype(float)
        K, d = self.centres.shape
        self.velocities = np.zeros_like(self.centres) if velocities is None else np.atleast_2d(velocities)
        self.amplitudes = np.ones(K) if amplitudes is None else np.asarray(amplitudes, float)
        self.widths = 0.1 * np.ones(K) if widths is None else np.asarray(widths, float)

    def __call__(self, coords: np.ndarray, t: float) -> np.ndarray:
        coords = np.atleast_2d(coords)         # (N, d)
        f = np.zeros(len(coords))
        for k in range(len(self.centres)):
            c_t = self.centres[k] + self.velocities[k] * t
            sq_dist = np.sum((coords - c_t) ** 2, axis=1)
            f += self.amplitudes[k] * np.exp(-sq_dist / (2 * self.widths[k] ** 2))
        return f

    @classmethod
    def random(
        cls,
        n_blobs: int = 3,
        spatial_dim: int = 2,
        domain: Tuple[float, float] = (0.0, 1.0),
        amplitude_range: Tuple[float, float] = (0.5, 2.0),
        width_range: Tuple[float, float] = (0.05, 0.2),
        speed_range: Tuple[float, float] = (0.0, 0.1),
        seed: Optional[int] = None,
    ) -> "GaussianBumpForce":
        """Factory: create a random configuration of Gaussian blobs."""
        rng = np.random.default_rng(seed)
        lo, hi = domain
        centres = rng.uniform(lo, hi, size=(n_blobs, spatial_dim))
        velocities = rng.uniform(-speed_range[1], speed_range[1], size=(n_blobs, spatial_dim))
        amplitudes = rng.uniform(*amplitude_range, size=n_blobs)
        widths = rng.uniform(*width_range, size=n_blobs)
        return cls(centres, velocities, amplitudes, widths)


# ---------------------------------------------------------------------------
# Sinusoidal Force
# ---------------------------------------------------------------------------

class SinusoidalForce(BaseForceGenerator):
    """
    Spatiotemporal Fourier sum:

        f(x, t) = Σ_k  A_k · sin(ω_k · t + k_x · x + k_y · y + φ_k)

    Parameters
    ----------
    wave_vectors : (K, d)  — spatial wave vectors
    frequencies  : (K,)    — temporal frequencies ω_k
    amplitudes   : (K,) or None
    phases       : (K,) or None
    """

    def __init__(
        self,
        wave_vectors: np.ndarray,
        frequencies: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ):
        self.kv = np.atleast_2d(wave_vectors).astype(float)  # (K, d)
        self.omega = np.asarray(frequencies, float)           # (K,)
        K = len(self.omega)
        self.amplitudes = np.ones(K) if amplitudes is None else np.asarray(amplitudes, float)
        self.phases = np.zeros(K) if phases is None else np.asarray(phases, float)

    def __call__(self, coords: np.ndarray, t: float) -> np.ndarray:
        coords = np.atleast_2d(coords)   # (N, d)
        # spatial phase:  coords @ kv.T  → (N, K)
        spatial_phase = coords @ self.kv.T
        # (N, K) → broadcast with (K,)
        arg = spatial_phase + self.omega * t + self.phases
        f = (self.amplitudes * np.sin(arg)).sum(axis=1)
        return f

    @classmethod
    def random(
        cls,
        n_modes: int = 6,
        spatial_dim: int = 2,
        k_max: float = 4 * np.pi,
        omega_max: float = 2 * np.pi,
        seed: Optional[int] = None,
    ) -> "SinusoidalForce":
        rng = np.random.default_rng(seed)
        wave_vectors = rng.uniform(-k_max, k_max, size=(n_modes, spatial_dim))
        frequencies = rng.uniform(0.0, omega_max, size=n_modes)
        amplitudes = rng.uniform(0.5, 1.5, size=n_modes)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_modes)
        return cls(wave_vectors, frequencies, amplitudes, phases)


# ---------------------------------------------------------------------------
# Gaussian Random Field Force
# ---------------------------------------------------------------------------

class RandomFieldForce(BaseForceGenerator):
    """
    Spatially-correlated Gaussian random field with Matérn covariance,
    evolving slowly in time via an Ornstein–Uhlenbeck process.

    K(r) = σ² · (2^{1-ν}/Γ(ν)) · (√2ν r/ℓ)^ν · K_ν(√2ν r/ℓ)   (Matérn)

    Parameters
    ----------
    coords_grid  : (N, d)  — fixed spatial grid (DOF coordinates)
    length_scale : float   — spatial correlation length ℓ
    amplitude    : float   — marginal standard deviation σ
    nu           : float   — Matérn smoothness ν  (0.5, 1.5, 2.5, ∞)
    temporal_corr: float   — OU mean-reversion rate θ  (larger → faster change)
    seed         : int | None
    """

    def __init__(
        self,
        coords_grid: np.ndarray,
        length_scale: float = 0.2,
        amplitude: float = 1.0,
        nu: float = 1.5,
        temporal_corr: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.coords = np.atleast_2d(coords_grid)
        self.ell = length_scale
        self.sigma = amplitude
        self.nu = nu
        self.theta = temporal_corr
        self.rng = np.random.default_rng(seed)
        self.N = len(self.coords)

        # Pre-compute Cholesky of covariance matrix
        self._L = self._build_cholesky()

        # Initialise spatial state
        self._z = self._L @ self.rng.standard_normal(self.N)

    def _matern_kernel(self, r: np.ndarray) -> np.ndarray:
        """Evaluate Matérn kernel at distances r (nu ∈ {0.5, 1.5, 2.5})."""
        r = np.clip(r, 1e-12, None)
        if abs(self.nu - 0.5) < 1e-6:          # Exponential
            return self.sigma ** 2 * np.exp(-r / self.ell)
        elif abs(self.nu - 1.5) < 1e-6:
            s = np.sqrt(3) * r / self.ell
            return self.sigma ** 2 * (1 + s) * np.exp(-s)
        elif abs(self.nu - 2.5) < 1e-6:
            s = np.sqrt(5) * r / self.ell
            return self.sigma ** 2 * (1 + s + s ** 2 / 3) * np.exp(-s)
        else:
            raise ValueError(f"nu must be 0.5, 1.5, or 2.5; got {self.nu}")

    def _build_cholesky(self) -> np.ndarray:
        D = cdist(self.coords, self.coords)
        K = self._matern_kernel(D)
        K += 1e-8 * np.eye(self.N)      # jitter for numerical stability
        return np.linalg.cholesky(K)

    def _ou_update(self, dt: float = 0.01) -> None:
        """Advance the OU process by one step."""
        noise = self._L @ self.rng.standard_normal(self.N)
        decay = np.exp(-self.theta * dt)
        self._z = decay * self._z + np.sqrt(1 - decay ** 2) * noise

    def __call__(self, coords: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate the random field.  Consecutive calls with increasing t
        produce a temporally correlated sequence.
        """
        # Advance OU process (use small fixed dt proportional to call)
        self._ou_update(dt=0.01)
        # If coords match stored grid: return directly
        if np.allclose(coords, self.coords):
            return self._z.copy()
        # Otherwise: kriging interpolation to new coords
        D_cross = cdist(np.atleast_2d(coords), self.coords)
        K_cross = self._matern_kernel(D_cross)    # (M, N)
        K_inv_z = np.linalg.solve(self._L.T, np.linalg.solve(self._L, self._z))
        return K_cross @ K_inv_z

    def sample_field(self) -> np.ndarray:
        """Return a fresh independent sample from the GP prior."""
        return self._L @ self.rng.standard_normal(self.N)


# ---------------------------------------------------------------------------
# Localised Pulse Force
# ---------------------------------------------------------------------------

class LocalisedPulseForce(BaseForceGenerator):
    """
    Compact-support pulses triggered at random times and locations.

        f(x, t) = Σ_k  A_k · φ(‖x − c_k‖/r_k) · ψ((t − t_k)/τ_k)

    where φ is a C²-smooth bump and ψ is a raised-cosine time envelope.

    Parameters
    ----------
    n_pulses  : int   — number of pulses to pre-generate
    domain    : (lo, hi) — spatial domain bounds
    t_max     : float — maximum trigger time
    seed      : int | None
    """

    def __init__(
        self,
        n_pulses: int = 10,
        domain: Tuple[float, float] = (0.0, 1.0),
        t_max: float = 1.0,
        spatial_dim: int = 2,
        seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)
        lo, hi = domain
        self.centres = rng.uniform(lo, hi, size=(n_pulses, spatial_dim))
        self.radii = rng.uniform(0.05, 0.25, size=n_pulses)
        self.trigger_times = rng.uniform(0.0, t_max, size=n_pulses)
        self.durations = rng.uniform(0.05, 0.3, size=n_pulses)
        self.amplitudes = rng.uniform(-2.0, 2.0, size=n_pulses)

    @staticmethod
    def _bump(r: np.ndarray) -> np.ndarray:
        """C²-smooth bump function supported on [0, 1]."""
        inside = r < 1.0
        out = np.zeros_like(r)
        ri = r[inside]
        out[inside] = np.where(ri > 0, np.exp(-1.0 / (1.0 - ri ** 2 + 1e-12)), 1.0)
        return out / (np.exp(-1.0) + 1e-12)   # normalise to ≈ 1 at r=0

    @staticmethod
    def _time_envelope(t: float, t0: float, tau: float) -> float:
        """Raised-cosine temporal envelope centred at t0 with half-width tau."""
        dt = abs(t - t0)
        if dt > tau:
            return 0.0
        return 0.5 * (1 + np.cos(np.pi * dt / tau))

    def __call__(self, coords: np.ndarray, t: float) -> np.ndarray:
        coords = np.atleast_2d(coords)
        f = np.zeros(len(coords))
        for k in range(len(self.centres)):
            te = self._time_envelope(t, self.trigger_times[k], self.durations[k])
            if te == 0.0:
                continue
            r = np.linalg.norm(coords - self.centres[k], axis=1) / self.radii[k]
            f += self.amplitudes[k] * te * self._bump(r)
        return f


# ---------------------------------------------------------------------------
# Composite Force  (additive superposition)
# ---------------------------------------------------------------------------

class CompositeForce(BaseForceGenerator):
    """
    Additive superposition of multiple forcing generators.

    Example
    -------
    >>> f = CompositeForce([GaussianBumpForce(...), SinusoidalForce(...)])
    """

    def __init__(self, generators: List[BaseForceGenerator], weights: Optional[List[float]] = None):
        self.generators = generators
        self.weights = weights if weights is not None else [1.0] * len(generators)

    def __call__(self, coords: np.ndarray, t: float) -> np.ndarray:
        return sum(w * g(coords, t) for w, g in zip(self.weights, self.generators))

    def add(self, generator: BaseForceGenerator, weight: float = 1.0) -> "CompositeForce":
        self.generators.append(generator)
        self.weights.append(weight)
        return self


# ---------------------------------------------------------------------------
# Data-driven Force (replays precomputed array)
# ---------------------------------------------------------------------------

class DataDrivenForce(BaseForceGenerator):
    """
    Replay a precomputed forcing time series ``F[time_idx, dof_idx]``.

    Between stored time steps, linear interpolation is used.

    Parameters
    ----------
    F      : (T_stored, N_dof) — forcing snapshots
    times  : (T_stored,)       — times corresponding to each row of F
    coords : (N_dof, d)        — DOF coordinates matching columns of F
    """

    def __init__(
        self,
        F: np.ndarray,
        times: np.ndarray,
        coords: np.ndarray,
    ):
        self.F = np.asarray(F, float)
        self.times = np.asarray(times, float)
        self.coords = np.asarray(coords, float)

    def __call__(self, coords: np.ndarray, t: float) -> np.ndarray:
        # Temporal interpolation
        f_t = np.array([
            np.interp(t, self.times, self.F[:, j])
            for j in range(self.F.shape[1])
        ])
        # If coords match stored: return directly
        if np.allclose(coords, self.coords):
            return f_t
        # Nearest-neighbour fallback for new coordinates
        idx = np.argmin(cdist(np.atleast_2d(coords), self.coords), axis=1)
        return f_t[idx]


# ---------------------------------------------------------------------------
# Observation Generator
# ---------------------------------------------------------------------------

class ObservationGenerator:
    """
    Generates paired (forcing, observed_solution) datasets by:

    1. Driving the PDE solver with a chosen forcing generator.
    2. Adding synthetic observation noise to the solution.
    3. Optionally sub-sampling to a sparse sensor network.

    Parameters
    ----------
    pde_solver :
        An instance of ``TimeDependentPDESolver`` (or any object with a
        ``step(f_func)`` and ``get_dof_coordinates()`` method).
    force_generator : BaseForceGenerator
        The latent forcing to inject.
    obs_noise_std : float
        Standard deviation of additive Gaussian observation noise.
    sensor_indices : array-like or None
        DOF indices at which observations are recorded.
        ``None`` → observe all DOFs (dense).
    """

    def __init__(
        self,
        pde_solver,
        force_generator: BaseForceGenerator,
        obs_noise_std: float = 0.005,
        sensor_indices: Optional[np.ndarray] = None,
    ):
        self.solver = pde_solver
        self.force_gen = force_generator
        self.noise_std = obs_noise_std
        self.sensor_indices = sensor_indices

    def _observe(self, u_dofs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply noise and (optionally) sub-sample."""
        if self.sensor_indices is not None:
            u_obs = u_dofs[self.sensor_indices]
        else:
            u_obs = u_dofs.copy()
        u_obs += rng.normal(0.0, self.noise_std, size=u_obs.shape)
        return u_obs

    def generate(
        self,
        n_steps: int,
        output_every: int = 1,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run the forward PDE and collect observations.

        Returns
        -------
        dict with keys:
          - ``'times'``         : (T_out,) time values
          - ``'forcing'``       : (T_out, N_dof)  true forcing field
          - ``'solution'``      : (T_out, N_dof)  true PDE solution
          - ``'observations'``  : (T_out, N_obs)  noisy observations
          - ``'coords'``        : (N_dof, d)       DOF coordinates
          - ``'sensor_indices'``: (N_obs,) or None
        """
        rng = np.random.default_rng(seed)
        coords = self.solver.get_dof_coordinates()

        times_out, forcing_out, solution_out, obs_out = [], [], [], []

        for i in range(n_steps):
            t = self.solver.t
            f_vals = self.force_gen(coords, t)

            # Build a Firedrake-injectable forcing
            def _f_func(x_fd, t_fd, _fv=f_vals, _solver=self.solver):
                # Interpolate pre-computed f_vals as a Function
                import firedrake as fd
                from firedrake import Function
                f_fn = Function(_solver.V)
                f_fn.dat.data[:] = _fv
                return f_fn

            u_dofs = self.solver.step(f_func=None)   # inject via f_h directly
            # Inject forcing manually
            try:
                self.solver.f_h.dat.data[:] = f_vals
                self.solver._solver.solve()
                self.solver.u_prev.assign(self.solver.u)
                self.solver.t += self.solver.dt_val
                u_dofs = self.solver.u.dat.data.copy()
            except Exception:
                pass   # fallback: solver already advanced one step

            if (i + 1) % output_every == 0:
                times_out.append(t)
                forcing_out.append(f_vals.copy())
                solution_out.append(u_dofs.copy())
                obs_out.append(self._observe(u_dofs, rng))

            if verbose:
                print(f"  step {i+1}/{n_steps}  t={t:.4f}")

        return {
            "times": np.array(times_out),
            "forcing": np.array(forcing_out),
            "solution": np.array(solution_out),
            "observations": np.array(obs_out),
            "coords": coords,
            "sensor_indices": self.sensor_indices,
        }

    def generate_batch(
        self,
        n_realisations: int,
        n_steps: int,
        output_every: int = 1,
        seed: Optional[int] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate multiple independent realisations (different forcing seeds).

        Returns a list of dicts, one per realisation.
        """
        rng = np.random.default_rng(seed)
        dataset = []
        for k in range(n_realisations):
            self.solver.reset()
            # Re-seed the random force generator if it supports it
            if hasattr(self.force_gen, 'rng'):
                self.force_gen.rng = np.random.default_rng(rng.integers(1 << 31))
            data = self.generate(n_steps, output_every=output_every, seed=int(rng.integers(1 << 31)))
            dataset.append(data)
        return dataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_pytorch_dataset(
    records: List[Dict[str, np.ndarray]],
    grid_shape: Tuple[int, ...],
    in_channels: int = 1,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Convert a list of observation records to a (X, Y) PyTorch tensor pair
    suitable for supervised training of a Neural Operator.

    X : (B, in_channels, *grid_shape)   — noisy observations / inputs
    Y : (B, 1, *grid_shape)             — ground truth forcing

    Parameters
    ----------
    records     : output of ``ObservationGenerator.generate_batch()``
    grid_shape  : (H, W) for 2-D grids
    in_channels : number of input channels (e.g. solution + coords channels)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for make_pytorch_dataset.")

    X_list, Y_list = [], []
    for rec in records:
        T = rec["observations"].shape[0]
        for t_idx in range(T):
            obs = rec["observations"][t_idx].reshape(grid_shape)
            f_gt = rec["forcing"][t_idx].reshape(grid_shape)
            X_list.append(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            Y_list.append(torch.tensor(f_gt, dtype=torch.float32).unsqueeze(0))

    X = torch.stack(X_list, dim=0)   # (N_samples, 1, H, W)
    Y = torch.stack(Y_list, dim=0)
    return X, Y