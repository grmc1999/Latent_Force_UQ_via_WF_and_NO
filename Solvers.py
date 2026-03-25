from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch

import numpy as np
import firedrake as fd

class FiredrakeTimeStepper(ABC):
    """
    Abstract differentiable Firedrake time-stepper.

    A subclass must define:
      - function space
      - boundary conditions
      - variational residual for one time step
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        dt: float,
        solver_parameters: Optional[dict] = None,
        point_evaluator: np.ndarray = None,
    ):
        self.mesh = mesh
        self.dt = fd.Constant(dt)
        self.solver_parameters = solver_parameters or {}

        self.V = self.build_function_space(mesh)
        self.bcs = self.build_bcs()
        self.point_evaluator = point_evaluator

        if isinstance(self.point_evaluator, np.ndarray):
            self.evaluation_shape = self.point_evaluator.shape
            vom = fd.VertexOnlyMesh(mesh,self.point_evaluator.reshape(-1,mesh.geometric_dimension()),reorder = False)
            self.P0DG = fd.FunctionSpace(vom, "DG", 0)

    @abstractmethod
    def build_function_space(self, mesh: fd.MeshGeometry):
        ...

    @abstractmethod
    def build_bcs(self):
        ...

    @abstractmethod
    def residual(self, u_np1: fd.Function, u_n: fd.Function, f_n: fd.Function):
        """
        Return the weak residual F(u_{n+1}; v, u_n) = 0 for one implicit step.
        """
        ...

    def step(self, u_n: fd.Function, f_n: fd.Function) -> fd.Function:
        """
        Pure Firedrake step: u_n -> u_{n+1}
        """
        u_np1 = fd.Function(self.V, name="u_np1")
        F = self.residual(u_np1, u_n, f_n)
        fd.solve(
            F == 0,
            u_np1,
            bcs=self.bcs,
            solver_parameters=self.solver_parameters,
        )
        return u_np1

    def build_torch_step_operator(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Build a PyTorch-callable operator corresponding to one Firedrake step.

        The control is u_n, and the output is u_{n+1}.
        """
        fd.adjoint.continue_annotation()
        u_n = fd.Function(self.V, name="u_n")
        u_np1 = fd.Function(self.V, name="u_np1_state")
        f = fd.Function(self.V, name="force")

        F = self.residual(u_np1, u_n, f)
        fd.solve(
            F == 0,
            u_np1,
            bcs=self.bcs,
            solver_parameters=self.solver_parameters,
        )
        if isinstance(self.point_evaluator, np.ndarray):
            u_np1 = fd.assemble(fd.interpolate(u_np1, self.P0DG))

        # Reduced functional whose "output" is the field u_{n+1}
        red = fd.adjoint.ReducedFunctional(u_np1,
                                           [
                                               fd.adjoint.Control(u_n),
                                               fd.adjoint.Control(f),
                                            ]
                                           )
        fd.adjoint.stop_annotating()
        return fd.ml.pytorch.fem_operator(red)
    
    def _inject_forcing(self, f_func: Optional[Callable]) -> None:
        """Interpolate an external forcing callable into self.f_h."""
        if f_func is None:
            self.f_h.assign(fd.Constant(0.0))
        elif isinstance(f_func, fd.Function):
            self.f_h.assign(f_func)
        elif callable(f_func):
            x = fd.SpatialCoordinate(self.mesh)
            self.f_h.interpolate(f_func(x, self.t))
        else:
            raise TypeError(f"Unsupported forcing type: {type(f_func)}")

    def run(
        self,
        n_steps: int,
        f_func: Optional[Callable] = None,
        output_every: int = 1,
        verbose: bool = False,
    ) -> List[np.ndarray]:
        """
        Run the solver for *n_steps* steps.
        Parameters
        ----------
        n_steps : int
            Number of time steps to advance.
        f_func :
            Forcing term (same conventions as ``step``).
        output_every : int
            Store solution snapshot every this many steps.
        verbose : bool
            Print progress.
        Returns
        -------
        List[np.ndarray]
            Snapshots of the DOF vector at requested intervals, including t=0.
        """
        snapshots: List[np.ndarray] = [self.u.dat.data.copy()]
        for i in range(n_steps):
            # Support time-varying forcing: pass current time in closure if callable
            _f = (lambda ff=f_func: ff) if not callable(f_func) else f_func
            snap = self.step(f_func)
            if (i + 1) % output_every == 0:
                snapshots.append(snap)
            if verbose:
                #print(f"  t={self.t:.4f}  ‖u‖={np.linalg.norm(snap):.4e}")
                print("placeholder")
        return snapshots

    def reset(self) -> None:
        """Reset solution and time to initial state."""
        self.u.assign(fd.Constant(0.0))
        self.u_prev.assign(fd.Constant(0.0))
        self.t = 0.0


class ImplicitDiffusionStepper(FiredrakeTimeStepper):
    """
    u_t - div(k grad u) = f
    Backward Euler:
        (u^{n+1} - u^n)/dt - div(k grad u^{n+1}) = f
    """

    def __init__(
        self,
        mesh: fd.MeshGeometry,
        dt: float,
        point_evaluator: np.ndarray = None,
        diffusivity: float = 1.0,
        forcing: float = 0.0,
        degree: int = 1,
        solver_parameters: Optional[dict] = None,
    ):
        self.degree = degree
        self.k = fd.Constant(diffusivity)
        self.f = fd.Constant(forcing)
        super().__init__(
            mesh=mesh,
            dt=dt,
            point_evaluator = point_evaluator,
            solver_parameters=solver_parameters
            )

    def build_function_space(self, mesh):
        return fd.FunctionSpace(mesh, "CG", self.degree)

    def build_bcs(self):
        # Replace as needed
        return [fd.DirichletBC(self.V, fd.Constant(0.0), "on_boundary")]

    def residual(self, u_np1: fd.Function, u_n: fd.Function, f_n: fd.Function):
        v = fd.TestFunction(self.V)

        return (
            ((u_np1 - u_n) / self.dt) * v * fd.dx
            + self.k * fd.dot(fd.grad(u_np1), fd.grad(v)) * fd.dx
            - f_n * v * fd.dx
        )

# ---------------------------------------------------------------------------
# Mesh factory helpers
# ---------------------------------------------------------------------------

class MeshFactory:
    """Convenience helpers for building common mesh types."""

    @staticmethod
    def unit_square(nx: int = 32, ny: int = 32):
        """Return a UnitSquareMesh with nx × ny cells."""
        return fd.UnitSquareMesh(nx, ny)

    @staticmethod
    def unit_interval(n: int = 128):
        """Return a UnitIntervalMesh with n cells."""
        return fd.UnitIntervalMesh(n)

    @staticmethod
    def rectangle(lx: float, ly: float, nx: int, ny: int):
        """Return a rectangular mesh [0, lx] × [0, ly]."""
        return fd.RectangleMesh(nx, ny, lx, ly)