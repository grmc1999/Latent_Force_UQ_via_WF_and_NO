from src import Solvers,StatisticalModels
import importlib
from src import Solvers
from src.Solvers import ImplicitDiffusionStepper
import firedrake as fd
import numpy as np
from typing import Optional
from src.LFEstimation import PipelineConfig, ForceGeneratorFactory, LatentForceEstimationPipeline

from firedrake.adjoint import continue_annotation
from firedrake import ml
from firedrake.ml import pytorch

import torch

if __name__ == "__main__":

    ForceGeneratorFactory.build("pulse")

    mesh = fd.UnitSquareMesh(10,10)

    ph_model = ImplicitDiffusionStepper(mesh = mesh, dt = 0.1)
    step_op = ph_model.build_torch_step_operator()
    breakpoint()
    pipe = LatentForceEstimationPipeline.from_config(PipelineConfig())
    pipe.generate_data(n_realisations=100, n_steps=50)