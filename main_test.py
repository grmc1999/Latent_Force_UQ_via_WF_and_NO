from Latent_Force_UQ_via_WF_and_NO import Solvers,StatisticalModels
import importlib
from Latent_Force_UQ_via_WF_and_NO import Solvers
from Latent_Force_UQ_via_WF_and_NO.Solvers import ImplicitDiffusionStepper
import firedrake as fd
import numpy as np
from typing import Optional
from Latent_Force_UQ_via_WF_and_NO.LFEstimation import PipelineConfig, ForceGeneratorFactory, LatentForceEstimationPipeline

if __name__ == "__main__":

    ForceGeneratorFactory.build("pulse")

    mesh = fd.UnitSquareMesh(10,10)

    ph_model = ImplicitDiffusionStepper(mesh = mesh, dt = 0.1)
    step_op = ph_model.build_torch_step_operator()
    breakpoint()
    pipe = LatentForceEstimationPipeline.from_config(PipelineConfig())
    pipe.generate_data(n_realisations=100, n_steps=50)