from .data_model import Simulation, SimulationConfig
from pathlib import Path

import numpy as np


def prepare_simulation(
        input_directory: Path,
        output_basename: str,
        n_images: int,
        image_sidelength: int,
        defocus_range: tuple[float],
        random_seed: int = None,
) -> Simulation:
    input_parameters = SimulationConfig(
        input_directory=input_directory,
        output_basename=output_basename,
        n_images=n_images,
        image_sidelength=image_sidelength,
        defocus_range=defocus_range,
        random_seed=random_seed
    )
    return Simulation.from_config(
        config=input_parameters, random_seed=random_seed
    )
