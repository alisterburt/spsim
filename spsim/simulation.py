import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from pydantic import BaseModel
from tempfile import TemporaryDirectory

import mrcfile
import gemmi
import numpy as np
import pandas as pd
from dask import delayed
import dask.array as da

from .rotation import generate_uniform_rotations
from .utils import files_in_directory
from .gemmi import rotate_structure, structure_to_cif
from .data_model import SingleImageParameters, Simulation
from .parakeet_interface import write_config


if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation


def load_rotate_save(
        structure_file: str,
        rotation: "Rotation",
        output_filename: str
) -> str:
    """Load a structure, rotate it in memory then save as a cif file.

    This is done in one operation because parallelism requires that operations
    are atomic.
    Room for IO optimisation here if we move away from gemmi but parallel reads
    on SCARF are supposed to be well optimised.
    """
    structure = gemmi.read_structure(structure_file)
    rotate_structure(structure, rotation, center=None)
    structure_to_cif(structure, output_filename)
    return output_filename


def simulate_image(
        image_parameters: SingleImageParameters, parakeet_config: dict
):
    base_directory = Path('.').absolute()
    with TemporaryDirectory() as tmp_dir:
        # change into temporary directory
        os.chdir(tmp_dir)

        # rotate structure and save
        load_rotate_save(
            structure_file=str(image_parameters.input_structure),
            rotation=image_parameters.rotation,
            output_filename=image_parameters.rotated_structure_filename
        )

        # write parakeet config file
        write_config(parakeet_config, 'parakeet_config.yaml')

        # run simulation
        subprocess.run(
            ['parakeet.sample.new', '-c', 'parakeet_config.yaml']
        )
        subprocess.run(
            ['parakeet.simulate.exit_wave', '-c', 'parakeet_config.yaml']
        )
        subprocess.run(
            ['parakeet.simulate.optics', '-c', 'parakeet_config.yaml']
        )
        subprocess.run(
            ['parakeet.simulate.image', '-c', 'parakeet_config.yaml']
        )
        subprocess.run(
            ['parakeet.export', 'image.h5', '-o', 'image.mrc']
        )

        # load image file
        with mrcfile.open('image.mrc') as mrc:
            image = np.squeeze(mrc.data)

    # change back to base directory
    os.chdir(base_directory)
    return image


def execute(
        simulation: Simulation, output_file: str = 'simulated_particles.zarr'
):
    """Execute a defined simulation and save particles as a zarr file."""
    # make lazy-version of the simulate_image function
    lazy_simulate_image = delayed(simulate_image)

    # precompute image shape
    nx = simulation.config.image_sidelength
    image_shape = (nx, nx)

    # lazy array calculation
    delayed_simulations = [
        lazy_simulate_image(simulation_parameters, parakeet_config)
        for simulation_parameters, parakeet_config
        in zip(
            simulation.per_image_parameters, simulation.parakeet_config_files
        )
    ]

    dask_arrays = [
        da.from_delayed(ds, shape=image_shape, dtype=np.float32)
        for ds in delayed_simulations
    ]

    # create stack from lazy images, save to zarr
    # images will be computed on the fly and saved into the zarr store using
    # available resources
    particle_stack = da.stack(dask_arrays, axis=0)
    particle_stack.to_zarr(output_file)
    return particle_stack