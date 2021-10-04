import os
import subprocess
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import gemmi
import mrcfile
import numpy as np
import zarr
from dask import delayed, array as da
from dask.distributed import fire_and_forget, Client

from .data_model import Simulation, SimulationConfig
from .gemmi import rotate_structure, structure_to_cif
from .parakeet_interface.config import write as write_config


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


def create_zarr_store(simulation: Simulation) -> str:
    n_images = len(simulation)
    nxy = simulation.config.image_sidelength
    filename = simulation.zarr_filename
    za = zarr.open(
        filename,
        mode='w',
        shape=(n_images, nxy, nxy),
        chunks=(1, nxy, nxy),
        dtype=np.float16,
    )
    return filename


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


def simulate_single_image(
        simulation: Simulation, idx: int, zarr_filename: Optional[str] = None
) -> np.ndarray:
    """Generate a single image from a single-particle simulation.

    Optionally saves image into zarr store
    """
    # get info required for simulation
    image_parameters = simulation.per_image_parameters[idx]
    parakeet_config = simulation.parakeet_config_files[idx]

    # do work in a temporary directory (parakeet makes a bunch of files)
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

        # run parakeet
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

        # load image file and invert
        with mrcfile.open('image.mrc') as mrc:
            image = np.squeeze(mrc.data) * -1

    # change back to base directory
    os.chdir(base_directory)

    # optionally save image into zarr store
    if zarr_filename is not None:
        save_image_into_zarr_store(
            image=image, idx=idx, zarr_filename=zarr_filename
        )
    return image


def save_image_into_zarr_store(image, idx, zarr_filename):
    zs = zarr.convenience.open(zarr_filename)
    zs[idx, ...] = image
    return True


def simulation_as_dask_array(
        simulation: Simulation
):
    """Provide a dask array around results of a simulation"""
    # make lazy-version of the simulate_image function
    lazy_simulate_single_image = delayed(simulate_single_image)

    # precompute image shape
    nx = simulation.config.image_sidelength
    image_shape = (nx, nx)

    # lazy array calculation
    delayed_images = [
        lazy_simulate_single_image(simulation, idx)
        for idx
        in range(len(simulation))
    ]

    dask_arrays = [
        da.from_delayed(delayed_image, shape=image_shape, dtype=np.float32)
        for delayed_image in delayed_images
    ]

    # create stack from lazy images
    particle_stack = da.stack(dask_arrays, axis=0)
    return particle_stack


def execute(
        simulation: Simulation,
        client: Client
):
    n_images = len(simulation)
    simulation.create_zarr_store()
    for idx in range(n_images):
        future = client.submit(simulation.simulate_image, idx)
        fire_and_forget(future)
    return
