import pathlib
from datetime import datetime
from functools import cached_property
from typing import Sequence

import numpy as np
from pydantic import BaseModel, confloat, conint, DirectoryPath, validator
from scipy.spatial.transform import Rotation

from .rotation import generate_uniform_rotations, rotation_to_relion_eulers
from .typing import DefocusRange
from .utils import generate_parakeet_config


class SingleImageParameters(BaseModel):
    """Parameters for a single image in a single-particle simulation.
    """
    input_structure: pathlib.Path
    rotation: Rotation
    defocus: confloat(gt=0, lt=10)

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)
        json_encoders = {
            Rotation: rotation_to_relion_eulers,
        }

    @validator('input_structure', allow_reuse=True)
    def resolve_path(cls, v):
        return v.resolve()

    @cached_property
    def rotated_structure_filename(self):
        stem = self.input_structure.stem
        timestamp = datetime.utcnow().strftime(format="%y%m%d%H%M%S%f")
        unique_id = id(self)
        return f'{stem}_{timestamp}_{unique_id}.cif'


class SimulationConfig(BaseModel):
    """Global parameters defining an entire single-particle simulation"""
    input_directory: DirectoryPath
    n_images: conint(gt=0)
    image_sidelength: conint(gt=0, multiple_of=2)
    defocus_range: DefocusRange
    output_basename: str

    @validator('input_directory')
    def contains_structure_files(cls, value: DirectoryPath):
        matches = [
            f for f in value.iterdir()
            if (f.match('*.pdb') or f.match('*.cif'))
        ]
        if len(matches) < 1:
            raise ValueError(
                f'directory {value} contains no structure files (cif/pdb)'
            )
        return value

    @property
    def structure_files(self):
        matches = [
            f for f in self.input_directory.rglob('*')
            if (f.match('*.pdb') or f.match('*.cif'))
        ]
        return matches


class Simulation(BaseModel):
    """Data defining a single particle simulation"""
    config: SimulationConfig
    per_image_parameters: Sequence[SingleImageParameters]

    class Config:
        keep_untouched = (cached_property,)
        json_encoders = {
            Rotation: rotation_to_relion_eulers,
        }

    @classmethod
    def from_config(cls, config: SimulationConfig,
                    random_seed: int = None):
        # random number generator for reproducibility
        rng = np.random.default_rng(random_seed)

        # n uniform samples from structure files in input directory
        structure_file_samples = rng.choice(
            config.structure_files,
            size=config.n_images,
            replace=True
        )

        # n random rotations, one per structure sample
        rotations = generate_uniform_rotations(
            n=config.n_images, random_seed=random_seed
        )

        # n uniform defocus values within defocus range
        defoci = rng.uniform(
            *config.defocus_range, size=config.n_images
        )

        # create per-image simulation parameters
        image_parameters = [
            SingleImageParameters(input_structure=i, rotation=r, defocus=d)
            for i, r, d in zip(structure_file_samples, rotations, defoci)
        ]

        return cls(
            config=config,
            per_image_parameters=image_parameters
        )

    @cached_property
    def parakeet_config_files(self):
        rotated_structure_files = [
            simulation_params.rotated_structure_filename
            for simulation_params in self.per_image_parameters
        ]
        defoci = [
            simulation_params.defocus
            for simulation_params in self.per_image_parameters
        ]

        config_files = [
            generate_parakeet_config(
                structure_file=s,
                image_sidelength=self.config.image_sidelength,
                defocus=d
            )
            for s, d in zip(rotated_structure_files, defoci)
        ]
        return config_files

    def __len__(self):
        return self.config.n_images

    @property
    def zarr_filename(self):
        return f'{self.config.output_basename}.zarr'

    def simulate_image(self, idx: int, save_into_zarr_store=False):
        if 0 > idx > len(self):
            raise IndexError
        from .simulation_functions import simulate_single_image
        zf = self.zarr_filename if save_into_zarr_store else None
        return simulate_single_image(
            simulation=self, idx=idx, zarr_filename=zf
        )

    def as_dask_array(self):
        from .simulation_functions import simulation_as_dask_array
        return simulation_as_dask_array(self)

    def create_zarr_store(self):
        """"creates a zarr store for the results of the simulation"""
        from .simulation_functions import create_zarr_store
        return create_zarr_store(simulation=self)

    def execute(self, client):
        from .simulation_functions import execute
        return execute(simulation=self, client=client)
