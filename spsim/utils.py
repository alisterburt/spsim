import json
import os
from copy import deepcopy
from pathlib import Path

import mrcfile
import starfile
import pandas as pd
import yaml
import zarr

from .parakeet_interface import CONFIG_TEMPLATE


def files_in_directory(directory):
    return (
        Path(entry)
        for entry in os.scandir(directory)
        if Path(entry).is_file()
    )


def yaml2dict(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def generate_parakeet_config(
        structure_file: str, image_sidelength: int, defocus: float
):
    parakeet_config = deepcopy(CONFIG_TEMPLATE)
    centre = image_sidelength
    box_size = image_sidelength * 2
    origin_offset = image_sidelength / 2

    # configure simulation size
    parakeet_config['sample']['box'] = [box_size, box_size, box_size]
    parakeet_config['sample']['centre'] = [centre, centre, centre]
    parakeet_config['microscope']['detector']['origin'] = [origin_offset, origin_offset]
    parakeet_config['microscope']['detector']['nx'] = image_sidelength
    parakeet_config['microscope']['detector']['ny'] = image_sidelength

    # configure simulation file
    parakeet_config['sample']['coords']['filename'] = structure_file

    # configure defocus, in angstroms in parakeet
    parakeet_config['microscope']['objective_lens']['c_10'] = int(-1e4 * defocus)

    return parakeet_config


def zarr2mrcs(zarr_file, mrcs_file):
    za = zarr.convenience.open(zarr_file)
    mrc = mrcfile.new_mmap(mrcs_file, shape=za.shape, mrc_mode=2)
    for idx in range(za.shape[0]):
        mrc.data[idx] = za[idx]
    mrc.close()
    return


def json2star(json_file, star_file):
    with open(json_file, 'r') as f:
        simulation_data = json.load(f)

    optics_df_dict = {
        "rlnOpticsGroup": [1],
        "rlnVoltage": [300],
        "rlnSphericalAberration": [2.7],
        "rlnAmplitudeContrast": [0.1],
        "rlnImagePixelSize": [1.0],
        "rlnImageSize": [simulation_data['config']['image_sidelength']],
        "rlnImageDimensionality": [2],
    }
    optics_df = pd.DataFrame.from_dict(optics_df_dict)

    particle_df_dict = {
        "rlnImageName": [
            f"{idx:06d}@{simulation_data['config']['output_basename']}.mrcs"
            for idx, _ in enumerate(simulation_data['per_image_parameters'])
        ],
        "rlnCoordinateX": 0,
        "rlnCoordinateY": 0,
        "rlnAngleRot": [
            parameters['rotation']['rlnAngleRot']
            for parameters in simulation_data['per_image_parameters']
        ],
        "rlnAngleTilt": [
            parameters['rotation']['rlnAngleTilt']
            for parameters in simulation_data['per_image_parameters']
        ],
        "rlnAnglePsi": [
            parameters['rotation']['rlnAnglePsi']
            for parameters in simulation_data['per_image_parameters']
        ],
        "rlnOpticsGroup": 1,
        "rlnDefocusU": [
            parameters['defocus'] * 1e5
            for parameters in simulation_data['per_image_parameters']
        ],
        "rlnDefocusV": [
            parameters['defocus'] * 1e5
            for parameters in simulation_data['per_image_parameters']
        ],
        "rlnDefocusAngle": 0,
    }
    particle_df = pd.DataFrame.from_dict(particle_df_dict)
    starfile.write(
        data={'optics': optics_df, 'particles': particle_df},
        filename=star_file
    )
    return

