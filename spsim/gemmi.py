"""
Get info from GEMMI into normal Python stuff...
"""
import numpy as np
from gemmi import Model, Structure, CRA
from scipy.spatial.transform import Rotation as R

from .rotation import rotate_coordinates


def xyz_from_model(model: Model):
    return np.array(
        [
            [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
            for cra in model.all()
        ]
    )


def update_xyz_in_model(model: Model, new_xyz: np.ndarray):
    for idx, cra in enumerate(model.all()):
        update_xyz_in_cra(cra, new_xyz[idx])


def update_xyz_in_cra(cra: CRA, new_xyz: np.ndarray):
    cra.atom.pos.x = new_xyz[0]
    cra.atom.pos.y = new_xyz[1]
    cra.atom.pos.z = new_xyz[2]


def rotate_structure(structure: Structure, rotation: R, center=None):
    for model in structure:
        rotate_model(model, rotation, center)
    return structure


def rotate_model(model: Model, rotation: R, center):
    atomic_coordinates = xyz_from_model(model)
    rotated_atomic_coordinates = rotate_coordinates(
        atomic_coordinates, rotation, center
    )
    update_xyz_in_model(model, rotated_atomic_coordinates)


def structure_to_cif(structure: Structure, output_filename: str):
    structure.make_mmcif_document().write_file(output_filename)