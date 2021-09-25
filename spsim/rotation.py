import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group


def generate_uniform_rotations(n: int, random_seed=None):
    return Rotation.from_matrix(special_ortho_group.rvs(
        dim=3, size=n, random_state=random_seed
    ))


def rotate_coordinates(coordinates: np.ndarray, rotation: Rotation, center: np.ndarray):
    if center is None:
        center = np.mean(coordinates, axis=0)
    coordinates_centered = coordinates - center
    return rotation.apply(coordinates_centered) + center


def rotation_to_relion_eulers(rotation: Rotation):
    eulers = rotation.inv().as_euler('ZYZ', degrees=True)
    print(eulers.shape, eulers)
    data = {
        'rlnAngleRot': eulers[0],
        'rlnAngleTilt': eulers[1],
        'rlnAnglePsi': eulers[2]
    }
    return data
