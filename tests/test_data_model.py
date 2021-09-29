import json

from scipy.spatial.transform import Rotation

from spsim.data_model import SimulationConfig, SingleImageParameters, Simulation
from .constants import TEST_DATA_DIR


def test_simulation_config():
    input_parameters = SimulationConfig(
        input_directory=TEST_DATA_DIR / 'trajectory',
        output_basename='test',
        n_images=200,
        image_sidelength=512,
        defocus_range=(0.5, 4.5)
    )
    assert isinstance(input_parameters, SimulationConfig)
    return input_parameters


def test_image_parameters_instantiation():
    image_parameters = SingleImageParameters(
        input_structure=TEST_DATA_DIR / 'trajectory' / '6vxx.pdb',
        rotation=Rotation.random(num=1),
        defocus=1.5
    )
    assert isinstance(image_parameters, SingleImageParameters)
    return image_parameters


def test_image_parameter_json_encoding():
    image_parameters = test_image_parameters_instantiation()
    encoded = image_parameters.json()
    assert isinstance(encoded, str)
    decoded = json.loads(encoded)

    # check that eulers are encoded as rln eulers in json
    for key in [f'rlnAngle{euler}' for euler in ('Rot', 'Tilt', 'Psi')]:
        assert key in decoded['rotation']


def test_simulation_from_input_parameters():
    input_parameters = test_simulation_config()
    simulation = Simulation.from_config(input_parameters)
    assert isinstance(simulation, Simulation)
    return simulation


def test_simulation_json_encoding():
    simulation = test_simulation_from_input_parameters()
    encoded = simulation.json()
    assert isinstance(encoded, str)
    return encoded


def test_simulation_from_json():
    encoded = test_simulation_json_encoding()
    simulation = Simulation.parse_obj()
