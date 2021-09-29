import spsim.simulation_functions
from spsim.simulation_functions import simulate_single_image
from .test_data_model import test_image_parameters_instantiation, \
    test_simulation_from_input_parameters


def test_simulate_image():
    image_parameters = test_image_parameters_instantiation()
    image = simulate_single_image(image_parameters)
    assert image.shape == (512, 512)


def test_execute_simulation():
    simulation = test_simulation_from_input_parameters()
    spsim.simulation_functions.simulation_as_dask_array()
