from .test_data_model import test_image_parameters_instantiation, test_simulation_from_input_parameters
from spsim.simulation import simulate_image


def test_simulate_image():
    image_parameters = test_image_parameters_instantiation()
    image = simulate_image(image_parameters)
    assert image.shape == (512, 512)


def test_execute_simulation():
    simulation = test_simulation_from_input_parameters()
    simulation.execute()