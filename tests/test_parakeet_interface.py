from spsim.parakeet_interface import CONFIG_TEMPLATE
import spsim.parakeet_interface.config
import pytest

@pytest.mark.parametrize(
    'entry', [
        ('cluster'),
        ('device'),
        ('microscope'),
        ('sample'),
        ('scan'),
        ('simulation')
    ]
)
def test_config_template(entry):
    assert isinstance(CONFIG_TEMPLATE, dict)
    assert entry in CONFIG_TEMPLATE


def test_config_write(tmp_path):
    tmp_file = tmp_path / 'config'
    spsim.parakeet_interface.config.write(CONFIG_TEMPLATE, tmp_file)
    assert tmp_file.exists()

