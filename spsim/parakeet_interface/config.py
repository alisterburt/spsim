from pathlib import Path
import yaml

example_config_file = Path(__file__).parent / 'template.yaml'

with open(example_config_file, 'r') as f:
    CONFIG_TEMPLATE = yaml.safe_load(f)


def write(config, file):
    with open(file, 'w') as f:
        yaml.dump(config, f)
    return