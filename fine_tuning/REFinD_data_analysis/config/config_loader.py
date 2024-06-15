import yaml

CONFIG = './config/config.yaml'

def load_config(file_path):
    """
    Load configuration from a YAML file.

    Parameters:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration loaded from the YAML file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
