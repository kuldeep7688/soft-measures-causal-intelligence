from config.config_loader import load_config, CONFIG
from src.utils.training_utils import initialize_training, train_and_evaluate

def train(config_path):
    """
    Loads the project configuration and initiates the training and evaluation process.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load the project configuration from the specified path
    config = load_config(config_path)

    # Initialize the training environment, data loaders, model, and other utilities
    trainer, lightning_model, train_loader, dev_loader, test_loader = initialize_training(config)

    # Start the training and evaluation process
    train_and_evaluate(trainer, lightning_model, train_loader, dev_loader, test_loader)
