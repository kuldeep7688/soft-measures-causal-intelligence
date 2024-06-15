from transformers import RobertaForSequenceClassification
from transformers import RobertaForSequenceClassification
import transformers

from config.config_loader import load_config, CONFIG
from src.models.lightning_model import LightningModel
from src.utils import get_model_config, get_tokenizer, constants

# Set logging verbosity
transformers.logging.set_verbosity_error()

# Load project configuration
config = load_config(CONFIG)
model_type = config['model']['type']
warmup_steps = config['training']['warmup_steps']
learning_rate = float(config['training']['learning_rate'])
num_classes = config['data']['num_labels']

def roberta_base(train_steps):
    """
    Initializes and returns a Roberta base model along with its corresponding LightningModel.

    Args:
        train_steps (int): Total number of training steps to be used in the LightningModel.

    Returns:
        tuple: A tuple containing the RobertaForSequenceClassification model instance and the LightningModel instance.
    """
    # Load the Roberta model with the specified configuration
    model = RobertaForSequenceClassification.from_pretrained(model_type, config=get_model_config())
    # Adjust the model's token embeddings to match the tokenizer's vocabulary size
    model.resize_token_embeddings(len(get_tokenizer()))
    print('Roberta base model loaded successfully.')

    # Initialize and return the LightningModel with the loaded model and training parameters
    lightning_model = LightningModel(model, learning_rate=learning_rate, t_total=train_steps, warmup_steps=warmup_steps, num_classes=num_classes)
    print('Lightning model instance created successfully.')
    
    return model, lightning_model

# Mapping of model names to their corresponding loading functions
MODEL_MAP = {
    'roberta-base': roberta_base,
}

def load_model_by_name(model_name, *args, **kwargs):
    """
    Loads and returns a model instance by its name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        object: An instance of the requested model.

    Raises:
        ValueError: If the model_name is not recognized.
    """
    if model_name in MODEL_MAP:
        return MODEL_MAP[model_name](*args, **kwargs)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")