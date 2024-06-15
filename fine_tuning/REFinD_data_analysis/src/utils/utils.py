import torch
from transformers import AutoConfig, AutoTokenizer
from .constants import E1_START_TOKEN, E1_END_TOKEN, E2_START_TOKEN, E2_END_TOKEN, LABEL_TO_ID
from config.config_loader import load_config, CONFIG

# Load configuration from file
config = load_config(CONFIG)

max_length = config['data']['max_length']

def convert_labels_to_ids(example):
    """
    Converts relation labels in an example to their corresponding numeric IDs.

    Args:
        example (dict): A single data example which includes a 'relation' key.

    Returns:
        dict: The modified example with 'label' key updated to a numeric ID.
    """
    example['label'] = LABEL_TO_ID[example['relation'].split(':')[-1]]
    return example

def prepare_text_with_entity_markers(example, e1_start=E1_START_TOKEN, e1_end=E1_END_TOKEN, 
                                     e2_start=E2_START_TOKEN, e2_end=E2_END_TOKEN):
    """
    Prepares text data by inserting entity markers around entities in a given example.

    Args:
        example (dict): A single data example.
        e1_start (str): Start token for the first entity.
        e1_end (str): End token for the first entity.
        e2_start (str): Start token for the second entity.
        e2_end (str): End token for the second entity.

    Returns:
        dict: The modified example with 'text' key updated to include entity markers.
    """
    token_list = (example['token'][:example['e1_start']] + [e1_start] + 
                  example['token'][example['e1_start']: example['e1_end']] + [e1_end] + 
                  example['token'][example['e1_end']:example['e2_start']] + [e2_start] + 
                  example['token'][example['e2_start']: example['e2_end']] + [e2_end] + 
                  example['token'][example['e2_end']:])
    example['text'] = ' '.join(token_list)
    return example

def data_collator(batch):
    """
    Custom data collator for padding and preparing batches for model input.

    Args:
        batch (list): A batch of data examples.

    Returns:
        dict: A dictionary with padded 'input_ids', 'attention_mask', and 'labels' for the batch.
    """
    tokenizer = get_tokenizer()
    padding_token_id = tokenizer.pad_token_id
    input_ids = [item["input_ids"][:max_length] for item in batch]
    attention_masks = [item["attention_mask"][:max_length] for item in batch]
    labels = [item["label"] for item in batch]

    max_len = min(max_length, max(len(ids) for ids in input_ids))
    input_ids = torch.tensor([ids + [padding_token_id] * (max_len - len(ids)) for ids in input_ids])
    attention_masks = torch.tensor([masks + [padding_token_id] * (max_len - len(masks)) for masks in attention_masks])
    labels = torch.tensor(labels)
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_masks,
        "labels": labels,
    }

def get_training_device_config():
    """
    Determines the best training device configuration based on available hardware.

    Returns:
        dict: A dictionary specifying the 'accelerator' type and 'devices' to use.
    """
    if torch.cuda.is_available():
        return {'accelerator': 'gpu', 'devices': [0]}
    return {'accelerator': 'cpu', 'devices': [0]}

def get_model_config():
    """
    Generates a model configuration object based on the project configuration.

    Returns:
        transformers.AutoConfig: The configuration object for the model.
    """
    model_type = config['model']['type']
    model_config = AutoConfig.from_pretrained(model_type)
    model_config.num_labels = config['data']['num_labels']
    return model_config

def get_tokenizer():
    """
    Initializes and returns a tokenizer based on the project configuration.

    Returns:
        transformers.AutoTokenizer: The initialized tokenizer with added tokens.
    """
    model_type = config['model']['type']
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    # Add special tokens for entity recognition
    tokenizer.add_tokens([E1_START_TOKEN, E1_END_TOKEN, E2_START_TOKEN, E2_END_TOKEN])
    return tokenizer
