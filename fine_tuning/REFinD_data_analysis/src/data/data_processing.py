import os
import json
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from src.utils import get_tokenizer, convert_labels_to_ids, prepare_text_with_entity_markers
from config.config_loader import load_config, CONFIG

def data_processing():
    """
    Preprocesses the raw data and returns the tokenized dataset.

    Returns:
        datasets.DatasetDict: Tokenized dataset containing train, validation, and test splits.
    """
    # Load configuration
    config = load_config(CONFIG)

    # Define paths
    base_path = config['paths']['data_raw_dir']
    processed_data_path = config['paths']['processed_data_dir']

    # Check if processed data exists, if so, load it
    if os.path.exists(processed_data_path) and os.listdir(processed_data_path):
        print("Processed data found. Loading from disk.")
        return load_from_disk(processed_data_path)

    # Initialize tokenizer
    tokenizer = get_tokenizer()

    # Load raw data
    train_df = pd.DataFrame(json.load(open(os.path.join(base_path, 'train_refind_official.json'), 'r')))
    test_df = pd.DataFrame(json.load(open(os.path.join(base_path, 'test_refind_official.json'), 'r')))
    dev_df = pd.DataFrame(json.load(open(os.path.join(base_path, 'dev_refind_official.json'), 'r')))

    # Select relevant columns
    columns = ['id', 'docid', 'relation', 'rel_group', 'token', 'e1_start', 'e1_end', 'e2_start', 'e2_end']
    train_df = train_df[columns]
    test_df = test_df[columns]
    dev_df = dev_df[columns]

    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Create a subset of the train dataset
    subset_size = float(config['training']['subset_size'])
    subset_indices = np.random.choice(len(train_dataset), size=int(len(train_dataset) * subset_size), replace=False)
    train_subset = train_dataset.select(subset_indices)

    # Combine datasets
    dataset = DatasetDict({
        'train': train_subset,
        'validation': dev_dataset,
        'test': test_dataset
    })

    # Convert labels to ids
    dataset = dataset.map(convert_labels_to_ids)

    # Prepare text with entity markers
    dataset = dataset.map(prepare_text_with_entity_markers)

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples['text'],
            padding='max_length',
            max_length=config['data']['max_length'],
            truncation=True,
        ),
        batched=True
    )

    # Remove unnecessary columns
    all_columns = tokenized_dataset['train'].column_names
    columns_to_keep = ['input_ids', "attention_mask", 'label']
    tokenized_dataset = tokenized_dataset.remove_columns([col for col in all_columns if col not in columns_to_keep])

    # Save processed data
    print(f"Saving processed data to {processed_data_path}")
    tokenized_dataset.save_to_disk(processed_data_path)

    return tokenized_dataset
