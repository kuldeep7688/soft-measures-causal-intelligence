from torch.utils.data import DataLoader
from .data_processing import data_processing
from src.utils import data_collator

def data_loader(data_path, batch_size, num_workers=8):
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
    data_path (str): Path to the data.
    batch_size (int): Batch size for data loader.
    num_workers (int, optional): Number of subprocesses to use for data loading.
                                  Defaults to 8.

    Returns:
    tuple: A tuple containing train_loader, dev_loader, and test_loader.
    """
    
    # Tokenize dataset
    tokenized_dataset = data_processing()

    # Define data loaders for train, validation, and test datasets
    train_loader = DataLoader(
        dataset=tokenized_dataset['train'],
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=data_collator
    )
    
    dev_loader = DataLoader(
        dataset=tokenized_dataset['validation'],
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=data_collator
    )
    test_loader = DataLoader(
        dataset=tokenized_dataset['test'],
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=data_collator
    )
    return train_loader, dev_loader, test_loader