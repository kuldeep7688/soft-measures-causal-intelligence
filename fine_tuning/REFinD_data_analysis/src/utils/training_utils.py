import os
import os.path as op
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from .utils import get_training_device_config
from ..data.data_loaders import data_loader
from ..models.load_model import load_model_by_name

def setup_environment(config):
    """
    Configure the environment variables based on the provided configuration.

    Args:
        config (dict): Configuration dictionary.
    """
    os.environ['SLURM_NTASKS_PER_NODE'] = str(config['slurm']['ntasks_per_node'])
    os.environ["TRANSFORMERS_CACHE"] = config['transformers']['cache_dir']
    os.environ["TOKENIZERS_PARALLELISM"] = str(config['transformers']['tokenizers_parallelism']).lower()

def configure_callbacks(config):
    """
    Configure and return PyTorch Lightning callbacks based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        list: A list of PyTorch Lightning callbacks.
    """
    checkpoint_dir = config['paths']['checkpoint_dir']
    model_checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_f1:.2f}',
        save_top_k=config['training']['save_top_k'],
        mode='max',
        every_n_epochs=config['training']['every_n_epochs']
    )

    early_stopping = EarlyStopping(
        monitor='val_f1',
        patience=config['training']['patience'],
        mode='max'
    )

    return [RichProgressBar(), model_checkpoint_callback, early_stopping]

def initialize_training(config):
    """
    Prepare the training setup including data loaders, model, callbacks, and logger.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: A tuple containing the trainer, Lightning model instance, and data loaders.
    """
    setup_environment(config)

    data_dir = config['paths']['data_raw_dir']
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # Initialize data loaders
    train_loader, dev_loader, test_loader = data_loader(data_dir, config['data']['batch_size'])

    # Device configuration and model initialization
    device_config = get_training_device_config()
    model_name = config['model']['type']
    model_args = (int(len(train_loader) * config['training']['num_epochs']),)
    model, lightning_model = load_model_by_name(model_name, *model_args)
    print("Training on:", device_config['accelerator'])

    # Callbacks and logger configuration
    callbacks = configure_callbacks(config)
    logger = TensorBoardLogger(save_dir=log_dir, name=model_name)

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        callbacks=callbacks,
        accelerator=device_config['accelerator'],
        devices=device_config['devices'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=config['training']['precision'],
        logger=logger,
        log_every_n_steps=config['training']['log_every_n_steps'],
        deterministic=True,
        gradient_clip_val=config['training']['gradient_clip_value']
    )

    return trainer, lightning_model, train_loader, dev_loader, test_loader

def train_and_evaluate(trainer, lightning_model, train_loader, dev_loader, test_loader):
    """
    Execute the training process and evaluate the model.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
        lightning_model (LightningModule): The model to train and evaluate.
        train_loader (DataLoader): DataLoader for the training data.
        dev_loader (DataLoader): DataLoader for the validation data.
        test_loader (DataLoader): DataLoader for the test data.
    """
    start_time = time.time()
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    elapsed_time = time.time() - start_time

    print(f"Training completed in {elapsed_time / 60:.2f} minutes")

    # Evaluate model
    trainer.validate(model=lightning_model, dataloaders=dev_loader, ckpt_path="best")
    test_results = trainer.test(model=lightning_model, dataloaders=test_loader)

    print(f"Validation checkpoint path: {trainer.ckpt_path}")
    print(f"Test Results: {test_results}")

    # Save test results
    with open(op.join(trainer.logger.log_dir, "test_outputs.txt"), "w") as f:
        f.write(f"Time elapsed: {elapsed_time / 60:.2f} minutes\n")
        f.write(f"Test Results: {test_results}")
