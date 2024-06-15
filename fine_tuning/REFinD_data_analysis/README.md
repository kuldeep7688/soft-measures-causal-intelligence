
# Entity Relationship Extraction: A Multi-Approach Study Using REFinD Dataset

## Overview

This project is dedicated to extracting and classifying relationships between entities in financial texts. It identifies entities—significant words or phrases that represent real-world objects or concepts—in short text segments derived from the [REFinD dataset](https://arxiv.org/abs/2305.18322). The goal is to classify the nature of relationships between these entities, such as "association," "part of," or "cause-effect." Understanding these relationships is crucial for deciphering the underlying structure and meaning of financial texts.

This repository houses the code and documentation necessary for further development and experimentation with relation extraction models.

## Model Fine-Tuning

We fine-tune the RoBERTa model, leveraging its capabilities for relation extraction. The process utilizes Hugging Face's `transformers` library.

### Environment Setup

To set up your environment for running the project code, ensure you have Python 3.8 or newer. Install the necessary dependencies by running:

```bash
conda env create -f environment.yaml
```

### Training the Model

To train the model, navigate to the project root directory and execute:

```bash
python run.py
```

This script initializes the training process using the configurations specified in `config/config.yaml`.

### Evaluation and Results

We assess the model's performance on the test set, focusing on metrics such as F1 score (both micro and macro) to gauge its effectiveness in relation classification.

## Project Structure

The project is organized as follows:

```
project/
│
├── config/                      
│   ├── config.yaml              # Configuration settings for models and training
│   └── config_loader.py         # Utility for loading configuration settings
│
├── data/                        
│   ├── processed/               # Processed data ready for model input
│   └── raw/                     # Original dataset files
│
├── models/                      
│   ├── checkpoints/             # Stored model files and training checkpoints
│   └── best/                    # Stored best performing model checkpoint
│
├── src/                         
│   ├── data/                    # Data handling modules
│   │   ├── data_loaders.py      # For creating data loaders for the model
│   │   └── data_processing.py   # For preprocessing datasets
│   │
│   ├── models/                  # Model definitions and utilities
│   │   ├── load_model.py        # Utility for loading models
│   │   └── lightning_model.py   # Defines the PyTorch Lightning model class
│   │
│   ├── utils/                   # Utility functions for training and data manipulation
│   │   ├── training_utils.py    # Utilities for model training and evaluation
│   │   ├── utils.py             # General utility functions and configurations
│   │   └── constants.py         # Constant values used throughout the project
│   │
│   ├── __init__.py              
│   └── train.py                 # Main training script for the model
│
├── docs/                        # Documentation and reference materials
│   └── original_paper.pdf       # The original research paper and related documentation
│
├── notebooks/                   
│   └── data_analysis.ipynb      # Jupyter notebook for initial data analysis and exploration
│
├── run.py                       # Entry point for running the project scripts
├── environment.yaml             # Conda environment file with Python dependencies
│
└── README.md                    
```
