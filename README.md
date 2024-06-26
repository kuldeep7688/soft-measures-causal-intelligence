# MATCH (Measure Alignment Through Comparative Human Judgement) for FCM Extraction
---

## Description  

Recent advances in natural language processing (NLP) have enhanced the automated extraction of fuzzy cognitive maps (FCMs) from text. This project presents MATCH (Measure Alignment Through Comparative Human Judgement), a novel validation method addressing the high expressiveness and partial correctness in FCM extraction. MATCH employs the Elo rating system to rank annotations through pairwise comparisons and evaluates various thresholded similarity measures against human judgment. We then apply these measures to evaluate state-of-the-art open-source large language models (LLMs) that have been fine-tuned for FCM extraction. Our findings indicate that:

1. MATCH'ed measures successfully quantify qualitative improvements in model inferences in a semi-automated fashion.  
2. Fine-tuned LLMs are more effective at generating FCMs from text than their foundation model counterparts.

This study underscores the potential benefits of developing human-aligned evaluation measures and provides a robust methodology for validating graph-based NLP predictions.

### A Note on this Repository

This repository contains the code and resources associated with the paper "MATCH: Measure Alignment Through Comparative Human Judgement
for FCM Extraction". The content is organized into various directories, each serving a specific purpose related to the project's objectives. As the paper is currently under review for the EMNLP conference, the repository remains anonymous until publication.

Upon acceptance and publication of the paper, the repository will be de-anonymized, and further details, including contact information and license, will be updated accordingly. We encourage collaboration and welcome contributions from the community. Feel free to reach out and cooperate with us; we will be glad to work together.

Please note that some components, especially those in the `user_interface` directory, require specific setup and dependencies as outlined in the respective README and INSTALL files. Ensure you follow the provided instructions for a seamless setup and usage experience.


## Getting Started
____

### Installation

To install and set up this project, please follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kuldeep7688/MATCH.git
   cd MATCH
   ```

2. **Install the required Python libraries:**
   ```bash
   pip install -r requirements.txt
   ```

For detailed installation instructions for the user interface, refer to `user_interface/INSTALL.md`.

### Executing program

To run the fine-tuning script, follow these steps:

1. **Navigate to the `fine-tuning` directory**  
2. **Execute the fine-tuning script:**
   
   ```bash
   python finetuning.py --model-name <model-name> --dataset-dir <dataset-dir> --output-dir <output-dir>
   ```
   Replace `<model-name>`, `<dataset-dir>`, and `<output-dir>` with the appropriate values. The script also accepts additional arguments as needed. Below is a list of some arguments you can provide:
      - `--lora-r`: LoRA rank parameter value
      - `--lora-alpha`: LoRA alpha parameter value
      - `--lora-dropout`: LoRA dropout parameter value
      - `--use-4bit`: Use 4-bit precision in base model loading
      - `--bnb-4bit-compute-dtype`: Compute dtype for 4-bit base model
      - `--bnb-4bit-quant-type`: Quantization type (fp4 or nf4)
      - `--fp16`: Enable fp16 training
      - `--bf16`: Enable bf16 training
      - `--num-train-epochs`: Number of training epochs
      - `--per-device-train-batch-size`: Batch size per GPU for training
      - `--per-device-eval-batch-size`: Batch size per GPU for evaluation
      - `--learning-rate`: Initial learning rate (AdamW optimizer)
      - `--weight-decay`: Weight decay to apply to all layers except bias/layernorm weights
      - `--optim`: Optimizer to use
      - `--lr-scheduler-type`: Learning rate scheduler to use
      - `--evaluation-strategy`: Run evaluation of eval set after every x
      - `--metric-for-best-model`: Metric to use to identify the best model


4. **Run the inference script**
   ```bash
   python inference.py --model-name <model-name> --saved-model-ckpt-path <saved-model-ckpt-path> --input-sentences-df-csv-file <input-file> --output-df-csv-file <output-file>
   ```
   Replace `<model-name>`, `<saved-model-ckpt-path>`, `<input-file>`, and `<output-file>` with the appropriate values.

## Usage
____

### Contributing

Feel free to contact us for cooperation; we will be glad to work together.

### License

The licensing details will be updated after the paper's publication and de-anonymization of the repository.

### Authors and Contact Information

Contact information will be published after the de-anonymization of the repository.

### Version History
____
* 0.1
    * Initial Release
