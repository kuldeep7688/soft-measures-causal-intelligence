# Soft Measures for Extracting Causal Collective Intelligence

---

## Description

This repository contains the code and resources associated with the paper "Soft Measures for Extracting Causal Collective Intelligence". The content is organized into various directories, each serving a specific purpose related to the project's objectives.

Please note that some components, especially those in the `user_interface` directory, require specific setup and dependencies as outlined in the respective README and INSTALL files. Ensure you follow the provided instructions for a seamless setup and usage experience.

### Data Directory

The `data` directory contains both raw and processed datasets used for model training, evaluation, and instruction tuning. The contents include:

- **`train_data.csv`, `val_data.csv`, `test_data.csv`**: These CSV files contain raw textual data, primarily consisting of sentences with causal relationships and corresponding sentence IDs.
  
- **`instruction_tuning_data/`**: This subdirectory contains processed data for instruction tuning. It includes:
  - **`train.jsonl`, `val.jsonl`, `test.jsonl`**: Processed data files in JSONL format with markers such as `<subj>`, `<obj>`, `[INST]`, and start/end of sentence tokens.
  - **`dataset_obj/`**: Contains Hugging Face object representations of the processed JSONL files.

## Getting Started

---

### Installation

To install and set up this project, please follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kuldeep7688/soft-measures-causal-intelligence.git
   cd soft-measures-causal-intelligence
   ```

2. **Install the required Python libraries:**
   ```bash
   pip install -r requirements.txt
   ```

For detailed installation instructions for the user interface, refer to `user_interface/INSTALL.md`.

### Executing program

#### Fine-Tuning

---

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

3. **Run the inference script**

   ```bash
   python inference.py --model-name <model-name> --saved-model-ckpt-path <saved-model-ckpt-path> --input-sentences-df-csv-file <input-file> --output-df-csv-file <output-file>
   ```
   

#### In-Context Learning

---

1. **Navigate to the `in_context_learning` directory**

2. **Execute the zero-shot prompting script:**

   ```bash
   python zero_shot_prompting.py --model-name <model-name> --saved-model-ckpt-path <saved-model-ckpt-path> --input-sentences-df-csv-file <input-file> --output-df-csv-file <output-file>
   ```

3. **Execute the three-shot prompting script:**
   ```bash
   python three_shot_prompting.py --model-name <model-name> --saved-model-ckpt-path <saved-model-ckpt-path> --input-sentences-df-csv-file <input-file> --output-df-csv-file <output-file>
   ```
   

#### Similarity Measures

---

1. **Navigate to the `similarity_measures` directory**

2. **Open the Jupyter notebook to evaluate measures:**

   ```bash
   jupyter notebook Measure_Evaluation.ipynb
   ```

   Follow the instructions within the notebook to run the evaluations.

#### Elo Rating Algorithm

---

1. **Navigate to the `elo_comparison` directory**

2. **Open the Jupyter notebook to generate Elo scores for human-LLM or inter-LLM comparisons:**

   ```bash
   jupyter notebook elo_ranking_20samples.ipynb
   ```

   **OR**

   ```bash
   jupyter notebook inter-rater_reliability_elo.ipynb
   ```

   Follow the instructions within the notebook to run the Elo algorithm.

#### User Interface

---

For detailed installation and usage instructions, refer to `user_interface/INSTALL.md` and `user_interface/README_data-labeling.md`.

1. **Navigate to the `user_interface` directory**

2. **Run the Dash UI for data labeling:**

   ```bash
   python DashUI-Data-Labeling.py
   ```

3. **Run the Dash UI for ELO comparison:**
   ```bash
   python DashUI-ELO-Comparison.py
   ```

## Project Information

---

### Contributing

Feel free to contact us for cooperation; we will be glad to work together.

### Authors and Contact Information

Contact information will be published after the de-anonymization of the repository.

### License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License (CC-BY 4.0)</a>.
