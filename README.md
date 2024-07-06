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

4. **Refer to the tutorial notebook for detailed examples and usage instructions:**
   
   Open and follow the instructions in `mistral_finetune_own_data_tutorial.ipynb`.

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

#### Inference
---

1. **Navigate to the `inference` directory**

2. **Execute the inference script:**
   ```bash
   python do_inference.py
   ```
   This script connects to a MongoDB database to fetch data, runs the inference, and saves the results to a JSON file. Ensure that the MongoDB credentials are set in the environment variables (`MONGO_HOST`, `MONGO_PORT`, `MONGO_USER`, `MONGO_PASS`, `MONGO_AUTH`).

3. **Save inference results to the database:**
   ```bash
   python save_inference_to_db.py
   ```
   This script reads the inference results from a JSON file and saves them to a MongoDB database. Ensure that the MongoDB credentials are set in the environment variables (`MONGO_HOST`, `MONGO_PORT`, `MONGO_USER_RW`, `MONGO_PASS_RW`, `MONGO_AUTH`).

**Note: Differences Between Inference Code in `fine-tuning` and `inference` Directories**

- Inference in `fine-tuning` Directory:
  Runs inference using the fine-tuned model specifically for evaluating its performance on new data. This script is typically run after the model has been fine-tuned and it's limited to the context of fine-tuning and immediate evaluation.

- Inference in `inference` Directory:
  Provides a more comprehensive and flexible inference pipeline, which includes fetching input data from a database, running inference, and saving results to a database. This script can be used independently of the fine-tuning process and is suitable for large-scale inference tasks.


#### Similarity Measures
---

1. **Navigate to the `similarity_measures` directory**

2. **Open the Jupyter notebook to evaluate measures:**
   ```bash
   jupyter notebook Measure_Evaluation.ipynb
   ```
   
   Follow the instructions within the notebook to run the evaluations.

#### Elo
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
____

### Contributing

Feel free to contact us for cooperation; we will be glad to work together.

### License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License (CC-BY 4.0)</a>.

### Authors and Contact Information

Contact information will be published after the de-anonymization of the repository.

### Version History

* 0.1
    * Initial Release
