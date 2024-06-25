# MATCH  
(Measure Alignment Through Comparative Human Judgement) for FCM Extraction
---

## Description  
Recent advances in natural language processing (NLP) have enhanced the automated extraction of fuzzy cognitive maps (FCMs) from text. This project presents MATCH (Measure Alignment Through Comparative Human Judgement), a novel validation method addressing the high expressiveness and partial correctness in FCM extraction. MATCH employs the Elo rating system to rank annotations through pairwise comparisons and evaluates various thresholded similarity measures against human judgment. We then apply these measures to evaluate state-of-the-art open-source large language models (LLMs) that have been fine-tuned for FCM extraction. Our findings indicate that:

1. MATCH'ed measures successfully quantify qualitative improvements in model inferences in a semi-automated fashion.  
2. Fine-tuned LLMs are more effective at generating FCMs from text than their foundation model counterparts.

This study underscores the potential benefits of developing human-aligned evaluation measures and provides a robust methodology for validating graph-based NLP predictions.

### Project Structure

```
MATCH/
├── fine-tuning/  
│ ├── finetuning.py  
│ ├── inference.py  
│ ├── mistral_finetune_own_data_tutorial.ipynb  
│ └── utils.py  
│  
├── in_context_learning/  
│ ├── three_shot_prompting.py  
│ ├── utils.py  
│ └── zero_shot_prompting.py  
│  
├── inference/  
│ ├── do_inference.py  
│ ├── inference.py  
│ └── save_inference_to_db.py  
│  
├── similarity_measures/  
│ ├── Measure_Evaluation.ipynb  
│ └── README.md  
│  
├── user_interface/  
│ ├── assets/  
│ ├── DashUI-Data-Labeling.py  
│ ├── DashUI-ELO-Comparison.py  
│ ├── DashUI-ELO-Data-Labeling.py  
│ ├── DashUI-ELO-inter-rater-reliability.py  
│ ├── INSTALL.md  
│ ├── README_data-labeling.md  
│ ├── Split-Creation.py  
│ ├── interagreement_splits.py  
│ └── requirements.txt
```

### A Note on this Repository

This repository contains the code and resources associated with the paper "MATCH: Measure Alignment Through Comparative Human Judgement
for FCM Extraction". The content is organized into various directories, each serving a specific purpose related to the project's objectives. As the paper is currently under review for the EMNLP conference, the repository remains anonymous until publication.

Upon acceptance and publication of the paper, the repository will be de-anonymized, and further details, including contact information and license, will be updated accordingly. We encourage collaboration and welcome contributions from the community. Feel free to reach out and cooperate with us; we will be glad to work together.

Please note that some components, especially those in the `user_interface` directory, require specific setup and dependencies as outlined in the respective README and INSTALL files. Ensure you follow the provided instructions for a seamless setup and usage experience.

### Dependencies

This project requires the following dependencies to run properly:  
```
argparse
torch
datasets
transformers

```

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


## Contributing
____
Feel free to contact us for cooperation; we will be glad to work together.


## License
____


## Authors and Contact Information
____
Contact information will be published after the de-anonymization of the repository.


## Version History
____
* 0.1
    * Initial Release
