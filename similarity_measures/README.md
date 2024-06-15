# Similarity Measure README

This project involves several components for evaluating text generation models using various similarity measures. Here's an overview of the main components and how they work together.

*For the purpose of this code, metric and measure are used synonomously but the chosen word preference is **measure** not metric*

## Components

### Libraries Used

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `nltk.translate.meteor_score`: For computing METEOR score.
- `rouge_score.rouge_scorer`: For computing ROUGE score.
- `nltk.translate.bleu_score.sentence_bleu`: For computing BLEU score.
- `bleurt.score.BleurtScorer`: For computing BLEURT score.

### Initialization

- `BleurtScorer`: Initializes the BLEURT scorer using the checkpoint "bleurt-base-128".
- Various imports for handling JSON, regular expressions (`re`), and time.

### Data Preprocessing (`DataPreprocessor` Class)

#### Methods:
- `extract_triples`: Extracts triplets from input strings based on a specific pattern.
- `convert_df_to_json`: Converts a DataFrame into JSON format, extracting triples from a specified column.

### Measure Computation (`Metrics` Class)

#### Methods:
- `compute_bleurt`: Computes BLEURT scores between candidate and reference texts.
- `compute_bleu`: Computes BLEU scores between candidate and reference texts.
- `compute_meteor`: Computes METEOR scores between candidate and reference texts.
- `compute_rouge`: Computes ROUGE scores between candidate and reference texts.
- `compute_verbatim`: Computes exact match scores between candidate and reference texts.

### Score Calculation (`ScoreCalculator` Class)

#### Methods:
- `compute_scores`: Computes scores using specified measures and thresholds.
- `find_node_mappings`: Finds mappings between predicted and ground truth nodes.
- `find_matching_edges`: Finds matching edges between predicted and ground truth edges.
- `calculate_f1_score`: Calculates the F1 score given true positives, false positives, and false negatives.
- `calculate_scores`: Calculates F1 scores across multiple models and metrics, using specified thresholds.

### Comparing Measures with Elo Rankings

- `compare_metrics_elo`: Compares F1 scores against Elo rankings using Spearman correlation.
- Uses Elo rankings and F1 scores to determine correlations and measures closer to Elo rankings.

## Usage

### Initial Setup

- Specify file paths and names for model annotations and Elo rankings.

### Measure Evaluation

- Evaluate measures (`bleurt`, `bleu`, `meteor`, `rouge`, `verbatim`) across multiple models.
- Tune thresholds for measures based on Elo rankings to find optimal settings.
- Calculate scores and correlations, outputting results for further analysis.

### Output

- Detailed results including F1 scores for each metric and model.
- Spearman correlations and comparisons with Elo rankings.

## Dependencies

Ensure the following dependencies are installed:

```bash
pip install numpy pandas nltk rouge-score bleurt scipy
```

As for installing specific bleurt checkpoints visit  [BLEURT Checkpoints](https://github.com/google-research/bleurt/blob/master/checkpoints.md) and for general BLEURT information visit [BLEURT](https://github.com/google-research/bleurt/tree/master).
