# NALPRO Project: 20 Newsgroups Text Classification

This repository contains my project work for the Natural Language Processing course using the **20 Newsgroups** dataset.

## Project Scope

The project is based on text classification using the dataset loaded with:

```python
from sklearn.datasets import fetch_20newsgroups
```

The completed work in this repository currently includes:

1. A simple neural network with two linear layers and a ReLU in between.
2. Input representations using:
   - Word2Vec
   - TF-IDF
3. A comparison of Word2Vec embeddings after different numbers of training epochs.
4. An additional experiment using TF-IDF with unigrams and bigrams.
5. Fine-tuning a BERT-base model for classification.
6. Fine-tuning BERT with masked language modeling first, then fine-tuning it again for classification.

## Current Project Status

The following parts are completed:

- Part 1: Simple neural network experiments
- Part 2: BERT-base fine-tuning for classification
- Part 3: BERT masked language modeling adaptation, then classification fine-tuning

The following part is not yet completed in this repository:

- Part 4: Llama-3 zero-shot and few-shot experiments

## Repository Structure

```text
nalpro_project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── figures/
├── reports/
├── notebooks/
├── results/
└── src/
    ├── config.py
    ├── data_utils.py
    ├── preprocess.py
    ├── train_tfidf_mlp.py
    ├── train_word2vec_mlp.py
    ├── extra_experiment.py
    ├── train_bert_classifier.py
    ├── train_bert_mlm_then_classifier.py
    ├── llama_zero_few_shot.py
    └── evaluate.py
```

## Dataset

This project uses the **20 Newsgroups** dataset from `scikit-learn`.

The dataset is not uploaded to this repository. It is downloaded programmatically when running the scripts.

## Preprocessing

The dataset is loaded using:

```python
remove=("headers", "footers", "quotes")
```

This was used to reduce noise from:
- email headers
- signatures and footers
- quoted replies

Additional preprocessing includes:
- lowercasing
- removing punctuation
- removing digits
- removing extra whitespace

For Word2Vec, tokenization is also applied.

## Methods Implemented

### 1. TF-IDF + MLP
A simple neural network was trained using TF-IDF document vectors as input.

Script:
- `src/train_tfidf_mlp.py`

Outputs:
- result file in `results/`
- training and validation plots in `figures/`

### 2. Word2Vec + MLP
A Word2Vec model was trained on the corpus and document representations were created by averaging word embeddings. These document vectors were used as input to the same neural network.

Script:
- `src/train_word2vec_mlp.py`

Outputs:
- result files in `results/`
- training and validation plots in `figures/`
- PCA plots comparing embeddings after 1 epoch and 10 epochs

### 3. Extra Experiment: TF-IDF with Bigrams
An additional experiment was performed using TF-IDF with unigram and bigram features while keeping the same neural network architecture.

Script:
- `src/extra_experiment.py`

Outputs:
- result file in `results/`
- training and validation plots in `figures/`

### 4. BERT-base Fine-tuning
A `bert-base-uncased` model was fine-tuned on the classification task.

Script:
- `src/train_bert_classifier.py`

Outputs:
- result file in `results/`

### 5. BERT MLM Adaptation Then Classification
BERT was first adapted using masked language modeling on a subset of the training corpus, then fine-tuned for classification.

Script:
- `src/train_bert_mlm_then_classifier.py`

Outputs:
- result file in `results/`

## Main Configuration

The main experiment settings are stored in:

- `src/config.py`

This file includes:
- random seed
- TF-IDF settings
- MLP settings
- Word2Vec settings
- BERT settings

## How to Run

### 1. Create and activate the virtual environment

On Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bat
pip install -r requirements.txt
```

## Run the completed parts

### Load and inspect the dataset

```bat
python src\data_utils.py
```

### Test preprocessing

```bat
python src\preprocess.py
```

### Run TF-IDF + MLP

```bat
python src\train_tfidf_mlp.py
```

### Run Word2Vec + MLP

```bat
python src\train_word2vec_mlp.py
```

### Run the extra experiment

```bat
python src\extra_experiment.py
```

### Run BERT classification

```bat
python src\train_bert_classifier.py
```

### Run MLM then classification

```bat
python src\train_bert_mlm_then_classifier.py
```

## Results and Figures

The repository includes:
- experiment result summaries in the `results/` folder
- plots and visualizations in the `figures/` folder

Examples:
- TF-IDF training curves
- Word2Vec training curves
- Word2Vec PCA visualization after 1 and 10 epochs
- TF-IDF bigram experiment curves
- BERT and MLM-based result summaries

## Results Summary

| Method | Representation / Setup | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---|---:|---:|---:|---|
| TF-IDF + MLP | TF-IDF unigram features + 2-layer MLP | **0.6620** | **0.6530** | **0.6635** | Strong classical baseline |
| Word2Vec + MLP (1 epoch) | Mean Word2Vec document vectors + 2-layer MLP | **0.2262** | **0.1749** | **0.1815** | Very weak after short embedding training |
| Word2Vec + MLP (10 epochs) | Mean Word2Vec document vectors + 2-layer MLP | **0.5068** | **0.4804** | **0.4948** | Clear improvement over 1 epoch |
| TF-IDF + MLP (bigrams) | TF-IDF unigram + bigram features + 2-layer MLP | **0.6599** | **0.6514** | **0.6616** | Very similar to unigram TF-IDF |
| BERT-base classifier | `bert-base-uncased`, direct classification fine-tuning | **0.6500** | **0.6300** | **0.6500** | CPU-friendly BERT setup |
| BERT MLM → classifier | MLM adaptation first, then classification fine-tuning | **0.6600** | **0.6390** | **0.6545** | Small improvement over direct BERT |

This table reflects the completed parts of the project up to Part 3. The Llama-3 zero-shot and few-shot experiments are not yet included.

## Tools Used

This project uses:
- Python
- PyTorch
- scikit-learn
- gensim
- transformers
- datasets
- matplotlib
- numpy
- pandas

## AI Usage

AI tools were used for assistance in:
- code structuring
- debugging
- explanation and planning support

All code used in this repository was reviewed and understood before use.

## External Notes

- The dataset itself is not uploaded to the repository.
- Large model checkpoints are not included in the repository.
- This repository currently contains the completed work up to Part 3 of the project.

## Final Report

The final scientific report will be placed in the `reports/` folder.

## Author

Abdul Jamil Azizi
