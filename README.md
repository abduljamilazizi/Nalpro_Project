# NALPRO Project: 20 Newsgroups Text Classification

This repository contains my Natural Language Processing course project on multi-class text classification using the 20 Newsgroups dataset. The project compares feature-based, transformer-based, and prompt-based methods on the same task and includes one bonus experiment with parameter-efficient fine-tuning.

## Project Summary

The work is organized into four main parts plus one bonus experiment:

### Part 1: Simple Neural Network with Classical Text Representations
This part uses the same MLP classifier with different input representations:
- Word2Vec + MLP
- TF-IDF + MLP
- TF-IDF Bigram + MLP

### Part 2: Direct BERT Fine-Tuning
This part fine-tunes `bert-base-uncased` directly for 20-class text classification.

### Part 3: BERT MLM Adaptation then Classification
This part first adapts BERT to the 20 Newsgroups corpus using masked language modeling and then fine-tunes it for classification.

### Part 4: Llama Zero-Shot and Few-Shot Prompting
This part evaluates an instruction-tuned LLM in:
- zero-shot classification
- few-shot classification

### Bonus: QLoRA Experiment
This bonus part explores lightweight parameter-efficient fine-tuning under limited GPU resources using a smaller instruct model and QLoRA-style training.

## Main Findings

- TF-IDF + MLP achieved the best overall result in this project setup.
- Word2Vec improved significantly when trained for more epochs.
- MLM adaptation slightly improved BERT over direct fine-tuning.
- Few-shot prompting performed much better than zero-shot prompting.
- The bonus QLoRA experiment worked as a proof-of-concept under hardware and memory constraints.

## Repository Structure

```text
Nalpro_Project/
├── figures/
├── llama_zeroshot_fewshot_result/
├── results/
├── src/
├── .gitignore
├── README.md
├── requirements.txt
└── (optional supporting project files)
```

## Folder Description

### `figures/`
Contains important generated plots and visual outputs, especially for Part 1:
- TF-IDF training curves
- Word2Vec training curves
- PCA visualizations
- TF-IDF bigram experiment curves

### `results/`
Contains saved results from the experiments, such as:
- TF-IDF result files
- Word2Vec result files
- BERT result files
- MLM + BERT result files
- Llama result files
- bonus experiment result files if saved separately

### `src/`
Contains the main source code for the project.

Typical files include:
- `config.py`
- `data_utils.py`
- `preprocess.py`
- `train_tfidf_mlp.py`
- `train_word2vec_mlp.py`
- `extra_experiment.py`
- `train_bert_classifier.py`
- `train_bert_mlm_then_classifier.py`
- `llama_zero_few_shot_kaggle.py`
- `Qlora_experiment.py`
- `evaluate.py`

## Dataset

This project uses the `fetch_20newsgroups` dataset from `scikit-learn`.

The dataset is not stored directly in the repository. It is downloaded automatically when the scripts are run.

The dataset is used after removing:
- headers
- footers
- quoted replies

This helps make the classification task more realistic and reduces label leakage.

## Preprocessing Overview

Preprocessing depends on the experiment type.

### Classical Models
For Word2Vec and TF-IDF experiments, preprocessing includes:
- lowercasing
- removing punctuation
- removing digits
- removing extra whitespace
- tokenization for Word2Vec

### BERT-Based Models
For BERT and MLM + BERT experiments, preprocessing is lighter:
- remove headers, footers, and quotes
- tokenize with the BERT tokenizer
- truncate or pad to the required sequence length

### Llama Prompting
For zero-shot and few-shot prompting:
- remove headers, footers, and quotes
- normalize spaces
- shorten text for prompt-friendly input
- format examples carefully for prompting

### Bonus QLoRA
For the bonus experiment:
- text is formatted as instruction-style classification examples
- a reduced dataset subset is used
- evaluation is done under limited compute settings

## Experimental Parts

### Part 1: Word2Vec and TF-IDF with MLP
The same MLP architecture is used across all Part 1 experiments so that the main comparison is between representations rather than classifier design.

Experiments included:
- Word2Vec with short training
- Word2Vec with longer training
- TF-IDF
- TF-IDF with bigrams

### Part 2: Direct BERT Classification
A standard BERT classifier is fine-tuned directly on the classification task.

### Part 3: MLM Adaptation + BERT Classification
BERT is first adapted using masked language modeling on in-domain text and then fine-tuned for classification.

### Part 4: Zero-Shot and Few-Shot Prompting
A Llama-based instruction model is evaluated using prompts only, without updating model weights.

### Bonus: QLoRA
A lightweight QLoRA-style experiment is included as bonus work to demonstrate parameter-efficient fine-tuning under resource constraints.

## Evaluation Metrics

The project uses:
- Accuracy
- Macro F1
- Weighted F1

Macro F1 is especially important because it gives equal importance to all classes in the 20-class setting.

## Reproducibility

To improve reproducibility:
- fixed random seeds are used where possible
- the same dataset source is used across experiments
- consistent preprocessing is applied within each experiment type
- saved result files are included in the repository

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

Run scripts from the project root.

Example:

```bash
python src/preprocess.py
python src/train_tfidf_mlp.py
python src/train_word2vec_mlp.py
python src/extra_experiment.py
python src/train_bert_classifier.py
python src/train_bert_mlm_then_classifier.py
python src/llama_zero_few_shot_kaggle.py
python src/Qlora_experiment.py
```

Depending on your environment, some scripts may be intended for:
- local execution
- Jupyter
- Kaggle GPU
- other cloud notebooks

## Notes on Hardware and Constraints

Some parts of the project, especially Llama-based prompting and the bonus QLoRA experiment, were affected by hardware and memory limitations.

Because of this:
- some experiments were run on subsets
- some experiments used lighter settings
- the bonus QLoRA part should be interpreted as exploratory work rather than a direct large-scale comparison

## Final Report Coverage

This repository is intended to reflect the final project submission and covers:
- Part 1
- Part 2
- Part 3
- Part 4
- Bonus QLoRA experiment
- figures
- result files
- implementation scripts

## Repository Link

GitHub repository:
[https://github.com/abduljamilazizi/Nalpro_Project](https://github.com/abduljamilazizi/Nalpro_Project)

## Acknowledgment

This project was carried out under the supervision of Professor Dr. Forooz Shahbaz Avarvand.

AI tools were used as support for wording, organization, and explanation help. The experiments, implementation, results, and conclusions were reviewed and understood by the author.

## Final Note

This repository represents the final version of the NALPRO project and includes the required components of the course project together with the bonus experiment.
