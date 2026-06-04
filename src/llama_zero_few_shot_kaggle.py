# Note:
# This script is intended for execution inside a Kaggle notebook environment.
# It uses kaggle_secrets for Hugging Face authentication.
# Part 4: Llama-3 zero-shot and few-shot experiments
# Final version executed on Kaggle GPU (T4 x2) with 4-bit quantization
# Model: meta-llama/Meta-Llama-3-8B-Instruct
# Output files:
# - results/llama_zero_few_shot_results.txt
# - results/llama_zero_shot_predictions.csv
# - results/llama_few_shot_predictions.csv
# Dependencies are installed in Kaggle notebook or listed in requirements/README.

import re
import random
import numpy as np
import pandas as pd
import torch

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# -----------------------------
# Config
# -----------------------------
RANDOM_SEED = 42
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_MAX_NEW_TOKENS = 8
LLAMA_TEST_SUBSET = 50
LLAMA_FEW_SHOT_EXAMPLES_PER_CLASS = 1


# -----------------------------
# Hugging Face login
# -----------------------------
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
print("Hugging Face login successful")


# -----------------------------
# Set seed
# -----------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -----------------------------
# Load dataset
# -----------------------------
train_dataset = fetch_20newsgroups(
    subset="train",
    shuffle=True,
    random_state=RANDOM_SEED,
    remove=("headers", "footers", "quotes")
)

test_dataset = fetch_20newsgroups(
    subset="test",
    shuffle=True,
    random_state=RANDOM_SEED,
    remove=("headers", "footers", "quotes")
)

X_train_full = train_dataset.data
y_train_full = train_dataset.target
X_test = test_dataset.data
y_test = test_dataset.target
target_names = train_dataset.target_names

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_train_full
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))
print("Classes:", len(target_names))


# -----------------------------
# Helper functions
# -----------------------------
def normalize_prediction(raw_text, target_names):
    text = raw_text.strip().lower()
    text = text.split("\n")[0].strip()

    for label in target_names:
        if label.lower() == text:
            return label

    for label in target_names:
        if label.lower() in text:
            return label

    cleaned = re.sub(r"[^a-z0-9\.\-_ ]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    for label in target_names:
        if label.lower() == cleaned:
            return label

    for label in target_names:
        if label.lower() in cleaned:
            return label

    return None


def build_zero_shot_prompt(text, target_names):
    labels_text = ", ".join(target_names)
    prompt = f"""Classify the text into exactly one label from this list:
{labels_text}

Return only the label name.

Text: {text[:500]}
Label:"""
    return prompt


def build_few_shot_examples(X_train, y_train, target_names, examples_per_class=1, max_chars=120):
    examples = []
    counts = {label: 0 for label in target_names}

    for text, y in zip(X_train, y_train):
        label = target_names[y]
        if counts[label] < examples_per_class:
            short_text = text[:max_chars].replace("\n", " ").strip()
            examples.append((short_text, label))
            counts[label] += 1

        if all(counts[label] >= examples_per_class for label in target_names):
            break

    return examples


def build_few_shot_prompt(text, target_names, few_shot_examples):
    labels_text = ", ".join(target_names)

    prompt = f"""Classify the text into exactly one label from this list:
{labels_text}

Return only the label name.

"""

    for example_text, example_label in few_shot_examples[:5]:
        prompt += f"Text: {example_text}\nLabel: {example_label}\n\n"

    prompt += f"Text: {text[:500]}\nLabel:"
    return prompt


def select_test_subset(X_test, y_test, target_names, subset_size):
    data = list(zip(X_test, y_test))
    random.shuffle(data)
    subset = data[:subset_size]

    X_subset = [x for x, _ in subset]
    y_subset_names = [target_names[y] for _, y in subset]

    return X_subset, y_subset_names


def evaluate_predictions(y_true_names, y_pred_names, target_names):
    final_preds = []
    invalid_count = 0

    for pred in y_pred_names:
        if pred is None:
            final_preds.append(target_names[0])
            invalid_count += 1
        else:
            final_preds.append(pred)

    accuracy = accuracy_score(y_true_names, final_preds)
    macro_f1 = f1_score(
        y_true_names,
        final_preds,
        average="macro",
        labels=target_names,
        zero_division=0
    )
    weighted_f1 = f1_score(
        y_true_names,
        final_preds,
        average="weighted",
        labels=target_names,
        zero_division=0
    )

    report = classification_report(
        y_true_names,
        final_preds,
        labels=target_names,
        target_names=target_names,
        zero_division=0,
    )

    return accuracy, macro_f1, weighted_f1, report, invalid_count, final_preds


# -----------------------------
# Prepare subset and examples
# -----------------------------
X_test_subset, y_test_subset_names = select_test_subset(
    X_test, y_test, target_names, LLAMA_TEST_SUBSET
)

few_shot_examples = build_few_shot_examples(
    X_train,
    y_train,
    target_names,
    examples_per_class=LLAMA_FEW_SHOT_EXAMPLES_PER_CLASS,
)

print("Test subset size:", len(X_test_subset))
print("Few-shot examples:", len(few_shot_examples))


# -----------------------------
# Load model in 4-bit
# -----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Optional: silence harmless generation-config warning
model.generation_config.temperature = None
model.generation_config.top_p = None


def run_generation(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=LLAMA_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated.split("\n")[0].strip()


# -----------------------------
# Zero-shot evaluation
# -----------------------------
zero_raw_outputs = []
zero_preds = []

print("\nRunning zero-shot...")
for i, text in enumerate(X_test_subset, start=1):
    prompt = build_zero_shot_prompt(text, target_names)
    raw_output = run_generation(prompt)
    pred = normalize_prediction(raw_output, target_names)

    zero_raw_outputs.append(raw_output)
    zero_preds.append(pred)

    print(f"[Zero-shot] {i}/{len(X_test_subset)} | Raw: {raw_output} | Pred: {pred}")

zero_acc, zero_macro_f1, zero_weighted_f1, zero_report, zero_invalid, zero_final_preds = evaluate_predictions(
    y_test_subset_names, zero_preds, target_names
)


# -----------------------------
# Few-shot evaluation
# -----------------------------
few_raw_outputs = []
few_preds = []

print("\nRunning few-shot...")
for i, text in enumerate(X_test_subset, start=1):
    prompt = build_few_shot_prompt(text, target_names, few_shot_examples)
    raw_output = run_generation(prompt)
    pred = normalize_prediction(raw_output, target_names)

    few_raw_outputs.append(raw_output)
    few_preds.append(pred)

    print(f"[Few-shot] {i}/{len(X_test_subset)} | Raw: {raw_output} | Pred: {pred}")

few_acc, few_macro_f1, few_weighted_f1, few_report, few_invalid, few_final_preds = evaluate_predictions(
    y_test_subset_names, few_preds, target_names
)


# -----------------------------
# Save outputs
# -----------------------------
zero_df = pd.DataFrame({
    "text_preview": [t[:300].replace("\n", " ") for t in X_test_subset],
    "true_label": y_test_subset_names,
    "raw_output": zero_raw_outputs,
    "normalized_prediction": zero_final_preds,
})
zero_df.to_csv("/kaggle/working/llama_zero_shot_predictions.csv", index=False)

few_df = pd.DataFrame({
    "text_preview": [t[:300].replace("\n", " ") for t in X_test_subset],
    "true_label": y_test_subset_names,
    "raw_output": few_raw_outputs,
    "normalized_prediction": few_final_preds,
})
few_df.to_csv("/kaggle/working/llama_few_shot_predictions.csv", index=False)

with open("/kaggle/working/llama_zero_few_shot_results.txt", "w", encoding="utf-8") as f:
    f.write("Llama-3 Zero-shot and Few-shot Results\n")
    f.write("=====================================\n\n")
    f.write(f"Model: {LLAMA_MODEL_NAME}\n")
    f.write(f"Test subset size: {LLAMA_TEST_SUBSET}\n")
    f.write(f"Max new tokens: {LLAMA_MAX_NEW_TOKENS}\n")
    f.write(f"Few-shot examples per class: {LLAMA_FEW_SHOT_EXAMPLES_PER_CLASS}\n\n")

    f.write("Zero-shot Results\n")
    f.write("-----------------\n")
    f.write(f"Accuracy: {zero_acc:.4f}\n")
    f.write(f"Macro F1: {zero_macro_f1:.4f}\n")
    f.write(f"Weighted F1: {zero_weighted_f1:.4f}\n")
    f.write(f"Invalid predictions: {zero_invalid}\n\n")
    f.write("Zero-shot Classification Report:\n")
    f.write(zero_report)
    f.write("\n\n")

    f.write("Few-shot Results\n")
    f.write("----------------\n")
    f.write(f"Accuracy: {few_acc:.4f}\n")
    f.write(f"Macro F1: {few_macro_f1:.4f}\n")
    f.write(f"Weighted F1: {few_weighted_f1:.4f}\n")
    f.write(f"Invalid predictions: {few_invalid}\n\n")
    f.write("Few-shot Classification Report:\n")
    f.write(few_report)

print("\nSaved files:")
print("/kaggle/working/llama_zero_few_shot_results.txt")
print("/kaggle/working/llama_zero_shot_predictions.csv")
print("/kaggle/working/llama_few_shot_predictions.csv")

print("\nFinal scores:")
print("Zero-shot Accuracy:", round(zero_acc, 4))
print("Zero-shot Macro F1:", round(zero_macro_f1, 4))
print("Few-shot Accuracy:", round(few_acc, 4))
print("Few-shot Macro F1:", round(few_macro_f1, 4))
