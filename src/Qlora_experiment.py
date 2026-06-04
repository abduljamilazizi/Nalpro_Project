import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import re
import math
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# =========================================================
# Config
# =========================================================
RANDOM_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_TEXT_CHARS = 220
MAX_SEQ_LENGTH = 192

TRAIN_SUBSET = 100
VAL_SUBSET = 20
TEST_SUBSET = 20

NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 1
WARMUP_STEPS = 3
WEIGHT_DECAY = 0.01

LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

OUTPUT_DIR = "/kaggle/working/qwen_qlora_scored"
RESULTS_TXT = "/kaggle/working/qwen_qlora_scored_results.txt"
PREDICTIONS_CSV = "/kaggle/working/qwen_qlora_scored_predictions.csv"


# =========================================================
# Reproducibility
# =========================================================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())


# =========================================================
# Hugging Face login
# =========================================================
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)


# =========================================================
# Load dataset
# =========================================================
train_data = fetch_20newsgroups(
    subset="train",
    shuffle=True,
    random_state=RANDOM_SEED,
    remove=("headers", "footers", "quotes"),
)

test_data = fetch_20newsgroups(
    subset="test",
    shuffle=True,
    random_state=RANDOM_SEED,
    remove=("headers", "footers", "quotes"),
)

X_train_full = train_data.data
y_train_full = train_data.target
X_test_full = test_data.data
y_test_full = test_data.target
target_names = train_data.target_names

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_train_full,
)

train_idx = np.random.choice(len(X_train), size=min(TRAIN_SUBSET, len(X_train)), replace=False)
val_idx = np.random.choice(len(X_val), size=min(VAL_SUBSET, len(X_val)), replace=False)
test_idx = np.random.choice(len(X_test_full), size=min(TEST_SUBSET, len(X_test_full)), replace=False)

X_train = [X_train[i] for i in train_idx]
y_train = [y_train[i] for i in train_idx]

X_val = [X_val[i] for i in val_idx]
y_val = [y_val[i] for i in val_idx]

X_test = [X_test_full[i] for i in test_idx]
y_test = [y_test[i] for i in test_idx]

print("Train subset:", len(X_train))
print("Val subset:", len(X_val))
print("Test subset:", len(X_test))
print("Classes:", len(target_names))


# =========================================================
# Prompt formatting
# =========================================================
label_list_text = " | ".join(target_names)

def clean_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:MAX_TEXT_CHARS]

def build_train_example(text: str, label: str) -> str:
    text = clean_text(text)
    return f"""Choose exactly one label from this list:
{label_list_text}

Return only one label exactly as written above.

Text: {text}
Label: {label}"""

def build_infer_prompt(text: str) -> str:
    text = clean_text(text)
    return f"""Choose exactly one label from this list:
{label_list_text}

Return only one label exactly as written above.

Text: {text}
Label:"""

train_texts = [build_train_example(x, target_names[y]) for x, y in zip(X_train, y_train)]
val_texts = [build_train_example(x, target_names[y]) for x, y in zip(X_val, y_val)]

train_ds = Dataset.from_dict({"text": train_texts})
val_ds = Dataset.from_dict({"text": val_texts})


# =========================================================
# 4-bit model load
# =========================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

model.config.use_cache = False

gc.collect()
torch.cuda.empty_cache()


# =========================================================
# LoRA config
# =========================================================
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)


# =========================================================
# Training config
# =========================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    logging_steps=5,
    fp16=False,
    bf16=False,
    report_to="none",
    save_strategy="no",
    eval_strategy="no",
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("Starting QLoRA fine-tuning...")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

gc.collect()
torch.cuda.empty_cache()


# =========================================================
# Scored evaluation instead of free generation
# =========================================================
def score_label(prompt: str, label: str) -> float:
    """
    Returns average log-probability of the candidate label given the prompt.
    Higher is better.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"].to(0)
    label_ids = tokenizer(" " + label, return_tensors="pt", add_special_tokens=False)["input_ids"].to(0)

    input_ids = torch.cat([prompt_ids, label_ids], dim=1)

    if input_ids.shape[1] > MAX_SEQ_LENGTH:
        input_ids = input_ids[:, -MAX_SEQ_LENGTH:]

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Score only the label tokens
    label_len = label_ids.shape[1]
    start = input_ids.shape[1] - label_len

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    total_logprob = 0.0
    count = 0

    for pos in range(start, input_ids.shape[1]):
        if pos == 0:
            continue
        token_id = input_ids[0, pos]
        token_logprob = log_probs[0, pos - 1, token_id].item()
        total_logprob += token_logprob
        count += 1

    return total_logprob / max(count, 1)

def predict_label(text: str, target_names):
    prompt = build_infer_prompt(text)
    scores = []

    for label in target_names:
        s = score_label(prompt, label)
        scores.append((label, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0], scores

true_labels = [target_names[y] for y in y_test]
pred_labels = []
top_scores = []

print("Evaluating with label scoring...")
for i, text in enumerate(X_test, start=1):
    pred, scores = predict_label(text, target_names)
    pred_labels.append(pred)
    top_scores.append(scores[:3])
    print(f"[{i}/{len(X_test)}] Pred: {pred} | Top-3: {scores[:3]}")

acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro", labels=target_names, zero_division=0)
weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", labels=target_names, zero_division=0)
report = classification_report(true_labels, pred_labels, labels=target_names, target_names=target_names, zero_division=0)

print("Accuracy:", acc)
print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)
print(report)

pred_df = pd.DataFrame({
    "text_preview": [clean_text(t)[:300] for t in X_test],
    "true_label": true_labels,
    "predicted_label": pred_labels,
    "top3_scores": [str(s) for s in top_scores],
})
pred_df.to_csv(PREDICTIONS_CSV, index=False)

with open(RESULTS_TXT, "w", encoding="utf-8") as f:
    f.write("Qwen QLoRA Scored Classification Results\n")
    f.write("========================================\n\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Train subset: {len(X_train)}\n")
    f.write(f"Val subset: {len(X_val)}\n")
    f.write(f"Test subset: {len(X_test)}\n")
    f.write(f"Epochs: {NUM_EPOCHS}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"LoRA r: {LORA_R}\n")
    f.write(f"LoRA alpha: {LORA_ALPHA}\n")
    f.write(f"LoRA dropout: {LORA_DROPOUT}\n\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Saved:")
print(RESULTS_TXT)
print(PREDICTIONS_CSV)
print(OUTPUT_DIR)
