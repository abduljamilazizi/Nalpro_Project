import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    BERT_WEIGHT_DECAY,
    RANDOM_SEED,
)
from data_utils import load_20newsgroups_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    set_seed(RANDOM_SEED)
    os.makedirs("results", exist_ok=True)

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, target_names = load_20newsgroups_data()

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # Smaller subset for CPU-friendly MLM adaptation
    mlm_subset_size = 2000
    X_train_mlm = X_train[:mlm_subset_size]

    print(f"Using MLM subset size: {len(X_train_mlm)}")

    print("Preparing MLM dataset...")
    mlm_dataset = MLMDataset(X_train_mlm, tokenizer, BERT_MAX_LENGTH)

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    print("Loading BERT for MLM...")
    mlm_model = BertForMaskedLM.from_pretrained(BERT_MODEL_NAME)

    mlm_args = TrainingArguments(
    output_dir="results/bert_mlm_output",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    seed=RANDOM_SEED,
)

    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_args,
        train_dataset=mlm_dataset,
        data_collator=mlm_collator,
    )

    print("Starting MLM adaptation...")
    mlm_trainer.train()

    adapted_model_path = "results/bert_mlm_adapted"
    mlm_model.save_pretrained(adapted_model_path)
    tokenizer.save_pretrained(adapted_model_path)
    print(f"Adapted MLM model saved to: {adapted_model_path}")

    print("Preparing classification datasets...")
    train_dataset = ClassificationDataset(X_train, y_train, tokenizer, BERT_MAX_LENGTH)
    val_dataset = ClassificationDataset(X_val, y_val, tokenizer, BERT_MAX_LENGTH)
    test_dataset = ClassificationDataset(X_test, y_test, tokenizer, BERT_MAX_LENGTH)

    print("Loading adapted model for classification...")
    clf_model = BertForSequenceClassification.from_pretrained(
        adapted_model_path,
        num_labels=len(target_names),
    )

    clf_args = TrainingArguments(
        output_dir="results/bert_mlm_then_classifier_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=BERT_LEARNING_RATE,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        num_train_epochs=BERT_EPOCHS,
        weight_decay=BERT_WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_SEED,
    )

    clf_trainer = Trainer(
        model=clf_model,
        args=clf_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting classification fine-tuning...")
    clf_trainer.train()

    print("Evaluating on validation set...")
    val_metrics = clf_trainer.evaluate(eval_dataset=val_dataset)
    print(val_metrics)

    print("Evaluating on test set...")
    test_predictions = clf_trainer.predict(test_dataset)
    test_logits = test_predictions.predictions
    test_preds = np.argmax(test_logits, axis=1)

    test_accuracy = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted")

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}")

    report = classification_report(y_test, test_preds, target_names=target_names)
    print("\nClassification Report:\n")
    print(report)

    with open("results/bert_mlm_then_classifier_results.txt", "w", encoding="utf-8") as f:
        f.write("BERT MLM Adaptation Then Classification Results\n")
        f.write("===============================================\n\n")
        f.write(f"Base Model: {BERT_MODEL_NAME}\n")
        f.write(f"MLM subset size: {mlm_subset_size}\n")
        f.write("MLM epochs: 1\n")
        f.write(f"Classification max length: {BERT_MAX_LENGTH}\n")
        f.write(f"Classification batch size: {BERT_BATCH_SIZE}\n")
        f.write(f"Classification learning rate: {BERT_LEARNING_RATE}\n")
        f.write(f"Classification epochs: {BERT_EPOCHS}\n\n")
        f.write("Validation Metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Macro F1: {test_macro_f1:.4f}\n")
        f.write(f"Test Weighted F1: {test_weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("\nResults saved to results/bert_mlm_then_classifier_results.txt")


if __name__ == "__main__":
    main()