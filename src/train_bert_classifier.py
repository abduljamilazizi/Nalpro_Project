import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
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


class NewsGroupsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    set_seed(RANDOM_SEED)

    os.makedirs("results", exist_ok=True)

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, target_names = load_20newsgroups_data()

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    print("Creating datasets...")
    train_dataset = NewsGroupsDataset(X_train, y_train, tokenizer, BERT_MAX_LENGTH)
    val_dataset = NewsGroupsDataset(X_val, y_val, tokenizer, BERT_MAX_LENGTH)
    test_dataset = NewsGroupsDataset(X_test, y_test, tokenizer, BERT_MAX_LENGTH)

    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(target_names)
    )

    training_args = TrainingArguments(
        output_dir="results/bert_classifier_output",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training started...")
    trainer.train()

    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print(val_metrics)

    print("\nEvaluating on test set...")
    test_predictions = trainer.predict(test_dataset)
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

    with open("results/bert_classifier_results.txt", "w", encoding="utf-8") as f:
        f.write("BERT-base Classification Results\n")
        f.write("===============================\n\n")
        f.write(f"Model: {BERT_MODEL_NAME}\n")
        f.write(f"Max Length: {BERT_MAX_LENGTH}\n")
        f.write(f"Batch Size: {BERT_BATCH_SIZE}\n")
        f.write(f"Learning Rate: {BERT_LEARNING_RATE}\n")
        f.write(f"Epochs: {BERT_EPOCHS}\n")
        f.write(f"Weight Decay: {BERT_WEIGHT_DECAY}\n\n")
        f.write("Validation Metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Macro F1: {test_macro_f1:.4f}\n")
        f.write(f"Test Weighted F1: {test_weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("\nResults saved to results/bert_classifier_results.txt")


if __name__ == "__main__":
    main()