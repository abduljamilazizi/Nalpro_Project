import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import (
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_HIDDEN_DIM,
    MLP_LEARNING_RATE,
    RANDOM_SEED,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
)
from data_utils import load_20newsgroups_data
from preprocess import preprocess_corpus


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dataloader(features, labels, batch_size, shuffle=False):
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, macro_f1, weighted_f1, all_labels, all_preds


def save_plots(train_losses, val_losses, val_accuracies, val_macro_f1s):
    os.makedirs("figures", exist_ok=True)

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TF-IDF + MLP Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/tfidf_mlp_loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
    plt.plot(epochs_range, val_macro_f1s, label="Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("TF-IDF + MLP Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/tfidf_mlp_validation_metrics.png")
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, target_names = load_20newsgroups_data()

    print("Preprocessing text...")
    X_train_clean = preprocess_corpus(X_train)
    X_val_clean = preprocess_corpus(X_val)
    X_test_clean = preprocess_corpus(X_test)

    print("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words=TFIDF_STOP_WORDS,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_clean).toarray()
    X_val_tfidf = vectorizer.transform(X_val_clean).toarray()
    X_test_tfidf = vectorizer.transform(X_test_clean).toarray()

    print(f"TF-IDF train shape: {X_train_tfidf.shape}")
    print(f"TF-IDF val shape: {X_val_tfidf.shape}")
    print(f"TF-IDF test shape: {X_test_tfidf.shape}")

    train_loader = create_dataloader(X_train_tfidf, y_train, MLP_BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val_tfidf, y_val, MLP_BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(X_test_tfidf, y_test, MLP_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMLP(
        input_dim=X_train_tfidf.shape[1],
        hidden_dim=MLP_HIDDEN_DIM,
        num_classes=len(target_names),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_macro_f1s = []

    print("Training started...")
    for epoch in range(MLP_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, val_macro_f1, val_weighted_f1, _, _ = evaluate_model(
            model, val_loader, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_macro_f1s.append(val_macro_f1)

        print(
            f"Epoch {epoch + 1}/{MLP_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f} | "
            f"Val Weighted F1: {val_weighted_f1:.4f}"
        )

    print("\nEvaluating on test set...")
    test_loss, test_acc, test_macro_f1, test_weighted_f1, test_labels, test_preds = evaluate_model(
        model, test_loader, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}")

    report = classification_report(test_labels, test_preds, target_names=target_names)
    print("\nClassification Report:\n")
    print(report)

    with open("results/tfidf_mlp_results.txt", "w", encoding="utf-8") as f:
        f.write("TF-IDF + MLP Results\n")
        f.write("====================\n\n")
        f.write(f"TF-IDF train shape: {X_train_tfidf.shape}\n")
        f.write(f"TF-IDF val shape: {X_val_tfidf.shape}\n")
        f.write(f"TF-IDF test shape: {X_test_tfidf.shape}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro F1: {test_macro_f1:.4f}\n")
        f.write(f"Test Weighted F1: {test_weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    save_plots(train_losses, val_losses, val_accuracies, val_macro_f1s)

    print("\nResults saved to results/tfidf_mlp_results.txt")
    print("Plots saved to figures/tfidf_mlp_loss_curve.png")
    print("Plots saved to figures/tfidf_mlp_validation_metrics.png")


if __name__ == "__main__":
    main()