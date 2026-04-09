import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import (
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_HIDDEN_DIM,
    MLP_LEARNING_RATE,
    RANDOM_SEED,
    W2V_EPOCHS_LONG,
    W2V_EPOCHS_SHORT,
    W2V_MIN_COUNT,
    W2V_SG,
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    W2V_WORKERS,
)
from data_utils import load_20newsgroups_data
from preprocess import tokenize_corpus


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


def document_to_vector(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def corpus_to_vectors(tokenized_corpus, model, vector_size):
    return np.array([document_to_vector(tokens, model, vector_size) for tokens in tokenized_corpus])


def train_mlp(train_features, y_train, val_features, y_val, test_features, y_test, num_classes):
    train_loader = create_dataloader(train_features, y_train, MLP_BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_features, y_val, MLP_BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test_features, y_test, MLP_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMLP(
        input_dim=train_features.shape[1],
        hidden_dim=MLP_HIDDEN_DIM,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_macro_f1s = []

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
        val_loss, val_acc, val_macro_f1, _, _, _ = evaluate_model(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_macro_f1s.append(val_macro_f1)

        print(
            f"Epoch {epoch + 1}/{MLP_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

    test_loss, test_acc, test_macro_f1, test_weighted_f1, test_labels, test_preds = evaluate_model(
        model, test_loader, device
    )

    return {
        "device": device,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_macro_f1s": val_macro_f1s,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_labels": test_labels,
        "test_preds": test_preds,
    }


def save_training_plots(train_losses, val_losses, val_accuracies, val_macro_f1s, prefix):
    os.makedirs("figures", exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{prefix.lower()}_loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
    plt.plot(epochs_range, val_macro_f1s, label="Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{prefix} Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{prefix.lower()}_validation_metrics.png")
    plt.close()


def plot_word_embeddings(model_short, model_long):
    os.makedirs("figures", exist_ok=True)

    words = [
        "computer", "graphics", "windows", "hockey", "baseball",
        "god", "jesus", "space", "gun", "car"
    ]

    short_words = [w for w in words if w in model_short.wv]
    long_words = [w for w in words if w in model_long.wv]
    common_words = [w for w in short_words if w in long_words]

    short_vectors = np.array([model_short.wv[w] for w in common_words])
    long_vectors = np.array([model_long.wv[w] for w in common_words])

    pca_short = PCA(n_components=2)
    reduced_short = pca_short.fit_transform(short_vectors)

    pca_long = PCA(n_components=2)
    reduced_long = pca_long.fit_transform(long_vectors)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(common_words):
        plt.scatter(reduced_short[i, 0], reduced_short[i, 1])
        plt.text(reduced_short[i, 0], reduced_short[i, 1], word)
    plt.title("Word2Vec Embeddings After 1 Epoch")
    plt.tight_layout()
    plt.savefig("figures/word2vec_epoch_1_pca.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(common_words):
        plt.scatter(reduced_long[i, 0], reduced_long[i, 1])
        plt.text(reduced_long[i, 0], reduced_long[i, 1], word)
    plt.title("Word2Vec Embeddings After 10 Epochs")
    plt.tight_layout()
    plt.savefig("figures/word2vec_epoch_10_pca.png")
    plt.close()


def save_results(filename, title, result_dict, target_names):
    report = classification_report(
        result_dict["test_labels"],
        result_dict["test_preds"],
        target_names=target_names
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write(f"Device: {result_dict['device']}\n")
        f.write(f"Test Loss: {result_dict['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {result_dict['test_acc']:.4f}\n")
        f.write(f"Test Macro F1: {result_dict['test_macro_f1']:.4f}\n")
        f.write(f"Test Weighted F1: {result_dict['test_weighted_f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, target_names = load_20newsgroups_data()

    print("Tokenizing text...")
    X_train_tokens = tokenize_corpus(X_train)
    X_val_tokens = tokenize_corpus(X_val)
    X_test_tokens = tokenize_corpus(X_test)

    print("Training Word2Vec model for 1 epoch...")
    w2v_short = Word2Vec(
        sentences=X_train_tokens,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        sg=W2V_SG,
        epochs=W2V_EPOCHS_SHORT,
        seed=RANDOM_SEED,
    )

    print("Training Word2Vec model for 10 epochs...")
    w2v_long = Word2Vec(
        sentences=X_train_tokens,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        sg=W2V_SG,
        epochs=W2V_EPOCHS_LONG,
        seed=RANDOM_SEED,
    )

    print("Converting documents to vectors for 1-epoch model...")
    X_train_short = corpus_to_vectors(X_train_tokens, w2v_short, W2V_VECTOR_SIZE)
    X_val_short = corpus_to_vectors(X_val_tokens, w2v_short, W2V_VECTOR_SIZE)
    X_test_short = corpus_to_vectors(X_test_tokens, w2v_short, W2V_VECTOR_SIZE)

    print("Converting documents to vectors for 10-epoch model...")
    X_train_long = corpus_to_vectors(X_train_tokens, w2v_long, W2V_VECTOR_SIZE)
    X_val_long = corpus_to_vectors(X_val_tokens, w2v_long, W2V_VECTOR_SIZE)
    X_test_long = corpus_to_vectors(X_test_tokens, w2v_long, W2V_VECTOR_SIZE)

    print(f"Word2Vec train shape (1 epoch): {X_train_short.shape}")
    print(f"Word2Vec train shape (10 epochs): {X_train_long.shape}")

    print("\nTraining MLP on 1-epoch Word2Vec vectors...")
    result_short = train_mlp(
        X_train_short, y_train,
        X_val_short, y_val,
        X_test_short, y_test,
        len(target_names)
    )

    print("\nTraining MLP on 10-epoch Word2Vec vectors...")
    result_long = train_mlp(
        X_train_long, y_train,
        X_val_long, y_val,
        X_test_long, y_test,
        len(target_names)
    )

    save_training_plots(
        result_short["train_losses"],
        result_short["val_losses"],
        result_short["val_accuracies"],
        result_short["val_macro_f1s"],
        "word2vec_1_epoch"
    )

    save_training_plots(
        result_long["train_losses"],
        result_long["val_losses"],
        result_long["val_accuracies"],
        result_long["val_macro_f1s"],
        "word2vec_10_epochs"
    )

    plot_word_embeddings(w2v_short, w2v_long)

    save_results(
        "results/word2vec_mlp_1_epoch_results.txt",
        "Word2Vec (1 Epoch) + MLP Results",
        result_short,
        target_names
    )

    save_results(
        "results/word2vec_mlp_10_epochs_results.txt",
        "Word2Vec (10 Epochs) + MLP Results",
        result_long,
        target_names
    )

    print("\nDone.")
    print("Saved:")
    print("- results/word2vec_mlp_1_epoch_results.txt")
    print("- results/word2vec_mlp_10_epochs_results.txt")
    print("- figures/word2vec_1_epoch_loss_curve.png")
    print("- figures/word2vec_1_epoch_validation_metrics.png")
    print("- figures/word2vec_10_epochs_loss_curve.png")
    print("- figures/word2vec_10_epochs_validation_metrics.png")
    print("- figures/word2vec_epoch_1_pca.png")
    print("- figures/word2vec_epoch_10_pca.png")


if __name__ == "__main__":
    main()