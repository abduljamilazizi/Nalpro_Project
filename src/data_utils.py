from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, VALIDATION_SIZE


def load_20newsgroups_data():
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
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_full
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, target_names


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, target_names = load_20newsgroups_data()

    print("Data loaded successfully.")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {len(target_names)}")

    print("\nFirst training sample:")
    print(X_train[0][:1000])

    print("\nFirst training label:")
    print(y_train[0], "-", target_names[y_train[0]])


if __name__ == "__main__":
    main()