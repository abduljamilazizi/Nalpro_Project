import re
import string


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercase
    - remove punctuation
    - remove digits
    - remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    """
    Clean text and split into tokens.
    """
    cleaned = clean_text(text)
    return cleaned.split()


def preprocess_corpus(texts: list[str]) -> list[str]:
    """
    Apply clean_text to a list of documents.
    """
    return [clean_text(text) for text in texts]


def tokenize_corpus(texts: list[str]) -> list[list[str]]:
    """
    Apply tokenize_text to a list of documents.
    """
    return [tokenize_text(text) for text in texts]


def main():
    sample_text = """
    Hello! This is a Sample text, from 2026.
    It has punctuation, numbers like 123, and   extra spaces.
    """

    print("Original text:")
    print(sample_text)

    print("\nCleaned text:")
    print(clean_text(sample_text))

    print("\nTokens:")
    print(tokenize_text(sample_text))


if __name__ == "__main__":
    main()