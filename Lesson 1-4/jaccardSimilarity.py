"""
You can calculate the Jaccard similarity between two sets of words, which is a measure of similarity between two sets by finding the size
of their intersection divided by the size of their union.
The code should read the two text documents, preprocess them by tokenizing, removing punctuation, and converting to sets.
Then, it should calculate the Jaccard similarity between the two sets. Jaccard similarity provides a measure of similarity based on the
intersection and union of the sets of words in the two documents.
                J(A,B) = |A intersecato B| / |A unito B|
"""
import string
import re
from typing import Set, Tuple


def preprocess_text(text: str) -> str:
    # Rimozione punteggiatura
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Rimozione spazi
    text = " ".join(text.split())

    return text


def get_word_set(text: str) -> Set[str]:

    processed_text = preprocess_text(text)
    return set(processed_text.split())


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    set1 = get_word_set(text1)
    set2 = get_word_set(text2)

    # Calcolo intersezione e unione
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Controllo sulla divisione per 0
    if union == 0:
        return 0.0

    return intersection / union


# Testi d'esempio
text1 = """The quick brown fox jumps over the lazy dog. 
           The fox is quick and brown."""
text2 = """The quick brown fox jumps over the dog. 
           The fox is fast and brown!"""

# Calcolo della similarità
similarity = calculate_jaccard_similarity(text1, text2)
print(f"Jaccard Similarity: {similarity:.3f}")
