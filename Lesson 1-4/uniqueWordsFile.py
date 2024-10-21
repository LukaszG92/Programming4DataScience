"""
Write a Python program that reads a text from a file and performs the following tasks:
        1. Tokenize the text into words (split by whitespace and punctuation).
        2. Create a list of unique words (case-insensitive) found in the text.
        3. Create a dictionary that counts the frequency of each word in the text using a `Counter`.
        4. Create a set of words that appear more than once in the text.
        5. Print the list of unique words, the dictionary of word frequencies, and the set of words that appear more than once.
"""
from collections import Counter

# Apriamo il file di testo e mettiamo il suo contenuto in una stringa.
with open('dummyText.txt', 'r') as f:
    text = f.read()

# Tokenizziamo il testo in una lista di parole
textWords = text.split(" ")

# Creiamo una lista per le parole uniche e popoliamola
uniqueWords = []
for word in textWords:
    # Se la parola non è già nella lista, aggiungiamola
    # Si usa lower
    if word.lower() not in uniqueWords:
        uniqueWords.append(word.lower())

wordFrequencies = Counter(textWords)

nonUniqueWords = []
for (word, frequency) in wordFrequencies.items():
    if frequency > 1:
        nonUniqueWords.append(word)

print(uniqueWords)
print(wordFrequencies)
print(nonUniqueWords)
