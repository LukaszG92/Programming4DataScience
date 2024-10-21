"""
In natural language processing, a common preprocessing step is removing stopwords from text data.
Stopwords are words that are considered to be of little value in text analysis because they are very common and don't carry much meaning (e.g., "the," "and," "in").
In this exercise, you will write a Python program to implement a simple stopwords list and use it to filter out stopwords from a given text.
        1. Create a list of stopwords containing common words (e.g., "the," "and," "in").
           You can choose to include more stopwords if you like.
        2. Write a Python program that takes a sentence or a paragraph of text as input from the user.
        3. Tokenize the input text into words (split by whitespace).
        4. Remove any stopwords from the list of words.
        5. Print the filtered list of words, which should exclude the stopwords.
"""

# Si definiscono le stop words
stopWords = ["the", "in", "and"]

# Si prende la stringa in input
inputString = input("Enter a sentence or paragraph: ")

# Si divide la stringa per ogni occorrenza del carattere " ", ovvero ad ogni parola
inputStringWords = inputString.split(" ")

# Si itera su tutte le parole della lista
for word in inputStringWords:
    #Se la parola è una stopword la rimuoviamo
    if word in stopWords:
        inputStringWords.remove(word)

# Si stampa la lista "unpaccata", in modo da ottenere una stringa e non una lista
print(*inputStringWords)