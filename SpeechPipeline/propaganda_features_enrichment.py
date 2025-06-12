# Importazione delle librerie necessarie
import pandas as pd  # Gestione dei dataframe
from collections import Counter  # Conteggio degli elementi
from nltk.util import ngrams  # Creazione di n-grammi
from nltk.corpus import stopwords  # Rimozione delle parole vuote
from nltk import word_tokenize  # Tokenizzazione del testo
import string  # Gestione della punteggiatura
import unicodedata  # Gestione dei caratteri Unicode
import nltk  # Libreria di elaborazione del linguaggio naturale

# Download delle risorse NLTK necessarie
nltk.download('stopwords')  # Scarica le stopwords 
nltk.download('punkt_tab')  # Scarica il tokenizzatore
nltk.download('averaged_perceptron_tagger_eng')  # Scarica il parte del discorso tagger


def get_most_common_ngrams(text: str) -> dict:
    """
    Estrae i bigrammi e trigrammi più comuni da un testo.

    Passaggi di elaborazione:
    1. Rimuove le stopwords (parole vuote come articoli, preposizioni)
    2. Normalizza il testo (rimuove accenti, punteggiatura)
    3. Crea bigrammi e trigrammi
    4. Trova i più comuni

    Args:
        text (str): Testo di input da analizzare

    Returns:
        dict: Dizionario con bigramma e trigramma più comuni
    """
    # Crea un set di stopwords in inglese
    stop_words = set(stopwords.words('english'))

    # Crea una tabella per rimuovere la punteggiatura
    punctuation_table = str.maketrans('', '', string.punctuation)

    # Normalizzazione del testo
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.translate(punctuation_table)

    # Tokenizzazione e rimozione stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # Creazione di bigrammi e trigrammi
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    # Trova i bigrammi e trigrammi più comuni
    common_bigram = Counter(bigrams).most_common(1)
    common_trigram = Counter(trigrams).most_common(1)

    # Estrae il primo bigramma e trigramma (se esistono)
    bigram = common_bigram[0][0] if common_bigram else None
    trigram = common_trigram[0][0] if common_trigram else None

    return {'most_common_bigram': bigram, 'most_common_trigram': trigram}


def propaganda_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce un DataFrame con l'analisi degli n-grammi.

    Args:
        input_data (pd.DataFrame): DataFrame contenente una colonna 'text'

    Returns:
        pd.DataFrame: DataFrame originale arricchito con colonne di n-grammi
    """
    # Applica get_most_common_ngrams a ogni testo nel DataFrame
    ngrams_df = pd.DataFrame(list(input_data['text'].map(get_most_common_ngrams)))

    # Concatena i risultati degli n-grammi con il DataFrame originale
    output_df = pd.concat([ngrams_df, input_data], axis=1)

    return output_df