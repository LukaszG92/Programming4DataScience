import os
import time
from groq import Groq
import yake
import pandas as pd
from transformers import AutoTokenizer

# Configurazione delle API e degli strumenti
api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
yake_extractor = yake.KeywordExtractor(lan="en", n=2, top=3)


def summarize_text(text: str) -> dict:
    """
    Genera un riassunto conciso del testo utilizzando l'API di Groq.

    Args:
        text (str): Testo da riassumere

    Returns:
        dict: Dizionario contenente l'abstract
    """
    try:
        # Selezione del modello in base alla lunghezza del testo
        model = 'llama-3.3-70b-versatile' if len(text.split()) <= 5500 else 'llama-3.1-8b-instant'

        # Prompt per la generazione del riassunto
        prompt = f"""
        Analizza il discorso fornito.
        Restituisci il risultato in formato JSON con la seguente struttura:
        {{
            "abstract": "valore",
        }}
        - Il campo "abstract" deve essere una stringa concisa (1-2 frasi) 
          che evidenzi gli argomenti principali e i punti chiave del testo.
        Discorso: {text}
        """

        # Chiamata all'API di Groq
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )

        # Parsing della risposta
        response = chat_completion.choices[0].message.content.strip()
        response = response[response.find("{"):response.find("}") + 1]

        result = eval(response)
        values = list(result.values())
        return {
            "abstract": values[0],
        }
    except Exception as e:
        return {"abstract": None}


def get_keywords(text: str, window_size=4096, stride=3584) -> dict:
    """
    Estrae le parole chiave da un testo utilizzando YAKE.

    Args:
        text (str): Testo da analizzare
        window_size (int): Dimensione della finestra di tokenizzazione
        stride (int): Sovrapposizione tra le finestre

    Returns:
        dict: Dizionario contenente le top 3 parole chiave
    """
    try:
        # Tokenizzazione del testo in finestre
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=window_size,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

        # Estrazione delle parole chiave per ogni finestra
        windows = encoded["input_ids"]
        all_keywords = []
        for window_tensor in windows:
            window = tokenizer.decode(window_tensor, skip_special_tokens=True)
            keywords = yake_extractor.extract_keywords(window)
            all_keywords.extend(keywords)

        # Calcolo del miglior punteggio per ogni parola chiave
        keyword_scores = {}
        for word, score in all_keywords:
            keyword_scores[word] = min(score, keyword_scores.get(word, float('inf')))

        # Ordinamento e restituzione delle top 3 parole chiave
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1])
        return {
            "keyword_1": sorted_keywords[0][0] if len(sorted_keywords) > 0 else None,
            "keyword_2": sorted_keywords[1][0] if len(sorted_keywords) > 1 else None,
            "keyword_3": sorted_keywords[2][0] if len(sorted_keywords) > 2 else None,
        }
    except Exception:
        return {"keyword_1": None, "keyword_2": None, "keyword_3": None}


def speech_based_feature_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce un DataFrame con riassunti e parole chiave.

    Args:
        input_data (pd.DataFrame): DataFrame contenente una colonna 'text'

    Returns:
        pd.DataFrame: DataFrame arricchito con riassunti e parole chiave
    """
    # Generazione dei riassunti
    print("Generazione dei riassunti...")
    start_time = time.time()
    summarized_speech_df = pd.DataFrame(list(input_data['text'].map(summarize_text)))
    print(f"Riassunti generati in {time.time() - start_time:.2f} secondi.\n")

    # Estrazione delle parole chiave
    print("Estrazione delle parole chiave...")
    start_time = time.time()
    keywords_df = pd.DataFrame(list(input_data['text'].map(get_keywords)))
    print(f"Parole chiave estratte in {time.time() - start_time:.2f} secondi.\n")

    # Combinazione dei DataFrame
    output_data = pd.concat([input_data, summarized_speech_df, keywords_df], axis=1)

    return output_data