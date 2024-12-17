import time
import pandas as pd
import os
from groq import Groq

# Recupero della chiave API da variabile d'ambiente
api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)


def get_propaganda_spans(text: str):
    """
    Identifica gli span di propaganda all'interno di un testo.

    Args:
        text (str): Testo da analizzare

    Returns:
        list: Metadati degli span di propaganda trovati
    """
    span_metadata = []

    # Suddivisione del testo in chunk per gestire testi lunghi
    for i in range(0, len(text), 5700):
        # Prompt dettagliato per l'identificazione degli span di propaganda
        prompt = f"""
        Ricerca nel testo le 2-3 fette di propaganda più rilevanti, con il relativo tipo di propaganda.
        Il formato del risultato deve essere un JSON strutturato come segue, senza altre aggiunte::
        {{
            "testo di propaganda": "tipo di propaganda",
            "testo di propaganda": "tipo di propaganda",
            "testo di propaganda": "tipo di propaganda"
        }}

        - Il "testo di propaganda" deve essere il frammento di propaganda trovato nel dataset, se non trovato restituire None
        - Il "tipo di propaganda" è la classificazione del testo di propaganda, se non trovato restituire None
        - Il "testo di propaganda" deve essere una frase, che inizia con una lettera maiuscola e termina con un punto
        - Il tipo di propaganda deve essere uno di questi:
        [Lista dettagliata di tecniche di propaganda]

        Testo: {text[i:i + 5700]}
        """

        # Chiamata all'API di Groq per analizzare il testo
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        # Estrazione e parsing della risposta
        response = chat_completion.choices[0].message.content.strip()
        response = response[response.find("{"):response.rfind("}") + 1]
        try:
            # Conversione della risposta in un dizionario
            result = eval(response)
        except:
            # Gestione di eventuali errori di parsing
            result = {}

        # Elaborazione dei risultati
        for key, value in result.items():
            # Calcolo degli offset del testo di propaganda
            start_offset = i + text[i:i + 5700].find(key)
            if start_offset == -1:
                continue
            end_offset = start_offset + len(key)
            propaganda_type = value

            # Memorizzazione dei metadati dello span
            span_metadata.append({
                'start_offset': start_offset,
                'end_offset': end_offset,
                'propaganda_type': propaganda_type
            })

    return span_metadata


def propaganda_span_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce un DataFrame con gli span di propaganda.

    Args:
        input_data (pd.DataFrame): DataFrame contenente una colonna 'text'

    Returns:
        pd.DataFrame: DataFrame arricchito con span di propaganda
    """
    print("Individuazione degli span di propaganda...")
    start_time = time.time()

    # Applicazione della funzione di identificazione degli span
    input_data['span_metadata'] = input_data['text'].apply(get_propaganda_spans)

    # Esplodere il DataFrame per creare righe separate per ogni span
    exploded_df = input_data.explode('span_metadata', ignore_index=True)

    # Normalizzazione dei metadati degli span
    span_metadata_df = pd.json_normalize(exploded_df['span_metadata'])

    # Concatenazione dei DataFrame
    final_df = pd.concat([exploded_df.drop(columns=['span_metadata']), span_metadata_df], axis=1)

    # Aggiunta di un indice per gli span multipli
    final_df['span_index'] = final_df.groupby('index').cumcount()

    # Impostazione dell'indice multiplo
    final_df.set_index(['index', 'span_index'], inplace=True)

    print(f"Propaganda span individuati e dataset ristrutturato in {time.time() - start_time:.2f} secondi.\n")

    return final_df