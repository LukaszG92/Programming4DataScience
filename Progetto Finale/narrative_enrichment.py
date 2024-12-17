# Importazione delle librerie necessarie
import os  # Per gestire variabili d'ambiente
from groq import Groq  # Libreria client per l'API Groq
import pandas as pd  # Libreria per la manipolazione dei dati

# Configurazione dell'API Groq utilizzando una chiave di accesso dall'ambiente
api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)


def get_narrative(text: str) -> dict:
    """
    Analizza un testo e determina il suo archetipo narrativo utilizzando un modello di intelligenza artificiale.

    Args:
        text (str): Il testo da analizzare

    Returns:
        dict: Un dizionario contenente l'archetipo narrativo
    """
    # Selezione del modello in base alla lunghezza del testo
    # Per testi più corti usa un modello più grande e versatile
    # Per testi più lunghi usa un modello più piccolo e veloce
    if len(text.split()) <= 3800:
        model = 'llama-3.3-70b-versatile'
    else:
        model = 'llama-3.1-8b-instant'

    # Costruzione di un prompt dettagliato per l'analisi narrativa
    prompt = f"""
    Analizza il seguente testo e fornisci il suo archetipo narrativo.

    Restituisci il risultato in formato JSON con la seguente struttura:
    {{
         'narrative archetype': 'valore'
    }}

    - Assicurati che la risposta sia strettamente in formato JSON, senza testo aggiuntivo
    - L'archetipo narrativo deve essere uno di questi:
    {{
        "Overcoming The Monster": "Il protagonista combatte una forza mostruosa che minaccia la sopravvivenza...",
        "Voyage And Return": "Il protagonista lascia casa, incontra un mondo nuovo e sfidante...",
        "Rags To Riches": "Il protagonista si solleva da un punto basso...",
        "The Quest": "Il protagonista parte per trovare un oggetto o una persona...",
        "Comedy": "Una serie di equivoci creano conflitto...",
        "Tragedy": "Il difetto del protagonista porta alla sua rovina...",
        "Rebirth": "Il protagonista compie un viaggio di redenzione..."
    }}

    Testo:  {text}
    """

    # Chiamata all'API Groq per ottenere l'analisi
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    # Estrazione e pulizia della risposta
    response = chat_completion.choices[0].message.content.strip()
    response = response[response.find("{"):response.rfind("}") + 1]

    # Conversione della risposta in un dizionario
    try:
        result_dict = eval(response)
        values = list(result_dict.values())
        result = {
            'narrative': values[0],
        }
    except Exception as e:
        # Gestione degli errori se la conversione fallisce
        result = {
            "narrative": None
        }

    return result


def narrative_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce un DataFrame aggiungendo una colonna con l'archetipo narrativo.

    Args:
        input_data (pd.DataFrame): DataFrame di input con una colonna 'text'

    Returns:
        pd.DataFrame: DataFrame originale con una nuova colonna 'narrative'
    """
    # Applica get_narrative a ogni testo nel DataFrame
    text_narrative_df = pd.DataFrame(list(input_data['text'].map(get_narrative)))

    # Concatena il DataFrame originale con i risultati dell'analisi narrativa
    output_data = pd.concat([input_data, text_narrative_df], axis=1)

    return output_data