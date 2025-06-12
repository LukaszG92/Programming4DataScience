# Importazione delle librerie necessarie
import os  # Per gestire variabili d'ambiente
from groq import Groq  # Libreria client per l'API Groq
import pandas as pd  # Libreria per la manipolazione dei dati

# Configurazione dell'API Groq utilizzando una chiave di accesso dall'ambiente
api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)


def analyze_text(text: str) -> dict:
    """
    Analizza i metadati relativi a un discorso utilizzando un modello di intelligenza artificiale.

    Args:
        text (str): Il testo del discorso da analizzare

    Returns:
        dict: Un dizionario contenente i metadati estratti (data, luogo, evento)
    """
    # Costruzione di un prompt dettagliato per l'estrazione dei metadati
    prompt = f"""
    Analizza i metadati relativi al discorso fornito.
    Restituisci il risultato in formato JSON con la seguente struttura:
    {{
        "data del discorso": "valore",
        "luogo del discorso": "valore", 
        "evento del discorso": "valore"
    }}

    - La "data del discorso" deve essere standardizzata nel formato "YYYY-MM-DD"
    - Se data, luogo o evento non sono disponibili, restituire "None"
    - Assicurarsi di restituire esclusivamente il formato JSON richiesto

    Discorso: {text[:5900]}
    """

    # Chiamata all'API Groq per ottenere l'analisi
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    # Estrazione e pulizia della risposta
    response = chat_completion.choices[0].message.content.strip()
    response = response[response.find("{"):response.rfind("}") + 1]

    try:
        # Conversione della risposta in un dizionario
        result = eval(response)
        values = list(result.values())
        result = {
            "date of the speech": values[0],
            "location of the speech": values[1],
            "event of the speech": values[2]
        }

    except Exception as e:
        # Gestione degli errori se la conversione fallisce
        result = {
            "date of the speech": None,
            "location of the speech": None,
            "event of the speech": None,
        }
    return result


def text_metadata_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce un DataFrame aggiungendo colonne con i metadati estratti.

    Args:
        input_data (pd.DataFrame): DataFrame di input con una colonna 'text'

    Returns:
        pd.DataFrame: DataFrame originale con nuove colonne per i metadati
    """
    # Applica analyze_text a ogni testo nel DataFrame
    text_metadata_df = pd.DataFrame(list(input_data['text'].map(analyze_text)))

    # Concatena il DataFrame originale con i risultati dell'analisi dei metadati
    output_data = pd.concat([input_data, text_metadata_df], axis=1)

    return output_data