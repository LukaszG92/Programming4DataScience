import pandas as pd
import time
from SPARQLWrapper import SPARQLWrapper, JSON

# Inizializza l'endpoint SPARQL di WikiData
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
# Aggiunge un header personalizzato per identificare l'utente (buona pratica con WikiData)
sparql.addCustomHttpHeader("User-Agent", "ProgrammingForDataScience/1.0 (Contact: lukasz.gajewski15@gmail.com)")

# Mappatura manuale per normalizzare i nomi degli autori
# Permette di tradurre abbreviazioni o varianti nei nomi completi
author_mapping = {
    'Obama': 'Barack Obama',
    'Churchill': 'Winston Churchill',
    'Trump': 'Donald Trump',
    'Goebbels': 'Joseph Goebbels'
}

def build_query(name: str) -> str:
    """
    Costruisce una query SPARQL per recuperare informazioni su un autore.

    Args:
        name (str): Nome dell'autore.

    Returns:
        str: Una query SPARQL formattata.
    """
    return f"""
    SELECT ?personLabel ?birthDate ?deathDate ?nationalityLabel ?positionLabel
    WHERE {{
        ?person rdfs:label "{name}"@en.  # Cerca entità con il nome specificato
        ?person wdt:P569 ?birthDate.      # Data di nascita obbligatoria
        OPTIONAL {{
            ?person wdt:P570 ?deathDate. # Data di morte opzionale
        }}
        OPTIONAL {{
            ?person wdt:P27 ?nationality. # Nazionalità opzionale
        }}  
        OPTIONAL {{
            ?person wdt:P39 ?position.   # Posizione occupata opzionale
        }}
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". # Etichetta in inglese
        }}
    }}
    LIMIT 10  # Limita il numero di risultati
    """

def parse_result(result: dict) -> dict:
    """
    Estrae le informazioni rilevanti dal risultato SPARQL.

    Args:
        result (dict): Un singolo risultato SPARQL.

    Returns:
        dict: Un dizionario con data di nascita, morte, nazionalità e posizione.
    """
    return {
        'date_of_birth': result.get('birthDate', {}).get('value', None),
        'date_of_death': result.get('deathDate', {}).get('value', None),
        'nationality': result.get('nationalityLabel', {}).get('value', None),
        'position': result.get('positionLabel', {}).get('value', None)
    }

def get_author_info(name: str) -> dict:
    """
    Recupera informazioni da WikiData per un autore specifico.

    Args:
        name (str): Nome dell'autore.

    Returns:
        dict: Informazioni sull'autore (data di nascita, morte, nazionalità, posizione).
    """
    query = build_query(name)  # Costruisce la query
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)  # Richiede i risultati in formato JSON
    try:
        results = sparql.query().convert()  # Esegue la query e converte i risultati
        if results['results']['bindings']:
            result = results['results']['bindings'][0]  # Prende il primo risultato (se presente)
            return parse_result(result)
    except Exception as e:
        # Log degli errori, restituisce valori predefiniti in caso di problemi
        print(f"Errore durante il recupero delle informazioni per {name}: {e}")
        return {
            'date_of_birth': None,
            'date_of_death': None,
            'nationality': None,
            'position': None
        }

def author_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce un DataFrame con informazioni aggiuntive sugli autori recuperate da WikiData.

    Args:
        input_data (pd.DataFrame): DataFrame con una colonna 'author' contenente i nomi degli autori.

    Returns:
        pd.DataFrame: DataFrame arricchito con informazioni aggiuntive sugli autori.
    """
    # Mappatura dei nomi abbreviati o parziali agli equivalenti completi
    print("Mapping del nome completo degli autori...")
    start_time = time.time()
    input_data['author'] = input_data['author'].map(author_mapping)  # Applica la mappatura
    print(f"Nomi completi mappati in {time.time() - start_time:.2f} secondi.\n")

    # Recupero delle informazioni sugli autori tramite WikiData
    print("Query WikiData per le informazioni sugli autori...")
    start_time = time.time()
    author_info_df = pd.DataFrame(list(input_data['author'].map(get_author_info)))  # Crea un DataFrame dai risultati
    print(f"Query completate in {time.time() - start_time:.2f} secondi.\n")

    # Combina i dati originali con le informazioni arricchite
    output_data = pd.concat([input_data, author_info_df], axis=1)

    return output_data