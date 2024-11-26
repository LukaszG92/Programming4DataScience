import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

df = pd.read_csv('datasets/speech-a.tsv', sep='\t', header=None, names=['author', 'code', 'text'])

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader("User-Agent", "ProgrammingForDataScience/1.0 (Contact: lukasz.gajewski15@gmail.com)")

author_mapping = {
    'Obama': 'Barack Obama',
    'Churchill': 'Winston Churchill',
    'Trump': 'Donald Trump',
    'Goebbels': 'Joseph Goebbels'
}

def build_query(name: str) -> str:
    """Builds the SPARQL query for a given name."""
    return f"""
    SELECT ?personLabel ?birthDate ?deathDate ?nationalityLabel ?positionLabel
    WHERE {{
        ?person rdfs:label "{name}"@en.
        ?person wdt:P569 ?birthDate.
        OPTIONAL {{
            ?person wdt:P570 ?deathDate.
        }}
        OPTIONAL {{
            ?person wdt:P27 ?nationality.
        }}  
        OPTIONAL {{
            ?person wdt:P39 ?position.
        }}
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
        }}
    }}
    LIMIT 10
    """

def parse_result(result: dict) -> dict:
    """Parses the result dictionary into a structured format."""
    return {
        'date_of_birth': result.get('birthDate', {}).get('value', None),
        'date_of_death': result.get('deathDate', {}).get('value', None),
        'nationality': result.get('nationalityLabel', {}).get('value', None),
        'position': result.get('positionLabel', {}).get('value', None)
    }

def get_author_info(name: str) -> dict:
    """Fetches information about an author from Wikidata."""
    print(f'Fetching author info for {name}')
    query = build_query(name)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if results['results']['bindings']:
            result = results['results']['bindings'][0]
            return parse_result(result)
    except Exception as e:
        print(f"Error retrieving data for author {name}: {e}")
    return {
        'date_of_birth': None,
        'date_of_death': None,
        'nationality': None,
        'position': None
    }

# Map the author names to full names
df['author'] = df['author'].map(author_mapping)

# Fetch author information and expand into separate columns
author_info = df['author'].map(get_author_info)
author_info_df = pd.DataFrame(author_info.tolist())

# Merge the new information into the original DataFrame
df = pd.concat([df, author_info_df], axis=1)

df.to_csv('datasets/wikidata_author_enrichment.csv', index=False)
