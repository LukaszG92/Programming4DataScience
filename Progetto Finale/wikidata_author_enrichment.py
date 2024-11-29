import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader("User-Agent", "ProgrammingForDataScience/1.0 (Contact: lukasz.gajewski15@gmail.com)")

author_mapping = {
    'Obama': 'Barack Obama',
    'Churchill': 'Winston Churchill',
    'Trump': 'Donald Trump',
    'Goebbels': 'Joseph Goebbels'
}

def build_query(name: str) -> str:
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
    return {
        'date_of_birth': result.get('birthDate', {}).get('value', None),
        'date_of_death': result.get('deathDate', {}).get('value', None),
        'nationality': result.get('nationalityLabel', {}).get('value', None),
        'position': result.get('positionLabel', {}).get('value', None)
    }

def get_author_info(name: str) -> dict:
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

def author_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    input_data['author'] = input_data['author'].map(author_mapping)

    author_info = input_data['author'].map(get_author_info)
    author_info_df = pd.DataFrame(author_info.tolist())

    output_data = pd.concat([input_data, author_info_df], axis=1)

    return output_data

