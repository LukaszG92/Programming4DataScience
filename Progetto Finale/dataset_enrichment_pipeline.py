import pandas as pd

from wikidata_author_enrichment import author_enrichment
from llm_text_metadata_enrichment import text_metadata_enrichment

df = pd.read_csv('datasets/speech-a.tsv', sep='\t', header=None, names=['author', 'code', 'text'])

author_df = author_enrichment(df)
speech_metadata_df = text_metadata_enrichment(author_df)

speech_metadata_df.to_csv('datasets/speech-b.csv', index=False)