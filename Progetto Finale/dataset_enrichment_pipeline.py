import pandas as pd

from wikidata_author_enrichment import author_enrichment
from llm_text_metadata_enrichment import text_metadata_enrichment
from text_based_feature_enrichment import text_based_enrichment
from speech_based_feature_enrichment import speech_based_feature_enrichment

from narrative_enrichment import narrative_enrichment

df = pd.read_csv('datasets/speech-a.tsv', sep='\t', header=None, names=['author', 'code', 'text'])

author_df = author_enrichment(df)
speech_metadata_df = text_metadata_enrichment(author_df)
text_based_df = text_based_enrichment(speech_metadata_df)
speech_based_df = speech_based_feature_enrichment(text_based_df)

speech_based_df.to_csv('datasets/speech-b.csv', index=False)

"""Ultimi passi della pipeline, attualmente commentati in attesa dei precedenti
narrative_df = narrative_enrichment(propaganda_features_df)
propaganda_span_df = propaganda_span_enrichment(narrative_df)

propaganda_span_df.to_csv('datasets/speech-b.csv', index=False)
"""