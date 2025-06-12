import time
import pandas as pd

# Importazione delle funzioni di arricchimento da moduli specifici
from wikidata_author_enrichment import author_enrichment
from llm_text_metadata_enrichment import text_metadata_enrichment
from text_based_feature_enrichment import text_based_enrichment
from speech_based_feature_enrichment import speech_based_feature_enrichment
from propaganda_features_enrichment import propaganda_enrichment
from narrative_enrichment import narrative_enrichment
from propaganda_span_enrichment import propaganda_span_enrichment

# Caricamento del dataset originale
df = pd.read_csv('datasets/speech-a.tsv', sep='\t', header=None, names=['author', 'code', 'text'])
df = df.reset_index()

print("Iniziando l'arricchimento con i dati dell'autore...")
start_time = time.time()
author_df = author_enrichment(df)
print(f"Dataset arricchito con i dati dell'autore in {time.time() - start_time:.2f} secondi.\n")

print("Iniziando l'arricchimento con i metadati del discorso...")
start_time = time.time()
speech_metadata_df = text_metadata_enrichment(author_df)
print(f"Dataset arricchito con i metadati del discorso in {time.time() - start_time:.2f} secondi.\n")

print("Iniziando l'arricchimento con i dati basati sul testo...")
start_time = time.time()
text_based_df = text_based_enrichment(speech_metadata_df)
print(f"Dataset arricchito con i dati basati sul testo in {time.time() - start_time:.2f} secondi.\n")

print("Iniziando l'arricchimento con i dati basati sul discorso...")
start_time = time.time()
speech_based_df = speech_based_feature_enrichment(text_based_df)
print(f"Dataset arricchito con i dati basati sul discorso in {time.time() - start_time:.2f} secondi.\n")

print("Iniziando l'arricchimento con i dati utili alla propaganda detection...")
start_time = time.time()
propaganda_df = propaganda_enrichment(speech_based_df)
print(f"Dataset arricchito con i dati utili alla propaganda detection in {time.time() - start_time:.2f} secondi.\n")

print("Iniziando l'arricchimento con i dati riguardo la narrativa...")
start_time = time.time()
narrative_df = narrative_enrichment(propaganda_df)
print(f"Dataset arricchito con i dati riguardo la narrativa in {time.time() - start_time:.2f} secondi.\n")

print("Iniziando l'arricchimento con gli span di propaganda..")
start_time = time.time()
propaganda_span_df = propaganda_span_enrichment(narrative_df)
print(f"Dataset arricchito con gli span di propaganda in {time.time() - start_time:.2f} secondi.\n")

propaganda_span_df.to_csv('datasets/speech-b.csv')