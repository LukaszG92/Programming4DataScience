import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
import textdescriptives as td
from textdescriptives.extractors import extract_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

nlp = spacy.load("en_core_web_md")

nlp.add_pipe("textdescriptives/descriptive_stats")
nlp.add_pipe("textdescriptives/readability")

def get_descriptive_and_readability(input_df: pd.DataFrame) -> pd.DataFrame:
    docs = nlp.pipe(input_df["text"])

    metrics_to_extract = ["descriptive_stats", "readability"]
    metrics_df = extract_df(docs, metrics=metrics_to_extract, include_text=False)

    labels = ['token_length_mean', 'token_length_median', 'token_length_std', 'sentence_length_std', 'syllables_per_token_mean', 'syllables_per_token_median', 'syllables_per_token_std', 'n_characters']
    metrics_df.drop(labels, axis=1, inplace=True)

    return metrics_df


def dispatcher(token, verbs, nouns, adjectives):
    token_lemma = token.lemma_.lower()
    if token.pos_ == "VERB" and not token.is_stop:
        verbs.append(token_lemma)
    elif token.pos_ == "NOUN" and not token.is_stop:
        nouns.append(token_lemma)
    elif token.pos_ == "ADJ" and not token.is_stop:
        adjectives.append(token_lemma)


def calculate_words_metrics(text: str):
    print(f"Analyzing {text}")
    doc = nlp(text)

    verbs = []
    nouns = []
    adjectives = []

    list(map(lambda token: dispatcher(token, verbs, nouns, adjectives), doc))

    most_common_verb = max(set(verbs), key=verbs.count) if verbs else None
    most_common_noun = max(set(nouns), key=nouns.count) if nouns else None
    most_common_adjective = max(set(adjectives), key=adjectives.count) if adjectives else None

    pos_counts = {
        "nouns_count": len(nouns),
        "verbs_count": len(verbs),
        "adjectives_count": len(adjectives),
        "most_common_verb": most_common_verb,
        "most_common_noun": most_common_noun,
        "most_common_adjective": most_common_adjective,
    }

    return pos_counts

def text_based_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    pos_counts = input_data['text'].map(calculate_words_metrics)
    pos_counts_df = pd.DataFrame(pos_counts.tolist())

    descriptive_df = get_descriptive_and_readability(input_data)

    output_data = pd.concat([input_data, pos_counts_df, descriptive_df], axis=1)

    return output_data

df = pd.read_csv('datasets/speech-a.tsv', sep='\t', header=None, names=['author', 'code', 'text'])
df = df[:10]
result_df = text_based_enrichment(df)

result_df.to_csv('datasets/speech-text_based.csv', index=False)


####### Estrazione del sentiment


def predict_sentiment(text, window_size=512, overlap=384, batch_size=32):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,  
        padding="max_length",
        max_length=window_size,
        stride=overlap
    )
    input_ids = encoded["input_ids"].to(device)

    results = []
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]

        with torch.no_grad(): 
            outputs = model(batch_input_ids)
            logits = outputs.logits
            results.append(logits)

    # Aggregazione dei risultati: media dei logit
    if results:
        all_logits = torch.cat(results)
        final_logits = torch.mean(all_logits, dim=0)
        id2label = model.config.id2label
        final_prediction = torch.argmax(final_logits, dim=-1)
        prediction_label = id2label[final_prediction.item()]
    else:
        prediction_label = "Nessuna predizione"

    return prediction_label

def apply_sentiment_analysis(df, column_name):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df['sentiment_prediction'] = df["text"].apply(predict_sentiment)
    return df


####### Estrazione delle emozioni

def predict_emotions(text, window_size=512, overlap=384, batch_size=32):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,  # Troncamento esplicito
        padding="max_length",  # Padding fino alla lunghezza massima
        max_length=window_size,
        stride=overlap
    )
    input_ids = encoded["input_ids"].to(device)

    # Suddivisione in batch
    results = []
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]

        with torch.no_grad():  
            outputs = model(batch_input_ids)
            logits = outputs.logits
            results.append(logits)

    if results:
        all_logits = torch.cat(results)
        final_logits = torch.mean(all_logits, dim=0)

  
        probabilities = torch.nn.functional.softmax(final_logits, dim=-1).cpu().numpy()
        id2label = model.config.id2label  

   
        sorted_indices = probabilities.argsort()[::-1] 
        top_3 = [(id2label[idx], probabilities[idx]) for idx in sorted_indices[:3]]  
    else:
        top_3 = []

    return top_3

def apply_emotion_extraction(df, column_name):
    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df['emotion_prediction'] = df[column_name].apply(predict_emotions)



