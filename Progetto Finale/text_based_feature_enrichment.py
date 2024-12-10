from typing import Union
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
import spacy
from spacy.tokens import Token
from spacy.language import Doc
from textdescriptives.extractors import extract_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

nlp = spacy.load("en_core_web_md")

nlp.add_pipe("textdescriptives/descriptive_stats")
nlp.add_pipe("textdescriptives/readability")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


def get_descriptive_and_readability(text_column: pd.Series) -> pd.DataFrame:
    docs = nlp.pipe(text_column)

    metrics_to_extract = ["descriptive_stats", "readability"]
    metrics_df = extract_df(docs, metrics=metrics_to_extract, include_text=False)

    labels = ['token_length_mean', 'token_length_median', 'token_length_std', 'sentence_length_std',
              'syllables_per_token_mean', 'syllables_per_token_median', 'syllables_per_token_std', 'n_characters']
    metrics_df.drop(labels, axis=1, inplace=True)

    return metrics_df


def get_topic(text_column: pd.Series) -> pd.DataFrame:
    # Utilizzo del pipeline di spaCy
    docs = nlp.pipe(text_column)

    # Tipi di parole da rimuovere
    removal = {'ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM'}

    # Funzione per preprocessare il testo
    def preprocess_text(doc: Doc) -> list[str]:
        return [token.lemma_.lower() for token in doc if token.pos_ not in removal and not token.is_stop and token.is_alpha]

    # Preprocessing dei documenti
    tokens = list(map(preprocess_text, docs))
    tokens = [tok if tok else ['unknown'] for tok in tokens]  # Righe vuote gestite con ['unknown']

    # Creazione DataFrame iniziale
    output_df = pd.DataFrame({'tokens': tokens})

    # Creazione del dizionario e corpus
    dictionary = Dictionary(output_df['tokens'])
    corpus = [dictionary.doc2bow(doc) for doc in output_df['tokens']]

    # Addestramento modello LDA
    num_topics = 4
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        iterations=150,
        num_topics=num_topics,
        workers=2,
        passes=10
    )

    # Estrazione delle parole chiave per ogni topic
    topic_words = {idx: [word for word, _ in lda_model.show_topic(idx, topn=5)] for idx in range(num_topics)}

    # Calcolo del topic dominante per ogni documento
    dominant_topics = []
    for doc in output_df['tokens']:
        bow = dictionary.doc2bow(doc)
        topic_distribution = lda_model[bow]
        if topic_distribution:
            dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        else:
            dominant_topic = -1  # Valore predefinito per righe senza topic
        dominant_topics.append(dominant_topic)

    # Aggiunta dei topic al DataFrame
    output_df['topic'] = dominant_topics
    output_df['topic_kw_1'] = output_df['topic'].map(lambda x: topic_words.get(x, ['No topic'])[0])
    output_df['topic_kw_2'] = output_df['topic'].map(lambda x: topic_words.get(x, ['No topic'])[1])
    output_df['topic_kw_3'] = output_df['topic'].map(lambda x: topic_words.get(x, ['No topic'])[2])

    # Rimozione della colonna tokens
    output_df.drop('tokens', axis=1, inplace=True)

    return output_df


def dispatcher(token: Token, verbs: dict, nouns: dict, adjectives: dict) -> None:
    if token.is_stop:
        return None

    token_lemma = token.lemma_.lower()
    if token.pos_ == "VERB":
        verbs[token_lemma] = verbs.get(token_lemma, 0) + 1
    if token.pos_ == "NOUN":
        nouns[token_lemma] = nouns.get(token_lemma, 0) + 1
    if token.pos_ == "ADJ":
        adjectives[token_lemma] = adjectives.get(token_lemma, 0) + 1


def get_words_metrics(text: str) -> dict:
    doc = nlp(text)

    verbs = {}
    nouns = {}
    adjectives = {}

    list(map(lambda token: dispatcher(token, verbs, nouns, adjectives), doc))

    most_common_verb = max(verbs, key=verbs.get) if verbs else None
    most_common_noun = max(nouns, key=nouns.get) if nouns else None
    most_common_adjective = max(adjectives, key=adjectives.get) if adjectives else None

    pos_counts = {
        "nouns_count": sum(nouns.values()),
        "verbs_count": sum(verbs.values()),
        "adjectives_count": sum(adjectives.values()),
        "most_common_verb": most_common_verb,
        "most_common_noun": most_common_noun,
        "most_common_adjective": most_common_adjective,
    }

    return pos_counts


def get_sentiment(text: str, window_size=512, overlap=128) -> Union[str, None]:
    encoded = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=window_size,
        stride=overlap,
        return_overflowing_tokens=True
    )

    num_windows = len(encoded["input_ids"])

    all_probabilities = []
    for i in range(num_windows):
        input_ids = encoded["input_ids"][i].to(device)
        attention_mask = encoded["attention_mask"][i].to(device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        if input_ids.size(1) == 0:
            continue

        with torch.no_grad():
            outputs = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            probabilities = torch.softmax(logits, dim=-1)
            all_probabilities.append(probabilities)

    avg_probabilities = torch.mean(torch.stack(all_probabilities), dim=0)

    prediction = torch.argmax(avg_probabilities, dim=-1)
    majority_vote = sentiment_model.config.id2label[prediction.item()]

    return majority_vote


def get_emotions(text: str, window_size=512, overlap=128) -> dict:
    encoded = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=window_size,
        stride=overlap,
        return_overflowing_tokens=True
    )

    input_ids = encoded["input_ids"].to(device)

    num_windows = input_ids.size(0)

    results = []

    for i in range(num_windows):
        window_input_ids = input_ids[i:i + 1]

        with torch.no_grad():
            outputs = emotion_model(window_input_ids)
            logits = outputs.logits
            results.append(logits)

    if results:
        all_logits = torch.cat(results)

        avg_logits = torch.mean(all_logits, dim=0)

        probabilities = torch.nn.functional.softmax(avg_logits, dim=-1).cpu().numpy()
        id2label = emotion_model.config.id2label

        sorted_indices = probabilities.argsort()[::-1]

        return {
            'emotion_1': id2label[sorted_indices[0]],
            'emotion_2': id2label[sorted_indices[1]],
            'emotion_3': id2label[sorted_indices[2]]
        }

    return {
        'emotion_1': None,
        'emotion_2': None,
        'emotion_3': None
    }


def text_based_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    print("Calcolo delle metriche delle parole...")
    start_time = time.time()
    word_metrics_df = pd.DataFrame(list(input_data['text'].map(get_words_metrics)))
    print(f"Metriche delle parole calcolate in {time.time() - start_time:.2f} secondi.")

    print("Calcolo delle metriche descrittive e di leggibilità...")
    start_time = time.time()
    descriptive_df = get_descriptive_and_readability(input_data['text'])
    print(f"Metriche descrittive e di leggibilità calcolate in {time.time() - start_time:.2f} secondi.")

    print("Calcolo del sentiment...")
    start_time = time.time()
    sentiment_model.to(device)
    sentiment_df = input_data['text'].map(get_sentiment)
    sentiment_df = sentiment_df.to_frame(name='sentiment')
    print(f"Sentiment calcolato in {time.time() - start_time:.2f} secondi.")

    print("Calcolo delle emozioni...")
    start_time = time.time()
    emotion_model.to(device)
    emotion_df = pd.DataFrame(list(input_data['text'].map(get_emotions)))
    print(f"Emozioni calcolate in {time.time() - start_time:.2f} secondi.")

    print("Estrazione dei topic...")
    start_time = time.time()
    topics_df = get_topic(input_data['text'])
    print(f"Topic estratti in {time.time() - start_time:.2f} secondi.")

    output_data = pd.concat([word_metrics_df, descriptive_df, sentiment_df, emotion_df, topics_df], axis=1)

    return output_data
