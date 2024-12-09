from typing import Union
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
import spacy
from spacy.tokens import Token
from textdescriptives.extractors import extract_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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
    docs = nlp.pipe(text_column)
    removal = {'ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM'}

    def preprocess_text(text: str) -> list[str]:
        doc = nlp(text)
        return [token.lemma_.lower() for token in doc if token.pos_ not in removal and not token.is_stop and token.is_alpha]

    df = pd.DataFrame()
    df['tokens'] = text_column.map(preprocess_text)
    df['tokens'] = df['tokens'].map(lambda x: x if x is not None and len(x) > 0 else [])

    dictionary = Dictionary(df['tokens'])

    corpus = [dictionary.doc2bow(doc) for doc in df['tokens']]

    num_topics = 4
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=150, num_topics=num_topics, workers = 2, passes=10)
    topic_words = {idx: [word for word, _ in lda_model.show_topic(idx, topn=5)] for idx in range(num_topics)}

    dominant_topics = []
    for doc in df['tokens']:
        bow = dictionary.doc2bow(doc)
        topic_distribution = lda_model[bow]
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        dominant_topics.append(dominant_topic)


    output_df = pd.DataFrame()
    output_df['topic'] = dominant_topics
    output_df['topic_kw_1', 'topic_kw_2', 'topic_kw_3'] = output_df['topic'].map(lambda x : (topic_words[x][0], topic_words[x][1], topic_words[x][2]))

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


def get_sentiment(text: str, window_size=512, overlap=384, batch_size=32) -> Union[str, None]:
    encoded = sentiment_tokenizer(
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
            outputs = sentiment_model(batch_input_ids)
            logits = outputs.logits
            results.append(logits)

    if results:
        all_logits = torch.cat(results)
        final_logits = torch.mean(all_logits, dim=0)
        id2label = sentiment_model.config.id2label
        final_prediction = torch.argmax(final_logits, dim=-1)
        return id2label[final_prediction.item()]

    return None


def get_emotions(text: str, window_size=512, overlap=384, batch_size=32) -> dict:
    encoded = emotion_tokenizer(
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
            outputs = emotion_model(batch_input_ids)
            logits = outputs.logits
            results.append(logits)

    if results:
        all_logits = torch.cat(results)
        final_logits = torch.mean(all_logits, dim=0)

        probabilities = torch.nn.functional.softmax(final_logits, dim=-1).cpu().numpy()
        id2label = emotion_model.config.id2label

        sorted_indices = probabilities.argsort()[::-1]
        return {
            'emotion_1' : id2label[sorted_indices[0]],
            'emotion_2' : id2label[sorted_indices[1]],
            'emotion_3' : id2label[sorted_indices[2]]
        }

    return {
        'emotion_1' : None,
        'emotion_2' : None,
        'emotion_3' : None
    }


def text_based_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    word_metrics = input_data['text'].map(get_words_metrics)
    word_metrics_and_topics_df = pd.DataFrame(word_metrics.tolist())

    descriptive_df = get_descriptive_and_readability(input_data["text"])

    sentiment_model.to(device)
    sentiment = input_data['text'].map(get_sentiment)
    sentiment_df = sentiment.to_frame(name='predicted_sentiment')

    emotion_model.to(device)
    emotion = input_data['text'].map(get_emotions)
    emotion_df = pd.DataFrame(emotion.tolist())

    topics_df = get_topic(input_data["text"])

    output_data = pd.concat([input_data, word_metrics_and_topics_df, descriptive_df, sentiment_df, emotion_df, topics_df], axis=1)

    return output_data