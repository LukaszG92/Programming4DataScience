# Importazione delle librerie necessarie
import pandas as pd
from typing import Union
import spacy
import torch
import time
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Caricamento del modello spaCy per l'elaborazione del linguaggio naturale
nlp = spacy.load("en_core_web_md")

# Aggiunta di pipeline per statistiche descrittive e leggibilità
nlp.add_pipe("textdescriptives/descriptive_stats")
nlp.add_pipe("textdescriptives/readability")

# Selezione del dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento dei modelli pre-addestrati per sentiment ed emozioni
emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


def get_descriptive_and_readability(text_column: pd.Series) -> pd.DataFrame:
    """
    Calcola statistiche descrittive e metriche di leggibilità per una colonna di testi.

    Args:
        text_column (pd.Series): Colonna di testi da analizzare

    Returns:
        pd.DataFrame: DataFrame con metriche descrittive e di leggibilità
    """
    docs = nlp.pipe(text_column)
    metrics_to_extract = ["descriptive_stats", "readability"]
    metrics_df = extract_df(docs, metrics=metrics_to_extract, include_text=False)

    # Rimozione di alcune metriche specifiche
    labels = ['token_length_mean', 'token_length_median', 'token_length_std',
              'sentence_length_std', 'syllables_per_token_mean',
              'syllables_per_token_median', 'syllables_per_token_std', 'n_characters']
    metrics_df.drop(labels, axis=1, inplace=True)

    return metrics_df


def get_topic(text_column: pd.Series) -> pd.DataFrame:
    """
    Estrae i topic principali utilizzando LDA (Latent Dirichlet Allocation).

    Args:
        text_column (pd.Series): Colonna di testi da analizzare

    Returns:
        pd.DataFrame: DataFrame con topic e parole chiave
    """
    # Preprocessing dei testi
    docs = nlp.pipe(text_column)
    removal = {'ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM'}

    def preprocess_text(doc):
        return [token.lemma_.lower() for token in doc if
                token.pos_ not in removal and not token.is_stop and token.is_alpha]

    tokens = list(map(preprocess_text, docs))
    tokens = [tok if tok else ['unknown'] for tok in tokens]

    # Creazione del dizionario e corpus per LDA
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    # Addestramento del modello LDA
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        iterations=150,
        num_topics=7,
        workers=2,
        passes=10
    )

    # Estrazione delle parole chiave per ogni topic
    topic_words = {idx: [word for word, _ in lda_model.show_topic(idx, topn=5)] for idx in range(7)}

    # Calcolo del topic dominante per ogni documento
    dominant_topics = []
    for doc in tokens:
        bow = dictionary.doc2bow(doc)
        topic_distribution = lda_model[bow]
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0] if topic_distribution else -1
        dominant_topics.append(dominant_topic)

    # Creazione del DataFrame dei risultati
    output_df = pd.DataFrame({
        'topic': dominant_topics,
        'topic_kw_1': [topic_words.get(x, ['No topic'])[0] for x in dominant_topics],
        'topic_kw_2': [topic_words.get(x, ['No topic'])[1] for x in dominant_topics],
        'topic_kw_3': [topic_words.get(x, ['No topic'])[2] for x in dominant_topics]
    })

    return output_df


def get_words_metrics(text: str) -> dict:
    """
    Calcola metriche sulle parole (verbi, sostantivi, aggettivi).

    Args:
        text (str): Testo da analizzare

    Returns:
        dict: Dizionario con conteggi e parole più comuni
    """
    doc = nlp(text)
    verbs, nouns, adjectives = {}, {}, {}

    for token in doc:
        if not token.is_stop:
            token_lemma = token.lemma_.lower()
            if token.pos_ == "VERB":
                verbs[token_lemma] = verbs.get(token_lemma, 0) + 1
            elif token.pos_ == "NOUN":
                nouns[token_lemma] = nouns.get(token_lemma, 0) + 1
            elif token.pos_ == "ADJ":
                adjectives[token_lemma] = adjectives.get(token_lemma, 0) + 1

    return {
        "nouns_count": sum(nouns.values()),
        "verbs_count": sum(verbs.values()),
        "adjectives_count": sum(adjectives.values()),
        "most_common_verb": max(verbs, key=verbs.get) if verbs else None,
        "most_common_noun": max(nouns, key=nouns.get) if nouns else None,
        "most_common_adjective": max(adjectives, key=adjectives.get) if adjectives else None,
    }


def get_sentiment(text: str, window_size=512, overlap=128) -> Union[str, None]:
    """
    Calcola il sentiment del testo utilizzando un modello pre-addestrato.

    Args:
        text (str): Testo da analizzare
        window_size (int): Dimensione della finestra di tokenizzazione
        overlap (int): Sovrapposizione tra le finestre

    Returns:
        str: Sentiment dominante
    """
    # Tokenizzazione e analisi del sentiment con finestre sovrapposte
    encoded = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=window_size,
        stride=overlap,
        return_overflowing_tokens=True
    )

    # Calcolo del sentiment medio su più finestre
    all_probabilities = []
    for i in range(len(encoded["input_ids"])):
        input_ids = encoded["input_ids"][i].to(device)
        attention_mask = encoded["attention_mask"][i].to(device)

        with torch.no_grad():
            outputs = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            all_probabilities.append(probabilities)

    # Voto a maggioranza
    avg_probabilities = torch.mean(torch.stack(all_probabilities), dim=0)
    prediction = torch.argmax(avg_probabilities, dim=-1)

    return sentiment_model.config.id2label[prediction.item()]


def get_emotions(text: str, window_size=512, overlap=128) -> dict:
    """
    Estrae le emozioni principali dal testo.

    Args:
        text (str): Testo da analizzare
        window_size (int): Dimensione della finestra di tokenizzazione
        overlap (int): Sovrapposizione tra le finestre

    Returns:
        dict: Dizionario con le tre emozioni principali
    """
    # Tokenizzazione e analisi delle emozioni con finestre sovrapposte
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
    results = []

    # Calcolo delle emozioni per ogni finestra
    for i in range(input_ids.size(0)):
        window_input_ids = input_ids[i:i + 1]

        with torch.no_grad():
            outputs = emotion_model(window_input_ids)
            results.append(outputs.logits)

    # Calcolo delle emozioni medie
    if results:
        avg_logits = torch.mean(torch.cat(results), dim=0)
        probabilities = torch.nn.functional.softmax(avg_logits, dim=-1).cpu().numpy()

        sorted_indices = probabilities.argsort()[::-1]

        return {
            'emotion_1': emotion_model.config.id2label[sorted_indices[0]],
            'emotion_2': emotion_model.config.id2label[sorted_indices[1]],
            'emotion_3': emotion_model.config.id2label[sorted_indices[2]]
        }

    return {'emotion_1': None, 'emotion_2': None, 'emotion_3': None}


def text_based_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Arricchisce un DataFrame con varie metriche testuali.

    Args:
        input_data (pd.DataFrame): DataFrame contenente una colonna 'text'

    Returns:
        pd.DataFrame: DataFrame arricchito con metriche testuali
    """
    # Calcolo delle metriche con stampa dei tempi di esecuzione
    print("Calcolo delle metriche delle parole...")
    start_time = time.time()
    word_metrics_df = pd.DataFrame(list(input_data['text'].map(get_words_metrics)))
    print(f"Metriche delle parole calcolate in {time.time() - start_time:.2f} secondi.")

    # Calcolo di altre metriche (descrittive, sentiment, emozioni, topic)
    descriptive_df = get_descriptive_and_readability(input_data['text'])
    sentiment_df = input_data['text'].map(get_sentiment).to_frame(name='sentiment')
    emotion_df = pd.DataFrame(list(input_data['text'].map(get_emotions)))
    topics_df = get_topic(input_data['text'])

    # Combinazione di tutti i DataFrame
    output_data = pd.concat([
        word_metrics_df,
        descriptive_df,
        sentiment_df,
        emotion_df,
        topics_df
    ], axis=1)

    # Filtro per testi con più di una frase
    output_data = output_data[output_data['n_sentences'] > 1]

    return output_data