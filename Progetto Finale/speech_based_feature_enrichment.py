from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from collections import defaultdict
import os
from groq import Groq
import pandas as pd
import spacy

api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)

def summarize_text(text: str) -> dict:
    print(f"Analyzing text: {text}")
    prompt = f"""
    Analyze the provided speech. 
    Return the result in JSON format with the following structure:
    {{
        "summary": "value",
    }}

    - The "summary field should be a concise string (1-2 sentences) that highlights the primary topics and key points discussed in the text.  

    Speech: {text}
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-8b-instant",
    )

    response = chat_completion.choices[0].message.content.strip()
    response = response[response.find("{"):response.find("}") + 1]
    print(response)
    try:
        result = eval(response)
    except Exception as e:
        result = {
            "summary": None,
            "error": str(e)
        }
    print(result)
    return result


def speech_based_feature_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    summarized_speech = input_data['text'].map(summarize_text)
    summarized_speech_df = pd.DataFrame(summarized_speech.tolist())

    output_data = pd.concat([input_data, summarized_speech_df], axis=1)

    return output_data


def extract_topic_modeling(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_md")
    removal = {'ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM'}

    def preprocess_text(text: str) -> list[str]:
        doc = nlp(text)
        return [token.lemma_.lower() for token in doc if token.pos_ not in removal and not token.is_stop and token.is_alpha]

    df['tokens'] = df[text_column].apply(preprocess_text)
    df = df[df['tokens'].map(len) > 0]

    dictionary = Dictionary(df['tokens'])
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)

    corpus = [dictionary.doc2bow(doc) for doc in df['tokens']]
    topic_ids = [
        "Reform", "Leadership", "Security", "Economy", "Energy", "Healthcare",
        "Education", "Justice", "Diplomacy", "Taxation", "Infrastructure",
        "Technology", "Poverty", "Conflict", "Rights", "Democracy", "Crisis",
        "Innovation", "Freedom", "Election", "Climate", "War", "Nationalism", "Immigration"
    ]

    num_topics = len(topic_ids)
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=150, num_topics=num_topics, workers=4, passes=10)

    topic_words = {idx: [word for word, _ in lda_model.show_topic(idx, topn=5)] for idx in range(num_topics)}
    custom_topic_ids = {idx: topic_ids[idx] for idx in range(len(topic_ids))}

    dominant_topics = []
    for doc in df['tokens']:
        bow = dictionary.doc2bow(doc)
        topic_distribution = lda_model[bow]
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        dominant_topics.append(dominant_topic)

    df['dominant_topic'] = dominant_topics
    df['dominant_topic_custom'] = df['dominant_topic'].map(custom_topic_ids)
    df['topic_keywords_custom'] = df['dominant_topic'].map(lambda x: ', '.join(topic_words[x]) if x is not None else "N/A")
    
    return df
