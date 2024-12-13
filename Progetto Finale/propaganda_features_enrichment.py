import pandas as pd
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import unicodedata
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def get_most_common_ngrams(text: str) -> dict:
    stop_words = set(stopwords.words('english'))
    punctuation_table = str.maketrans('', '', string.punctuation)  # Table for removing punctuation

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.translate(punctuation_table)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]  # Remove stop words

    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    common_bigram = Counter(bigrams).most_common(1)
    common_trigram = Counter(trigrams).most_common(1)

    bigram = common_bigram[0][0] if common_bigram else None
    trigram = common_trigram[0][0] if common_trigram else None

    return {'most_common_bigram': bigram, 'most_common_trigram': trigram}

def propaganda_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    ngrams_df = pd.DataFrame(list(input_data['text'].map(get_most_common_ngrams)))

    output_df = pd.concat([ngrams_df, input_data], axis=1)

    return output_df
