import os
import time
from groq import Groq
import yake
import pandas as pd
from transformers import AutoTokenizer

api_key = os.getenv("GROQ_KEY", "il_tuo_token_groq")
client = Groq(api_key=api_key)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
yake_extractor = yake.KeywordExtractor(lan="en", n=2, top=3)


def summarize_text(text: str) -> dict:
    try:
        if len(text.split()) <= 4700:
            model = 'llama-3.3-70b-versatile'
        elif len(text.split()) <= 10700:
            model = 'llama3-groq-70b-8192-tool-use-preview'
        else:
            model = 'llama-3.1-8b-instant'

        prompt = f"""
        Analyze the provided speech. 
        Return the result in JSON format with the following structure:
        {{
            "abstract": "value",
        }}
        - The "abstract" field must be a concise string (1-2 sentences) that highlights the primary topics and key points discussed in the text.  
        Speech: {text}
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        response = chat_completion.choices[0].message.content.strip()
        response = response[response.find("{"):response.find("}") + 1]

        result = eval(response)
        return result
    except Exception:
        return {"abstract": None}


def get_keywords(text: str, window_size=4096, stride=3584) -> dict:
    try:
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=window_size,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        windows = encoded["input_ids"]
        all_keywords = []
        for window_tensor in windows:
            window = tokenizer.decode(window_tensor, skip_special_tokens=True)
            keywords = yake_extractor.extract_keywords(window)
            all_keywords.extend(keywords)

        keyword_scores = {}
        for word, score in all_keywords:
            keyword_scores[word] = min(score, keyword_scores.get(word, float('inf')))

        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1])
        return {
            "keyword_1": sorted_keywords[0][0] if len(sorted_keywords) > 0 else None,
            "keyword_2": sorted_keywords[1][0] if len(sorted_keywords) > 1 else None,
            "keyword_3": sorted_keywords[2][0] if len(sorted_keywords) > 2 else None,
        }
    except Exception:
        return {"keyword_1": None, "keyword_2": None, "keyword_3": None}


def speech_based_feature_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:

    print("Generazione dei riassunti...")
    start_time = time.time()
    summarized_speech_df = pd.DataFrame(list(input_data['text'].map(summarize_text)))
    print(f"Riassunti generati in {time.time() - start_time:.2f} secondi.\n")


    print("Estrazione delle parole chiave...")
    start_time = time.time()
    keywords_df = pd.DataFrame(list(input_data['text'].map(get_keywords)))
    print(f"Parole chiave estratte in {time.time() - start_time:.2f} secondi.\n")

    output_data = pd.concat([input_data, summarized_speech_df, keywords_df], axis=1)

    return output_data
