import os
from groq import Groq
import yake
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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


def create_sliding_windows(text, window_size, stride):
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
    return windows


def get_keywords(text, window_size=4096, stride=3584):
    yake_extractor = yake.KeywordExtractor(lan="en", n=2, top=3)

    windows = create_sliding_windows(text, window_size, stride)

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
        "keyword_1": sorted_keywords[0][0],
        "keyword_2": sorted_keywords[1][0],
        "keyword_3": sorted_keywords[2][0],
    }


def speech_based_feature_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    summarized_speech = input_data['text'].map(summarize_text)
    summarized_speech_df = pd.DataFrame(summarized_speech.tolist())

    keywords = input_data['text'].map(get_keywords)
    keywords_df = pd.DataFrame(keywords.tolist())

    output_data = pd.concat([input_data, summarized_speech_df, keywords_df], axis=1)

    return output_data
