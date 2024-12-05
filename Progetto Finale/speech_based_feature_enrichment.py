from groq import Groq
import pandas as pd

api_key = ""
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
