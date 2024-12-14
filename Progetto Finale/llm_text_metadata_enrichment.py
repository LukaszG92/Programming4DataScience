import os
from groq import Groq
import pandas as pd

api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)

def analyze_text(text: str) -> dict:
    prompt = f"""
    Analyze the metadata related to the provided speech. 
    Return the result in JSON format with the following structure:
    {{
        "date of the speech": "value",
        "location of the speech": "value",
        "event of the speech": "value"
    }}

    - The "date of the speech" must be standardized to the format "YYYY-MM-DD". If the date is not available or invalid, return "None".
    - If the "location of the speech" or "event of the speech" is not available or invalid, return "None" for those fields.
    - Ensure that the response is strictly in the specified JSON format, with no extra text.

    Speech: {text[:5900]}
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
    )

    response = chat_completion.choices[0].message.content.strip()
    response = response[response.find("{"):response.rfind("}") + 1]
    try:
        result = eval(response)
        values = list(result.values())
        result = {
            "date of the speech": values[0],
            "location of the speech": values[1],
            "event of the speech": values[2]
        }

    except Exception as e:
        result = {
            "date of the speech": None,
            "location of the speech": None,
            "event of the speech": None,
        }
    return result

def text_metadata_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    text_metadata_df = pd.DataFrame(list(input_data['text'].map(analyze_text)))

    output_data = pd.concat([input_data, text_metadata_df], axis=1)

    return output_data