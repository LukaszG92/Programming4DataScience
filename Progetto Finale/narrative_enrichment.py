import os
from groq import Groq
import pandas as pd

api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)

def get_narrative(text: str) -> dict:
    if len(text.split()) <= 4700:
        model = 'llama-3.3-70b-versatile'
    elif len(text.split()) <= 10700:
        model = 'llama3-groq-70b-8192-tool-use-preview'
    else:
        model = 'llama-3.1-8b-instant'

    prompt = f"""
    Analyze the following text and provide a detailed explanation of its narrative scheme, including the underlying themes, emotions, and storytelling framework.
    Identify whether it aligns with any classical or modern narrative archetypes (e.g., hero’s journey, rise and fall, transformation) and explain how the text develops its core message through its narrative structure.
    Additionally, suggest contextual or symbolic elements that enhance the understanding of the story being told.
    
    Return the result as a JSON with the following structure:
    {{
         'narrative archetype': 'value'
    }}

        - Ensure that the response is strictly the JSON format, without any extra text
    
    Text:  {text}
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    response = chat_completion.choices[0].message.content.strip()
    response = response[response.find("{"):response.find("}") + 1]
    print(response)
    try:
        result = eval(response)
    except Exception as e:
        result = {
            "narrative": None,
            "error": str(e)
        }

    print(result)
    return result


def narrative_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    text_narrative_df = pd.DataFrame(list(input_data['text'].map(get_narrative)))

    output_data = pd.concat([input_data, text_narrative_df], axis=1)

    return output_data