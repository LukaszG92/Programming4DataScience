import os
from groq import Groq
import pandas as pd

api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)

def get_narrative(text: str) -> dict:
    if len(text.split()) <= 3800:
        model = 'llama-3.3-70b-versatile'
    else:
        model = 'llama-3.1-8b-instant'

    prompt = f"""
    Analyze the following text and provide its narrative archetype.
    
    Return the result as a JSON with the following structure:
    {{
         'narrative archetype': 'value'
    }}

    - Ensure that the response is strictly the JSON format, without any extra text
    - The narrative archetype should be one of this:
    {{
        "Overcoming The Monster": "The protagonist battles a monstrous force threatening survival and represents a larger existential issue.",
        "Voyage And Return": "The protagonist leaves home, encounters a challenging new world, and returns transformed.",
        "Rags To Riches": "The protagonist rises from a low point to achieve empowerment and fulfillment.",
        "The Quest": "The protagonist sets out to find an object or person, facing mounting challenges and making sacrifices.",
        "Comedy": "A series of misunderstandings create conflict, eventually resolving happily.",
        "Tragedy": "The protagonist’s flaw or mistake leads to their undoing and fall.",
        "Rebirth": "The protagonist undergoes a redemptive journey leading to a hopeful outcome."
    }}

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
    response = response[response.find("{"):response.rfind("}") + 1]

    try:
        result_dict = eval(response)
        values = list(result_dict.values())
        result = {
            'narrative': values[0],
        }
    except Exception as e:
        result = {
            "narrative": None
        }

    return result


def narrative_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    text_narrative_df = pd.DataFrame(list(input_data['text'].map(get_narrative)))

    output_data = pd.concat([input_data, text_narrative_df], axis=1)

    return output_data