from groq import Groq
import pandas as pd

api_key = ""
client = Groq(api_key=api_key)

def get_narrative(text: str):
    prompt = f"""
    Analyze the provided text and generate a brief narrative summary emphasizing the main topics of the speech.  
    Return the result in JSON format with the following structure:  
    {{  
        "narrative": "value"  
    }}  
    
    - The "narrative" field should be a concise string (no more than 1-2 sentences) that highlights the primary topics and key points discussed in the text.  
    - If no clear narrative can be extracted, return "None".  
    - Ensure that the response is strictly in the specified JSON format, with no extra text.  
    
    Text: {text[:5900]}
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
    text_narrative = input_data['text'].map(get_narrative)
    text_narrative_df = pd.DataFrame(text_narrative.tolist())

    output_data = pd.concat([input_data, text_narrative_df], axis=1)

    return output_data