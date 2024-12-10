import time
import pandas as pd
import os
from groq import Groq

api_key = os.environ['GROQ_KEY']
client = Groq(api_key=api_key)

def get_propaganda_spans(text: str):
    span_metadata = []
    for i in range(0, len(text), 5700):
        print(f"Analyzing: {text[i:i+5700]}")
        prompt = f"""
        Search in this text the 2-3 most relevant propaganda slices, with the associated propaganda type.
        The result format have to be a JSON structured as follows, without any other addiction::
        {{
            "text of propaganda": "propaganda type",
            "text of propaganda": "propaganda type",
            "text of propaganda": "propaganda type"
        }}

        - The "text of propaganda" should be the slice of propaganda found in the dataset, if not found return None
        - The "propaganda type" is the classification of the text of propaganda, if not fount return None
        - The "text of propaganda" have to be a sentence, starting from a Capital letter to a point.
        - The propaganda type must be on of this:
        {{
            {{
                "technique":"Name calling"
                "definition":"attack an object/subject of the propaganda with an insulting label"
            }},
            {{
                "technique":"Repetition"
                "definition":"repeat the same message over and over"
            }},
            {{
                "technique":"Slogans"
                "definition":"use a brief and memorable phrase"
            }},
            {{
                "technique":"Appeal to fear"
                "definition":"support an idea by instilling fear against other alternatives"
            }},
            {{
                "technique":"Doubt"
                "definition":"questioning the credibility of some-one/something"
            }},
            {{
                "technique":"Exaggeration/minimizat."
                "definition":" exaggerate or minimize something"
            }},
            {{
                "technique":"Flag-Waving"
                "definition":"appeal to patriotism or identity"
            }},
            {{
                "technique":"Loaded Language"
                "definition":"appeal to emotions or stereotypes"
            }},
            {{
                "technique":"Reduction ad hitlerum"
                "definition":"disapprove an idea suggesting it is popular with groups hated by the audience"
            }},
            {{
                "technique":"Bandwagon"
                "definition":"appeal to the popularity of an idea"
            }},
            {{
                "technique":"Casual oversimplification"
                "definition":" assume a simple cause for a complex event"
            }},
            {{
                "technique":"Obfuscation, intentional vagueness"
                "definition":"use deliberately unclear and obscure expressions to confuse the audience"
            }},
            {{
                "technique":"Appeal to authority"
                "definition":"use authority's support as evidence"
            }},
            {{
                "technique":"Black&white fallacy"
                "definition":"present only two options among many"
            }},
            {{
                "technique":"Thought terminating clichés"
                "definition":"phrases that discourage critical thought and meaningful discussions"
            }},
            {{
                "technique":"Red herring"
                "definition":"introduce irrelevant material to distract"
            }},
            {{
                "technique":"Straw men"
                "definition":"refute argument that was not presented"
            }},
            {{
                "technique":"Whataboutism"
                "definition":"charging an opponent with hypocrisy"
            }}
        }}

        Text: {text[i:i + 5700]}
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
        except:
            result = {}

        for key, value in result.items():
            start_offset = i+ text[i:i + 5700].find(key)
            if start_offset == -1:
                continue
            end_offset = start_offset + len(key)
            propaganda_type = value
            span_metadata.append({
                'start_offset': start_offset,
                'end_offset': end_offset,
                'propaganda_type': propaganda_type
            })

    return span_metadata


def propaganda_span_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:

    print("Individuazione degli span di propaganda...")
    start_time = time.time()
    input_data['span_metadata'] = input_data['text'].apply(get_propaganda_spans)

    exploded_df = input_data.explode('span_metadata', ignore_index=True)

    span_metadata_df = pd.json_normalize(exploded_df['span_metadata'])

    final_df = pd.concat([exploded_df.drop(columns=['span_metadata']), span_metadata_df], axis=1)

    final_df['span_index'] = final_df.groupby('index').cumcount()

    final_df.set_index(['index', 'span_index'], inplace=True)
    print(f"Propaganda span individuati e dataset ristrutturato in {time.time() - start_time:.2f} secondi.\n")

    return final_df