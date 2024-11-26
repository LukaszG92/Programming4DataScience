from pprint import pprint
from groq import Groq
import pandas as pd

api_key = ""

client = Groq(
    api_key=api_key,
)

def analyze_text(text: str):
    elementi = ['data del discorso', 'luogo del discorso', 'evento del discorso']
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""Analizza i metadati relativi al discorso fornito.
                            Restituiscimi {elementi} da questo: {text[:5900]}
                            La risposta deve essere solo {elementi}, senza nessun altro testo.""",
            }
        ],
        model="llama-3.1-70b-versatile",
    )

    pprint(chat_completion.choices[0].message.content)

df = pd.read_csv('datasets/speech-a.tsv', sep='\t', header=None, names=['author', 'code', 'text'])
df = df[104:]
for text in df['text']:
    analyze_text(text)