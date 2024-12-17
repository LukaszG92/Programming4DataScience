#### **Modulo** `wikidata_author_enrichment`
Questo modulo consente di arricchire un DataFrame contenente informazioni sugli autori con dati estratti da Wikidata, come date di nascita e morte, nazionalità e posizione ricoperte.
##### **Funzione principale**
`author_enrichment(input_data: pd.DataFrame) -> pd.DataFrame`
Questa funzione:
1. Mappa i cognomi degli autori nei loro nomi completi.
2. Esegue query SPARQL su Wikidata per arricchire i dati degli autori.
3. Combina le informazioni originali con quelle arricchite in un unico DataFrame.
**Parametri**
- `input_data` (`pd.DataFrame`): DataFrame contenente una colonna `author` con i nomi degli autori.
**Output**
- `pd.DataFrame`: DataFrame arricchito, con colonne aggiuntive contenenti dati estratti da Wikidata:
	- `date_of_birth`: Data di nascita.
	- `date_of_death`: Data di morte (se disponibile).
	- `nationality`: Nazionalità.
	- `position`: Posizioni ricoperte.
**Implementazione**
```python
def author_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    print("Mapping del nome completo degli autori...")
    start_time = time.time()
    
    input_data['author'] = input_data['author'].map(author_mapping)
    
    print(f"Nomi completi mappati in {time.time() -        start_time:.2f} secondi.\n")
    
    print("Query WikiData per le informazioni sugli autori...")
    start_time = time.time()
    
    author_info_df = pd.DataFrame(
						    list(input_data['author'].map(get_author_info)))
						    
    print(f"Query completate in {time.time() - start_time:.2f} secondi.\n")
    
    output_data = pd.concat([input_data, author_info_df], axis=1)
    
    return output_data
```
---
##### **Funzioni di supporto**
1. **Mappatura dei nomi**
	La funzione utilizza il dizionario `author_mapping` per convertire i nomi abbreviati degli autori nei loro nomi completi.
```python
	author_mapping = {
	    'Obama': 'Barack Obama',
	    'Churchill': 'Winston Churchill',
	    'Trump': 'Donald Trump',
	    'Goebbels': 'Joseph Goebbels'
	}
```
2. **Recupero informazioni da Wikidata**
	La funzione `get_author_info(name: str) -> dict` invia una query SPARQL all'endpoint di Wikidata per recuperare informazioni su un autore specifico.
	**Parametri**
	- `name` (`str`): Nome completo dell'autore.
	**Output**
	- `dict`: Dizionario contenente i dati estratti:
		  - `date_of_birth`
		  - `date_of_death` (opzionale)
		  - `nationality` (opzionale)
		  - `position` (opzionale)
	La funzione gestisce errori durante le query SPARQL restituendo valori `None` per i dati mancanti.
	**Implementazione**
```python
	def get_author_info(name: str) -> dict:
	    print(f'Fetching author info for {name}')
	    
	    query = build_query(name)
	    sparql.setQuery(query)
	    sparql.setReturnFormat(JSON)
	    
	    try:
	        results = sparql.query().convert()
	        if results['results']['bindings']:
	            result = results['results']['bindings'][0]
	            return parse_result(result)
	            
	    except Exception as e:
	        print(f"Errore durante la query per {name}: {e}")
	        return {
	            'date_of_birth': None,
	            'date_of_death': None,
	            'nationality': None,
	            'position': None
	        }
```
3. **Costruzione della query SPARQL**
	La funzione `build_query(name: str) -> str` genera una query SPARQL per ottenere i dettagli su un autore. Nello specifico qui vediamo una query d'esempio, seguita dall'implementazione del metodo
```sparql
	SELECT ?birthDate ?deathDate ?nationalityLabel ?positionLabel
	WHERE {
	    ?person rdfs:label "Barack Obama"@en.
	    ?person wdt:P569 ?birthDate.
	    OPTIONAL { ?person wdt:P570 ?deathDate. }
	    OPTIONAL { ?person wdt:P27 ?nationality. }
	    OPTIONAL { ?person wdt:P39 ?position. }
	    SERVICE wikibase:label {
	        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
	    }
	}
	LIMIT 10
```
```python
	def build_query(name: str) -> str:
	    return f"""
	        SELECT ?birthDate ?deathDate ?nationalityLabel ?positionLabel
	        WHERE {{
	            ?person rdfs:label "{name}"@en.
	            ?person wdt:P569 ?birthDate.
	            OPTIONAL {{ ?person wdt:P570 ?deathDate. }}
	            OPTIONAL {{ ?person wdt:P27 ?nationality. }}
	            OPTIONAL {{ ?person wdt:P39 ?position. }}
	            SERVICE wikibase:label {{
	                bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
	            }}
	        }}
	        LIMIT 10
	    """
```
4. **Parsing dei risultati**
	La funzione `parse_result(result: dict) -> dict` estrae i valori dai risultati JSON restituiti da Wikidata.
	**Implementazione**
```python
	def parse_result(result: dict) -> dict:
	    return {
	        'date_of_birth': result.get('birthDate', {}).get('value', None),
	        'date_of_death': result.get('deathDate', {}).get('value', None),
	        'nationality': result.get('nationalityLabel', {}).get('value', None),
	        'position': result.get('positionLabel', {}).get('value', None)
	    }
```
5. **Configurazione dell'endpoint SPARQL**
	L'endpoint SPARQL di Wikidata è configurato con un'intestazione HTTP personalizzata per identificare la richiesta.
```python
	sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
	sparql.addCustomHttpHeader("User-Agent",
	    "ProgrammingForDataScience/1.0 (Contact: lukasz.gajewski15@gmail.com)")
```
___
#### **Modulo** `llm_text_metadata_enrichment`
Questo modulo ha il compito di arricchire i discorsi contenuti in un dataset con metadata, come la data, la location e l'evento associato a ciascun discorso. Utilizza un modello di linguaggio (LLM) per analizzare i testi e estrarre informazioni pertinenti.
##### **Funzione principale**
`text_metadata_enrichment(input_data: pd.DataFrame) -> pd.DataFrame`
Questa funzione arricchisce un DataFrame di discorsi con i metadata ottenuti dalla funzione `analyze_text`, nello specifico:
1. Per ogni riga della colonna `text` del DataFrame di input, viene applicata la funzione `analyze_text` per ottenere i metadata.
2. I risultati vengono combinati in un nuovo DataFrame che include le colonne originali e i nuovi metadata.
**Parametri**
- `input_data` (`pd.DataFrame`): Un DataFrame contenente una colonna `text` con i discorsi da analizzare.
**Output**
- `output_data` (`pd.DataFrame`): Un DataFrame che include le informazioni originali, arricchite con colonne contenti i metadata estratti:
	- `date_of_the_speech`: Data in cui si è tenuto il discorso.
	- `location_of_the_speech`: Luogo in cui si è tenuto il discorso.
	- `event_of_the_speech`: Evento durante il quale si è tenuto il discorso
**Implementazione**
```python
def text_metadata_enrichment(input_data: pd.DataFrame) -> pd.DataFrame: 
    text_metadata_df = pd.DataFrame(list(input_data['text'].map(analyze_text)))
    output_data = pd.concat([input_data, text_metadata_df], axis=1)
    return output_data
```
---
##### **Funzioni di supporto**
`analyze_text(text: str) -> dict`
Questa funzione invia un testo a un modello di linguaggio per estrarre metadata relativi a un discorso, quali:
- **Date of the speech**: La data del discorso in formato "YYYY-MM-DD".
- **Location of the speech**: La località del discorso (se disponibile).
- **Event of the speech**: L'evento associato al discorso (se disponibile).
**Funzionamento:**
1. **Definizione del prompt**:
	Crea un prompt strutturato per il modello, includendo il testo da analizzare e la richiesta di una risposta in formato JSON standardizzato.
   ```python
   def analyze_text(text: str) -> dict:
       prompt = f"""
       Analyze the metadata related to the provided speech.
       Return the result in JSON format with the following structure:
       {{
           "date of the speech": "value",
           "location of the speech": "value",
           "event of the speech": "value"
       }}
       
       - The "date of the speech" must be standardized to the format
			"YYYY-MM-DD". If the date is not available or invalid, return "None".
       - If the "location of the speech" or "event of the speech" is not
			available or invalid, return "None" for those fields.
       - Ensure that the response is strictly in the specified JSON format,
		    with no extra text.
       
       Speech: {text[:5900]}
       """
   ```

2. **Chiamata al modello**:
	Esegue una richiesta al modello di linguaggio specificato (ad esempio, `"llama-3.3-70b-versatile"`) utilizzando il prompt definito.
   ```python
	chat_completion = client.chat.completions.create(
	   messages=[
		   {"role": "user",
		    "content": prompt}
		],
	   model="llama-3.3-70b-versatile",
	)
   ```
3. **Estrazione della risposta**:
	La risposta del modello è recuperata e pulita, estraendo solo la parte in formato JSON.
   ```python
	response = chat_completion.choices[0].message.content.strip()
	response = response[response.find("{"):response.rfind("}") + 1]
   ```
4. **Parsing della risposta**:
	La risposta viene convertita da JSON a un dizionario Python. In caso di errore, viene restituito un dizionario con valori `None` per i campi richiesti.
```python
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
```
---
#### **Modulo** `text_based_feature_enrichment`
Il modulo `text_based_feature_enrichment` definisce diverse funzioni per arricchire un dataset di testo con metriche basate su caratteristiche linguistiche, sentiment, emozioni e topic. Ecco le principali funzioni incluse:
1. `get_descriptive_and_readability`:
	Calcola le metriche descrittive e di leggibilità di una colonna di testo, come la lunghezza dei token, la lunghezza delle frasi, e la complessità della sintassi. Utilizza SpaCy per analizzare i testi e rimuove metriche non rilevanti.
2. `dispatcher`:
	Analizza i token di un documento per classificare e contare i lemmi di verbi, sostantivi e aggettivi. I risultati vengono aggiornati in dizionari separati e utilizzati per calcolare metriche sui tipi di parole.
3. `get_words_metrics`:
	Restituisce un dizionario con il conteggio dei verbi, sostantivi e aggettivi più comuni in un testo, calcolando anche i lemmi più frequenti per ogni categoria grammaticale.
4. `get_sentiment`:
	Utilizza un modello pre-addestrato per analizzare il sentimento di un testo. Supporta testi lunghi tramite una tecnica di "sliding window" che suddivide il testo in segmenti e calcola la probabilità del sentimento per ciascun segmento.
5. `get_emotions`:
	Estrae le emozioni predominanti (le tre emozioni principali) da un testo, utilizzando un modello pre-addestrato che gestisce anche testi lunghi con una finestra scorrevole.
6. `get_topic`:
	Estrae i topic principali da un testo utilizzando un modello LDA (Latent Dirichlet Allocation) per l'analisi dei topic. Viene applicata una pre-elaborazione per rimuovere i token non rilevanti, e i topic vengono estratti in base a una rappresentazione bag-of-words. La visualizzazione interattiva dei topic estratti è disponibile [QUI](https://dellasala.github.io/tmvisualizer/#topic=1&lambda=1&term=).
7. `text_based_enrichment`:
	Funzione di alto livello che aggrega tutte le funzioni precedenti per arricchire il dataset con le informazioni calcolate (metriche delle parole, descrizione e leggibilità, sentiment, emozioni, topic). Utilizza i metodi definiti per calcolare e aggiungere queste feature al DataFrame di input.
##### **Funzione principale**
`text_based_enrichment`:
La funzione arricchisce un dataset di testo calcolando diverse metriche basate sul contenuto testuale, inclusi verbi, sostantivi, aggettivi, metriche descrittive e di leggibilità, sentiment, emozioni e argomenti principali. .
**Input**:
- `input_data`: Un DataFrame contenente una colonna `text` con i discorsi da analizzare.
**Output**:
- Un `pandas.DataFrame` che contiene le seguenti colonne:
	- Colonne derivate da `get_words_metrics`: metriche relative a sostantivi, verbi e aggettivi, come il conteggio totale e le parole più comuni per ciascuna categoria grammaticale.
	- Colonne derivate da `get_descriptive_and_readability`: metriche descrittive e di leggibilità (ad esempio, lunghezza del testo, complessità, etc.).
	- Colonna `'sentiment'`: il sentiment del testo (positivo, negativo, neutro).
	- Colonne derivate da `get_emotions`: le emozioni prevalenti nel testo, con le prime 3 emozioni identificate.
	- Colonne derivate da `get_topic`: il topic dominante e le prime tre parole chiave per ciascun topic.
**Funzionamento**:
1. **Calcolo delle metriche delle parole**:
	   - La funzione `get_words_metrics` viene applicata a ciascun documento nel dataset per calcolare le metriche relative a sostantivi, verbi e aggettivi.
2. **Calcolo delle metriche descrittive e di leggibilità**:
	   - La funzione `get_descriptive_and_readability` calcola varie metriche descrittive e di leggibilità dei testi.
3. **Calcolo del sentiment**:
	   - Il modello di sentiment viene applicato a ciascun documento per determinare il sentiment complessivo (positivo, negativo, neutro).
4. **Calcolo delle emozioni**:
	   - La funzione `get_emotions` calcola le emozioni predominanti in ogni documento.
5. **Estrazione dei topic**:
	   - La funzione `get_topic` utilizza il modello LDA per identificare il topic dominante per ciascun documento e le parole chiave principali di ogni topic. (Visualizzazione dei topic estratti [QUI](https://dellasala.github.io/tmvisualizer/#topic=1&lambda=1&term=))
6. **Composizione del DataFrame finale**:
	   - Tutti i risultati vengono concatenati in un unico DataFrame, che viene poi filtrato per rimuovere i documenti con una sola frase.
**Implementazione**:
```python
def text_based_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    print("Calcolo delle metriche delle parole...")
    start_time = time.time()
    word_metrics_df = pd.DataFrame(
						    list(input_data['text'].map(get_words_metrics)))
    print(f"Metriche delle parole calcolate in
							{time.time() - start_time:.2f} secondi.")
	
    print("Calcolo delle metriche descrittive e di leggibilità...")
    start_time = time.time()
    descriptive_df = get_descriptive_and_readability(input_data['text'])
    print(f"Metriche descrittive e di leggibilità calcolate in
						    {time.time() - start_time:.2f} secondi.")
	
    print("Calcolo del sentiment...")
    start_time = time.time()
    sentiment_model.to(device)
    sentiment_df = input_data['text'].map(get_sentiment)
    sentiment_df = sentiment_df.to_frame(name='sentiment')
    print(f"Sentiment calcolato in {time.time() - start_time:.2f} secondi.")
	
    print("Calcolo delle emozioni...")
    start_time = time.time()
    emotion_model.to(device)
    emotion_df = pd.DataFrame(list(input_data['text'].map(get_emotions)))
    print(f"Emozioni calcolate in {time.time() - start_time:.2f} secondi.")
	
    print("Estrazione dei topic...")
    start_time = time.time()
    topics_df = get_topic(input_data['text'])
    print(f"Topic estratti in {time.time() - start_time:.2f} secondi.")
	
    output_data = pd.concat([word_metrics_df, descriptive_df, sentiment_df,
						     emotion_df, topics_df], axis=1)
	
    output_data = output_data[output_data['n_sentences'] > 1]
	
    return output_data
```
##### **Funzioni secondarie**
1. **get_descriptive_and_readability**
	Calcola le metriche descrittive e di leggibilità per un insieme di testi, utilizzando il modello di SpaCy per l'elaborazione linguistica.
	**Input**: una serie di testi (ogni elemento è una stringa di testo).
	**Output**: un DataFrame che contiene diverse statistiche sui testi, tra cui:
		- Media e mediana della lunghezza delle frasi
		- Numero totale delle frasi
		- Numero di token totali, token unici, e il rapporto tra token 
		- Metriche di leggibilità
	**Funzionamento**:
	1. **Elaborazione dei testi con SpaCy**:
		   - La funzione `nlp.pipe(text_column)` viene utilizzata per elaborare i testi in modo efficiente. `nlp.pipe` applica il modello di SpaCy a ciascun documento del DataFrame in modo batch, il che significa che i testi vengono elaborati in parallelo per migliorare le performance, specialmente quando si lavora con grandi volumi di dati.
	2. **Estrazione delle metriche**:
		   - Viene chiamata la funzione `extract_df`, che raccoglie le metriche di leggibilità e statistiche descrittive per ogni documento. Questa funzione utilizza il modello linguistico di SpaCy per analizzare le strutture linguistiche, come la lunghezza dei token e delle frasi, la frequenza delle sillabe per token, e altro ancora.
	3. **Filtraggio delle metriche**:
		- La funzione `get_descriptive_and_readability` rimuove specifiche colonne, come la lunghezza dei caratteri e altre statistiche sui token, per restituire solo le metriche significative per la leggibilità e la complessità.
	**Implementazione**:
	```python
	def get_descriptive_and_readability(text_column: pd.Series) -> pd.DataFrame:
	    docs = nlp.pipe(text_column)
	    
	    metrics_to_extract = ["descriptive_stats", "readability"]
	    metrics_df = extract_df(docs, metrics=metrics_to_extract, 
							    include_text=False)
	    
	    labels = ['token_length_mean', 'token_length_median','token_length_std', 
	              'sentence_length_std', 'syllables_per_token_mean', 
	              'syllables_per_token_median', 'syllables_per_token_std', 
	              'n_characters']
	    
	    metrics_df.drop(labels, axis=1, inplace=True)
	    return metrics_df
	```
2. **dispatcher**
	Raccoglie informazioni sulla frequenza dei verbi, sostantivi e aggettivi presenti in un testo, aggiornando i dizionari corrispondenti per ciascuna categoria grammaticale.
	**Input**:
		- un oggetto `Token` di SpaCy, che rappresenta una parola nel testo, insieme a tre dizionari per i verbi, i sostantivi e gli aggettivi.
	**Output**:
		- Dizionari aggiornati contenenti i lemmi e la loro frequenza nel testo per ogni categoria grammaticale.
	**Funzionamento**:
	1. **Filtraggio delle stop words**:
		   - Se un token è una "stop word", la funzione restituisce `None` e non aggiorna i dizionari.
	2. **Aggiornamento dei dizionari**:
		   - Se il token è un verbo (`token.pos_ == "VERB"`), viene aggiunto al dizionario dei verbi.
		   - Se il token è un sostantivo (`token.pos_ == "NOUN"`), viene aggiunto al dizionario dei sostantivi.
		   - Se il token è un aggettivo (`token.pos_ == "ADJ"`), viene aggiunto al dizionario degli aggettivi.
	**Implementazione**:
	```python
	def dispatcher(token: Token, verbs: dict,
					nouns: dict, adjectives: dict) -> None:
	    
	    if token.is_stop:
	        return None
	
	    token_lemma = token.lemma_.lower()
	    if token.pos_ == "VERB":
	        verbs[token_lemma] = verbs.get(token_lemma, 0) + 1
	
	    if token.pos_ == "NOUN":
	        nouns[token_lemma] = nouns.get(token_lemma, 0) + 1
	
	    if token.pos_ == "ADJ":
	        adjectives[token_lemma] = adjectives.get(token_lemma, 0) + 1
	```
3. **get_words_metrics**
	Calcola le metriche relative ai verbi, sostantivi e aggettivi in un dato testo, restituendo un dizionario con i conteggi e le parole più comuni per ciascuna categoria grammaticale.
	**Input**: una stringa di testo.
	**Output**: un dizionario che contiene:
	  - Il numero totale di sostantivi, verbi e aggettivi.
	  - Le parole più comuni per ogni categoria grammaticale.
	**Funzionamento**:
	1. **Analisi del testo**:
		   - Viene creato un oggetto `Doc` tramite SpaCy, che rappresenta l'intero testo.
	2. **Raccolta dei lemmi**:
		   - La funzione `dispatcher` viene applicata a ciascun token nel testo per raccogliere la frequenza dei lemmi di verbi, sostantivi e aggettivi.
	3. **Restituzione dei risultati**:
		   - Dopo aver calcolato le frequenze per ciascun tipo di parola, la funzione restituisce il numero totale di verbi, sostantivi e aggettivi, oltre ai lemmi più comuni per ciascuna categoria.
	**Implementazione**:
	```python
	def get_words_metrics(text: str) -> dict:
	    doc = nlp(text)
	    verbs = {}
	    nouns = {}
	    adjectives = {}
	    
	    list(map(lambda token: dispatcher(token, verbs, nouns, adjectives), doc))
	
	    most_common_verb = max(verbs, key=verbs.get) if verbs else None
	    most_common_noun = max(nouns, key=nouns.get) if nouns else None
	    most_common_adjective = max(adjectives, key=adjectives.get) if
														    adjectives else None
	    
	    pos_counts = {
	        "nouns_count": sum(nouns.values()),
	        "verbs_count": sum(verbs.values()),
	        "adjectives_count": sum(adjectives.values()),
	        "most_common_verb": most_common_verb,
	        "most_common_noun": most_common_noun,
	        "most_common_adjective": most_common_adjective,
	    }
	
	    return pos_counts
	```
4. **get_sentiment**
	Calcola il sentiment di un testo suddividendo il testo in finestre (windows) e analizzandole tramite un modello di sentiment analysis pre-addestrato. Restituisce il sentiment prevalente tra tutte le finestre.
	**Input**: 
	- Una stringa di testo (`text`): il testo da analizzare.
	- Un intero (`window_size`): la dimensione della finestra in token (default 512).
	- Un intero (`overlap`): il numero di token che si sovrappongono tra finestre consecutive (default 128).
	**Output**:
	- Una stringa che rappresenta il sentiment prevalente nel testo, come previsto dal modello.
	**Funzionamento**:
	1. **Tokenizzazione del testo**:
		   - Il testo viene suddiviso in finestre (window) di dimensione `window_size`, con sovrapposizione di `overlap` token tra finestre consecutive tramite un tokenizer predefinito (`sentiment_tokenizer`).
	2. **Analisi delle finestre**:
		   - Ogni finestra viene inviata al modello di sentiment analysis (`sentiment_model`) per ottenere le probabilità delle diverse etichette di sentiment. Le probabilità vengono calcolate utilizzando la funzione `softmax` sui logits del modello.
	3. **Calcolo del sentiment prevalente**:
		   - Dopo aver analizzato tutte le finestre, le probabilità per ogni finestra vengono mediate per ottenere una probabilità media per ogni etichetta di sentiment.
	4. **Predizione finale**:
		   - Il sentiment prevalente viene determinato tramite la previsione della classe che ha la probabilità più alta dopo la media delle probabilità di tutte le finestre.
	**Implementazione**:
	```python
	def get_sentiment(text: str, window_size=512, overlap=128)
														-> Union[str, None]:
	    
	    encoded = sentiment_tokenizer(
	        text,
	        return_tensors="pt",
	        truncation=True,
	        padding=True,
	        max_length=window_size,
	        stride=overlap,
	        return_overflowing_tokens=True
	    )

	    num_windows = len(encoded["input_ids"]

	    all_probabilities = []
	    for i in range(num_windows):
	        input_ids = encoded["input_ids"][i].to(device)
	        attention_mask = encoded["attention_mask"][i].to(device)
	        if input_ids.dim() == 1:
	            input_ids = input_ids.unsqueeze(0)
	            attention_mask = attention_mask.unsqueeze(0)
	
	        if input_ids.size(1) == 0:
	            continue
	
	        with torch.no_grad():
	            outputs = sentiment_model(
		            input_ids=input_ids,
		            attention_mask=attention_mask
		        )
		        
	            logits = outputs.logits
	
	            if logits.dim() == 1:
	                logits = logits.unsqueeze(0)
	
	            probabilities = torch.softmax(logits, dim=-1)
	            all_probabilities.append(probabilities)
	
	    avg_probabilities = torch.mean(torch.stack(all_probabilities), dim=0)
	
	    prediction = torch.argmax(avg_probabilities, dim=-1)
	    majority_vote = sentiment_model.config.id2label[prediction.item()]
	
	    return majority_vote
	```
5. **get_emotions**
	Calcola le emozioni presenti in un testo suddividendo il testo in finestre (windows) e analizzandole tramite un modello di analisi delle emozioni pre-addestrato. Restituisce le tre emozioni principali rilevate nel testo, ordinandole per probabilità.
	**Input**: 
	- Una stringa di testo (`text`): il testo da analizzare.
	- Un intero (`window_size`): la dimensione della finestra in token (default 512).
	- Un intero (`overlap`): il numero di token che si sovrappongono tra finestre consecutive (default 128).
	**Output**:
	- Un dizionario con le tre emozioni più probabili nel testo, ordinate per probabilità decrescente. Le chiavi del dizionario sono:
		  - `'emotion_1'`: la prima emozione con la probabilità più alta.
		  - `'emotion_2'`: la seconda emozione con la probabilità più alta.
		  - `'emotion_3'`: la terza emozione con la probabilità più alta.
	Se non vengono rilevate emozioni, il dizionario conterrà valori `None` per tutte le emozioni.
	**Funzionamento**:
		1. **Tokenizzazione del testo**:
			   - Il testo viene suddiviso in finestre (window) di dimensione `window_size`, con sovrapposizione di `overlap` token tra finestre consecutive tramite un tokenizer predefinito (`emotion_tokenizer`).
		2. **Analisi delle finestre**:
			   - Ogni finestra viene inviata al modello di analisi delle emozioni (`emotion_model`) per ottenere i logits associati alle diverse emozioni.
		3. **Calcolo delle probabilità**:
			   - Dopo aver analizzato tutte le finestre, i logits per ogni finestra vengono concatenati e mediati per ottenere i logits complessivi per il testo.
			   - Le probabilità delle emozioni vengono calcolate tramite la funzione `softmax` sui logits mediati.
		4. **Restituzione delle emozioni**:
			   - Le probabilità vengono ordinate in modo decrescente e vengono restituite le tre emozioni con la probabilità più alta, insieme ai loro nomi corrispondenti tramite `id2label`.
	**Implementazione**:
	```python
	def get_emotions(text: str, window_size=512, overlap=128) -> dict:
		
	    encoded = emotion_tokenizer(
	        text,
	        return_tensors="pt",
	        truncation=True,
	        padding="max_length",
	        max_length=window_size,
	        stride=overlap,
	        return_overflowing_tokens=True
	    )
		
	    input_ids = encoded["input_ids"].to(device)
		
	    num_windows = input_ids.size(0)
	    
	    results = []
	    for i in range(num_windows):
	        window_input_ids = input_ids[i:i + 1]
	        with torch.no_grad():
	            outputs = emotion_model(window_input_ids)
	            logits = outputs.logits
	            results.append(logits)
		
	    if results:
	        all_logits = torch.cat(results)
	        avg_logits = torch.mean(all_logits, dim=0)
	        probabilities = torch.nn.functional.softmax(avg_logits,
												        dim=-1).cpu().numpy()
	        id2label = emotion_model.config.id2label
	        sorted_indices = probabilities.argsort()[::-1]
	        
	        return {
	            'emotion_1': id2label[sorted_indices[0]],
	            'emotion_2': id2label[sorted_indices[1]],
	            'emotion_3': id2label[sorted_indices[2]]
	        }
		
	    return {
	        'emotion_1': None,
	        'emotion_2': None,
	        'emotion_3': None
	    }
	
	```
6. **get_topic**
	Identifica il topic dominante per ciascun documento di un dataset di testo, utilizzando il modello LDA (Latent Dirichlet Allocation). Restituisce un DataFrame con i topic assegnati a ciascun documento, insieme alle parole chiave più rilevanti per ogni topic.
	**Input**: 
	- Una colonna di testo in formato `pandas.Series` (`text_column`): una serie di documenti (una stringa per ciascun documento) da analizzare.
	**Output**:
	- Un `pandas.DataFrame` con le seguenti colonne:
		  - `'topic'`: il topic dominante assegnato a ciascun documento.
		  - `'topic_kw_1'`, `'topic_kw_2'`, `'topic_kw_3'`: le prime tre parole chiave del topic dominante per ciascun documento.
	**Funzionamento**:
	1. **Preprocessing del testo**:
		   - Viene creato un oggetto `Doc` tramite SpaCy per ogni documento.
	2. **Creazione del dizionario e del corpus**:
		   - Viene creato un dizionario che mappa i token dei documenti a un indice univoco. Il corpus è una rappresentazione in formato "bag of words" (BoW) dei documenti, utilizzata per addestrare il modello LDA.
	3. **Addestramento del modello LDA**:
		   - Un modello LDA viene addestrato sul corpus. Il numero di topic da identificare è fissato a 7.
	4. **Estrazione delle parole chiave per ogni topic**:
		   - Le 5 parole più comuni per ciascun topic vengono estratte e memorizzate in un dizionario.
	5. **Calcolo del topic dominante per ogni documento**:
		   - Per ciascun documento, viene calcolata la distribuzione dei topic e viene assegnato il topic con la probabilità più alta.
	6. **Creazione del DataFrame finale**:
		   - I risultati (topic dominante e parole chiave) vengono aggiunti al DataFrame originale, mentre la colonna dei token viene rimossa per ottenere il formato finale.
	**Implementazione**:
	```python
	def get_topic(text_column: pd.Series) -> pd.DataFrame:
	    docs = nlp.pipe(text_column)
		
	    removal = {'ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART',
				   'DET', 'ADP', 'SPACE', 'NUM', 'SYM'}
		
	    def preprocess_text(doc: Doc) -> list[str]:
	        return [token.lemma_.lower() for token in doc
										 if token.pos_ not in removal and
								         not token.is_stop and token.is_alpha]
		
	    tokens = list(map(preprocess_text, docs))
	    tokens = [tok if tok else ['unknown'] for tok in tokens]
		
	    output_df = pd.DataFrame({'tokens': tokens})
		
	    dictionary = Dictionary(output_df['tokens'])
	    corpus = [dictionary.doc2bow(doc) for doc in output_df['tokens']]
		
	    num_topics = 7
	    lda_model = LdaMulticore(
	        corpus=corpus,
	        id2word=dictionary,
	        iterations=150,
	        num_topics=num_topics,
	        workers=2,
	        passes=10
	    )
		
	    topic_words = {idx: [word for word, _ in lda_model.show_topic(idx, 
			topn=5)] for idx in range(num_topics)}
	
	    dominant_topics = []
	    for doc in output_df['tokens']:
	        bow = dictionary.doc2bow(doc)
	        topic_distribution = lda_model[bow]
	        if topic_distribution:
	            dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
	        else:
	            dominant_topic = -1
	        dominant_topics.append(dominant_topic)
		
	    output_df['topic'] = dominant_topics
	    output_df['topic_kw_1'] = output_df['topic'].map(
							    lambda x: topic_words.get(x, ['No topic'])[0])
	    output_df['topic_kw_2'] = output_df['topic'].map(
							    lambda x: topic_words.get(x, ['No topic'])[1])
	    output_df['topic_kw_3'] = output_df['topic'].map(
							    lambda x: topic_words.get(x, ['No topic'])[2])
		
	    output_df.drop('tokens', axis=1, inplace=True)
		
	    return output_df
	```
#### **Modulo** `speech_based_feature_enrichment`
Questo modulo arricchisce un DataFrame contenente discorsi con due tipi di informazioni estratte automaticamente dai testi: riassunti e parole chiave.
Utilizza modelli di linguaggio (LLM) e strumenti di estrazione (YAKE) per analizzare i testi e restituire informazioni significative.
##### **Funzione principale**
`speech_based_feature_enrichment(input_data: pd.DataFrame) -> pd.DataFrame`
Questa funzione arricchisce un DataFrame di discorsi con i riassunti e le parole chiave ottenuti dalle funzioni `summarize_text` e `get_keywords`. Il flusso di lavoro è:
1. Per ogni riga della colonna `text` del DataFrame di input, viene applicata la funzione `summarize_text` per generare un riassunto del testo.
2. Successivamente, viene applicata la funzione `get_keywords` per estrarre le parole chiave principali dal testo.
3. I risultati vengono combinati in un nuovo DataFrame che include le colonne originali, i riassunti e le parole chiave estratte.
**Input**
- `input_data` (`pd.DataFrame`): Un DataFrame contenente una colonna `text` con i discorsi da analizzare.
**Output**
- `output_data` (`pd.DataFrame`): Un DataFrame che include le informazioni originali, arricchite con:
	  - `abstract`: Un riassunto del discorso (1-2 frasi).
	  - `keyword_1`, `keyword_2`, `keyword_3`: Le tre parole chiave più significative estratte dal discorso.
**Implementazione**
```python
def speech_based_feature_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:

    print("Generazione dei riassunti...")
    start_time = time.time()
    summarized_speech_df = pd.DataFrame(
							    list(input_data['text'].map(summarize_text)))
    print(f"Riassunti generati in {time.time() - start_time:.2f} secondi.\n")

    print("Estrazione delle parole chiave...")
    start_time = time.time()
    keywords_df = pd.DataFrame(list(input_data['text'].map(get_keywords)))
    print(f"Parole chiave estratte in {time.time() - start_time:.2f} secondi.\n")

    output_data = pd.concat([input_data, summarized_speech_df, keywords_df],
						    axis=1)

    return output_data
```
---
##### **Funzioni di supporto**
`sumarize_text(text: str) -> dict`
Questa funzione genera un riassunto conciso di un testo fornito. Il riassunto è realizzato tramite un modello di linguaggio (LLM) che, a seconda della lunghezza del testo, seleziona un modello adatto (potente o rapido). Restituisce un dizionario con il campo `"abstract"` contenente il riassunto del testo.
**Funzionamento:**
1. **Selezione del modello**:
	-  Viene scelto un modello LLM in base alla lunghezza del testo. Per testi brevi (meno di 5500 parole) viene usato un modello potente, per testi più lunghi viene selezionato un modello rapido.
2. **Creazione del prompt**:
	- Il testo viene analizzato per generare un riassunto conciso di 1-2 frasi.
3. **Chiamata al modello**:
	- Il testo viene inviato a un modello LLM per generare il riassunto.
4. **Estrazione del riassunto**:
   - La risposta del modello viene estratta e restituita come un dizionario con il campo `"abstract"`.
	```python
	def summarize_text(text: str) -> dict:
	    try:
	        if len(text.split()) <= 5500:
	            model = 'llama-3.3-70b-versatile'
	        else:
	            model = 'llama-3.1-8b-instant'
	        prompt = f"""
	        Analyze the provided speech.
	        Return the result in JSON format with the following structure:
	        {{
	            "abstract": "value",
	        }}
	        - The "abstract" field must be a concise string (1-2 sentences)
		      that highlights the primary topics and key points discussed in
		      the text.  
		      
	        Speech: {text}
	        """
	        
	        chat_completion = client.chat.completions.create(
	            messages=[{"role": "user", "content": prompt}],
	            model=model,
	        )
	        
	        response = chat_completion.choices[0].message.content.strip()
	        response = response[response.find("{"):response.find("}") + 1]
	        result = eval(response)
	        values = list(result.values())
	        
	        return {"abstract": values[0]}
		
	    except Exception as e:
	        return {"abstract": None}
	```
___
`get_keywords(text: str, window_size=4096, stride=3584) -> dict`
Questa funzione estrae le parole chiave principali da un testo utilizzando YAKE (Yet Another Keyword Extractor). Se il testo supera il limite di token, il testo viene suddiviso in finestre (sliding windows) e per ciascuna finestra vengono estratte le parole chiave. Le parole chiave vengono ordinate per rilevanza e vengono restituite le prime tre.
**Funzionamento:**
1. **Suddivisione del testo**:
	   - Il testo viene suddiviso in finestre per rispettare il limite di token del modello.
2. **Estrazione delle parole chiave**:
	   - Vengono estratte le parole chiave per ciascuna finestra utilizzando YAKE.
3. **Selezione delle parole chiave**:
	- Le parole chiave vengono ordinate per punteggio di rilevanza e vengono restituite le tre più rilevanti.
```python
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
	            window = tokenizer.decode(window_tensor, 
								          skip_special_tokens=True)
	            keywords = yake_extractor.extract_keywords(window)
	            all_keywords.extend(keywords)
	
	        keyword_scores = {}
	        for word, score in all_keywords:
	            keyword_scores[word] = min(score,
								    keyword_scores.get(word, float('inf')))
	
	        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1])
	        return {
	            "keyword_1": sorted_keywords[0][0] if len(sorted_keywords) > 0 
														            else None,
	            "keyword_2": sorted_keywords[1][0] if len(sorted_keywords) > 1 
															        else None,
	            "keyword_3": sorted_keywords[2][0] if len(sorted_keywords) > 2 
														            else None,
	        }
	
	    except Exception:
	        return {"keyword_1": None, "keyword_2": None, "keyword_3": None}
```
___

#### **Modulo** `propaganda_features_enrichment`
`propaganda_enrichment`
La funzione `propaganda_enrichment` arricchisce un DataFrame contenente messaggi o testi con informazioni relative ai bigrammi (coppie di parole) e trigrammi (terzetti di parole) più comuni all'interno di ciascun testo.
**Input**
- `input_data` (`pd.DataFrame`): Un DataFrame contenente una colonna `text` con i testi da analizzare.
**Output**
- `output_df` (`pd.DataFrame`): Un DataFrame arricchito con due nuove colonne:
	  - `most_common_bigram`: Il bigramma più comune estratto dal testo.
	  - `most_common_trigram`: Il trigramma più comune estratto dal testo.
**Implementazione**
```python
def propaganda_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    ngrams_df = pd.DataFrame(
				    list(input_data['text'].map(get_most_common_ngrams)))
    
    output_df = pd.concat([ngrams_df, input_data], axis=1)
    
    return output_df
```
---
##### **Funzioni di supporto**
La funzione `get_most_common_ngrams` è progettata per identificare i bigrammi e trigrammi più frequenti in un dato testo. Viene prima normalizzato il testo (rimozione di accenti, punteggiatura, e stop words), successivamente vengono estratti i bigrammi e trigrammi usando la libreria `nltk`. Infine, vengono restituiti i bigrammi e trigrammi più comuni utilizzando `Counter` per calcolare la frequenza di occorrenza.
**Funzionamento**:
1. **Normalizzazione**:
	- Rimozione di punteggiatura, accenti e parole inutili (stop words).
2. **Tokenizzazione**:
	   - Suddivisione del testo in parole.
3. **Generazione degli n-grams**:
	   - Creazione di bigrammi e trigrammi.
4. **Calcolo della frequenza**:
	   - Utilizzo di `Counter` per determinare i n-grams più comuni.
```python
def get_most_common_ngrams(text: str) -> dict:
    stop_words = set(stopwords.words('english'))
    punctuation_table = str.maketrans('', '', string.punctuation)
    
    text = unicodedata.normalize('NFKD', text).encode('ascii',
													  'ignore').decode('utf-8')
    text = text.translate(punctuation_table)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    
    common_bigram = Counter(bigrams).most_common(1)
    common_trigram = Counter(trigrams).most_common(1)
    
    bigram = common_bigram[0][0] if common_bigram else None
    trigram = common_trigram[0][0] if common_trigram else None
    
    return {'most_common_bigram': bigram, 'most_common_trigram': trigram}
```
---
#### **Modulo** `narrative_enrichment`
`narrative_enrichment`
La funzione `narrative_enrichment` arricchisce un DataFrame contenente testi con un campo che descrive l'archetipo narrativo di ciascun testo. L'archetipo narrativo viene determinato analizzando il testo tramite un LLM, il quale identifica quale dei sette archetipi narrativi principali (ad esempio, "Overcoming The Monster" o "The Quest") meglio descrive il contenuto del testo.
**Input**
- `input_data` (`pd.DataFrame`): Un DataFrame contenente una colonna `text` con i testi da analizzare.
**Output**
- `output_data` (`pd.DataFrame`): Un DataFrame arricchito con una nuova colonna `narrative` che contiene l'archetipo narrativo associato a ciascun testo.
**Implementazione**
```python
def narrative_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    text_narrative_df = pd.DataFrame(list(input_data['text'].map(get_narrative)))
    
    output_data = pd.concat([input_data, text_narrative_df], axis=1)
    
    return output_data
```
---
##### **Funzioni di supporto** 
La funzione `get_narrative` analizza un testo e ne determina l'archetipo narrativo.
Se il testo è lungo meno di 3800 parole, viene utilizzato il modello `llama-3.3-70b-versatile`; altrimenti, se il testo è più lungo, viene usato il modello `llama-3.1-8b-instant`.
Il modello analizza il testo e restituisce un archetipo narrativo associato a uno dei seguenti valori:
	- **"Overcoming The Monster"**: Il protagonista combatte una forza mostruosa che minaccia la sopravvivenza.
	- **"Voyage And Return"**: Il protagonista lascia casa, affronta un mondo sconosciuto e ritorna trasformato.
	- **"Rags To Riches"**: Il protagonista passa da una condizione di povertà o sconfitta a una situazione di successo e realizzazione.
	- **"The Quest"**: Il protagonista intraprende una ricerca per un oggetto o persona, affrontando sfide e sacrifici.
	- **"Comedy"**: Una serie di incomprensioni crea conflitti che si risolvono felicemente.
	- **"Tragedy"**: Il protagonista commette un errore che porta alla sua rovina.
	- **"Rebirth"**: Il protagonista attraversa un percorso di redenzione che porta a un esito positivo.
**Funzionamento**:
	1. Il testo viene passato al modello di linguaggio.
	2. Il modello restituisce una risposta in formato JSON con l'archetipo narrativo identificato.
	3. Il risultato viene analizzato e convertito in un dizionario con la chiave `narrative`.
```python
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
	
	    - Ensure that the response is strictly the JSON format, without
		  any extra text
	    - The narrative archetype should be one of this:
	    {{
	        "Overcoming The Monster": "The protagonist battles a monstrous force
	         threatening survival and represents a larger existential issue.",
	        "Voyage And Return": "The protagonist leaves home, encounters a 
	         challenging new world, and returns transformed.",
	        "Rags To Riches": "The protagonist rises from a low point to achieve
	         empowerment and fulfillment.",
	        "The Quest": "The protagonist sets out to find an object or person,
	         facing mounting challenges and making sacrifices.",
	        "Comedy": "A series of misunderstandings create conflict, eventually
	         resolving happily.",
	        "Tragedy": "The protagonist’s flaw or mistake leads to their undoing
	         and fall.",
	        "Rebirth": "The protagonist undergoes a redemptive journey leading 
	        to a hopeful outcome."
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
	```
#### **Modulo** `propaganda_span_enrichment`
Questo modulo si occupa di arricchire un DataFrame con metadati relativi agli **span di propaganda** all'interno dei testi. La funzione principale `propaganda_span_enrichment` identifica i segmenti di testo che corrispondono a tecniche propagandistiche, analizzando il testo attraverso l'uso di un modello di linguaggio.
**Funzione principale**
`propaganda_span_enrichment`
Aggiunge al DataFrame originale una colonna che contiene gli "span" di propaganda, ossia i segmenti di testo che corrispondono a determinate tecniche propagandistiche, con la loro posizione all'interno del testo e il tipo di propaganda.
**Funzionamento**:
	1. **Identificazione degli span**: La funzione `get_propaganda_spans` esamina i testi e identifica i segmenti più rilevanti che utilizzano le tecniche propagandistiche definite.
	2. **Ristrutturazione del DataFrame**: Ogni span di propaganda viene trasformato in una riga separata, consentendo una facile analisi.
	3. **Esplosione del DataFrame**: La funzione esplode i dati per trattare ogni span come una riga separata, normalizzando i metadati associati.
	4. **Aggiunta degli indici**: Viene aggiunto un indice per ogni span all'interno di ogni riga originale, per mantenere la struttura del DataFrame intatta.
**Input**:
- `input_data` (`pd.DataFrame`): DataFrame contenente una colonna `text` con i testi da analizzare.
**Output**:
- `final_df` (`pd.DataFrame`): DataFrame arricchito con metadati sugli span di propaganda, inclusi i dettagli sulla posizione (offset di inizio e fine) e il tipo di propaganda.
**Implementazione**:
```python
def propaganda_span_enrichment(input_data: pd.DataFrame) -> pd.DataFrame:
    print("Individuazione degli span di propaganda...")
    start_time = time.time()
	
    input_data['span_metadata'] = input_data['text'].apply(get_propaganda_spans)
	
    exploded_df = input_data.explode('span_metadata', ignore_index=True)
	
    span_metadata_df = pd.json_normalize(exploded_df['span_metadata'])
	
    final_df = pd.concat([exploded_df.drop(columns=['span_metadata']), 
						  span_metadata_df], axis=1)
	
    final_df['span_index'] = final_df.groupby('index').cumcount()
	
    final_df.set_index(['index', 'span_index'], inplace=True)
    
    print(f"Propaganda span individuati e dataset ristrutturato in
		    {time.time() - start_time:.2f} secondi.\n")

    return final_df
```
**Funzioni di supporto**
La funzione `get_propaganda_spans` si occupa di estrarre gli **span di propaganda** da un testo, restituendo un dizionario con il "testo di propaganda" e il "tipo di propaganda" associato.
**Funzionamento**
1. **Suddivisione del testo**: Il testo viene suddiviso in blocchi da 5700 caratteri
2. **Creazione del prompt**: Un prompt viene creato per ogni blocco, chiedendo al modello di cercare le 2-3 frasi più rilevanti che utilizzano tecniche propagandistiche.
3. **Analisi con il modello**: Il modello risponde con i segmenti di testo e il tipo di propaganda.
4. **Raccolta degli span**: Per ciascun segmento trovato, vengono registrati l'offset di inizio e fine, e il tipo di propaganda.
**Formato di risposta**:
```json
{
    "text of propaganda": "propaganda type",
    "text of propaganda": "propaganda type",
    "text of propaganda": "propaganda type"
}
```
**Implementazione**:
```python
	def get_propaganda_spans(text: str):
	    span_metadata = []
	    for i in range(0, len(text), 5700):
	        prompt = f"""
	        Search in this text the 2-3 most relevant propaganda slices, with
			the associated propaganda type.
	        The result format have to be a JSON structured as follows, without
	        any other addition::
	        {{
	            "text of propaganda": "propaganda type",
	            "text of propaganda": "propaganda type",
	            "text of propaganda": "propaganda type"
	        }}
			
	        - The "text of propaganda" should be the slice of propaganda found
		      in the dataset, if not found return None
	        - The "propaganda type" is the classification of the text of
	          propaganda, if not found return None
	        - The "text of propaganda" have to be a sentence, starting from a
	          Capital letter to a point.
	        - The propaganda type must be one of these techniques:
	          "Name calling", "Repetition", "Slogans", etc.
			
	        Text: {text[i:i + 5700]}
	        """
			
	        chat_completion = client.chat.completions.create(
	            messages=[
	                {
	                    "role": "user",
	                    "content": prompt,
	                }
	            ],
	            model="llama-3.3-70b-versatile",
	        )
	
	        response = chat_completion.choices[0].message.content.strip()
	        response = response[response.find("{"):response.rfind("}") + 1]
	        try:
	            result = eval(response)
	        except:
	            result = {}
	
	        for key, value in result.items():
	            start_offset = i + text[i:i + 5700].find(key)
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
```