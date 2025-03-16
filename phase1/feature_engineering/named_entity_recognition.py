import pandas as pd
import spacy

# Load data
df = pd.read_csv('../datasets/transactions.csv')

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Extract entities from text data
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Apply NER on merchant and session_metadata
df['entities'] = df['merchant'].apply(extract_entities)

# Save NER output
df.to_csv('../datasets/ner_output.csv', index=False)
print("Named Entity Recognition complete!")