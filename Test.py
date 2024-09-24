import pandas as pd
import streamlit as st
from transformers import CamembertModel, CamembertTokenizer
import torch
import torch.nn.functional as F
import numpy as np

# Charger le modèle et le tokenizer CamemBERT
model = CamembertModel.from_pretrained("almanach/camembert-large")
tokenizer = CamembertTokenizer.from_pretrained("almanach/camembert-large")

# Fonction pour encoder une phrase
def encode_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Utiliser le premier token (CLS token) comme représentation de la phrase
    sentence_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return sentence_embedding

# Fonction pour calculer la similarité cosinus entre deux vecteurs
def cosine_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2.T)

# Titre du streamlit
st.title('Comparateur et similitude entre idées')

# Explication du fonctionnement
st.header('Début des tests !')

# Charger le dataframe avec les descriptions
df = pd.read_csv("Perso.csv")
st.dataframe(df)

# Zone d'expression libre
test_phrase = st.text_area("Quel héros voulez-vous être ?", "")

if test_phrase == "":
    st.write("Veuillez entrer une phrase.")
else:
    # Encoder la phrase saisie par l'utilisateur
    test_embedding = encode_sentence(test_phrase)

    # Encoder toutes les descriptions du dataframe
    df['embedding'] = df['Description'].apply(encode_sentence)

    # Calculer la similarité cosinus entre la phrase saisie et toutes les descriptions
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(test_embedding, x).item())

    # Trier les résultats par similarité décroissante
    top_matches = df.sort_values(by='similarity', ascending=False).head(3)

    # Afficher les 3 meilleures correspondances
    st.header('Top 3 des meilleures correspondances')
    for idx, row in top_matches.iterrows():
        st.write(f"{idx+1}. {row['Description']} (Similarité : {row['similarity']:.2f})")
