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
def analyse_description(description):
    phrase = tokenizer(description, return_tensors = 'pt', truncation = True, padding = True)
    with torch.no_grad():
        token = model(**phrase)
    return token.last_hidden_state.cpu().numpy()

# Fonction comparaison description et saisie
def cosine_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2.T)

# Dataframe
df_personnage = pd.read_csv("Perso.csv")
st.dataframe(df_personnage)

# Bouton Rôle
roles_perso = st.toggle('Rôle du personnage en combat :')
if roles_perso :
    col1, col2, col3 = st.columns(3)
    with col1 :
        role_1 = st.selectbox("Rôle principal :",
                     ("Non défini", "Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )
    with col2 :
        role_2 = st.selectbox("Rôle secondaire :",
                     ("Non défini", "Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )
    with col3 :
        role_3 = st.selectbox("Rôle tertiaire :",
                     ("Non défini", "Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )

# Saisie utilisateur et comparaison avec les descriptions
phrase_utilisateur = st.text_area("Quel héros voulez vous être ?","")
if phrase_utilisateur == "" :
    st.warning("Saisissez une phrase ", icon = '⚠️')
else :
    analyse_user = analyse_description(phrase_utilisateur)
    df_personnage['Encodage'] = df_personnage['Description'].apply(analyse_description)
    df_personnage['Similarity'] = df_personnage["Encodage"].apply(lambda x: cosine_similarity(analyse_user, x).item())
    top_matches = df_personnage.sort_values(by='Similarity', ascending=False).head(3)


st.dataframe(df_personnage)