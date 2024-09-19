import pandas as pd
import streamlit as st
from transformers import CamembertModel, CamembertTokenizer
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize

# Charger le modèle et le tokenizer CamemBERT
model = CamembertModel.from_pretrained("camembert/camembert-base-ccnet-4gb")
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base-ccnet-4gb")

# Fonction pour encoder une phrase

# Titre du streamlit
st.title('Comparateur et similitude entre idées')

# Explication du fonctionnement
st.header('Début des tests !')

# Dataframe
df = pd.read_csv("Perso.csv")
st.dataframe(df)

# Zone d'expression libre
test_phrase = st.text_area("Tape le texte","")
if test_phrase == None :
    return "Wait"
else :
    return "Ok"

# Initialisation du dictionnaire des similarités

# Encoder la phrase de test

# Boucle pour encoder les phrases

# Trouver la phrase avec la similarité maximale

# Afficher la phrase correspondante avec la similarité maximale

# Afficher le top 3 des meilleures correspondances
top_1 = "First"
top_2 = "Lose"
top_3 = "Fail"

# Organiser les résultats en colonne
col1, col2, col3 = st.columns(3)
with col1 :
    st.header('3ème')
    st.write(top_3)
with col2 :
    st.header('1er')
    st.write(top_1)
with col3 :
    st.header('2nd')
    st.write(top_2)

# Bouton Rôle
role_1 = st.toggle('Rôle du personnage en combat :')
if role_1 :
    col1, col2, col3 = st.columns(3)
    with col1 :
        st.selectbox("Rôle principal :",
                     ("Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )
    with col2 :
        st.selectbox("Rôle secondaire :",
                     ("Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )
    with col3 :
        st.selectbox("Rôle tertiaire :",
                     ("Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )