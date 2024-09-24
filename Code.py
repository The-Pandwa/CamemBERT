import pandas as pd
import streamlit as st
from transformers import CamembertModel, CamembertTokenizer
import torch
import torch.nn.functional as F
import numpy

# Charger le modèle et le tokenizer CamemBERT
model = CamembertModel.from_pretrained("almanach/camembert-large")
tokenizer = CamembertTokenizer.from_pretrained("almanach/camembert-large")

# Fonction pour encoder une phrase

# Titre du streamlit
st.title('Comparateur et similitude entre idées')

# Explication du fonctionnement
st.header('Début des tests !')

# Dataframe
df = pd.read_csv("Perso.csv")
st.dataframe(df)

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

# Zone d'expression libre
test_phrase = st.text_area("Quel héros voulez vous être ?","")
if test_phrase == "" :
    st.write("Wait")
else :
    st.write("OK")

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

# Fonction NLP camembert
def analyse_description(description):
    phrase = tokenizer(description, return_tensors = 'pt', truncation = True, padding = True)
    with torch.no_grad():
        token = model(**phrase)
    return token.last_hidden_state.cpu().numpy()
    
df['test'] = df['Description'].apply(analyse_description)
st.dataframe(df)
