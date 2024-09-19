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
def phrases_encoder(phrase):
    inputs = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Lecture du dataframe
# Test
dict_perso = {'ougi' : "Quand on est un barbare avec un caractère de chienchien, il faut savoir montrer les crocs pour se faire respecter. Le regard fier, le poil brillant, la truffe humide : c'est ainsi que l'Ouginak traverse les épreuves et triomphe de l'adversité. Traquer ses proies sans relâche, laisser éclater sa rage pour écraser ses ennemis, les voir mourir devant soi et entendre les miaulements terrorisés de leurs chachas… Voilà une vie digne d'être vécue !",
            'forge' : "Les Forgelances sont des lanciers qui ne croient qu'en eux-mêmes ! Tirant leurs pouvoirs de la Lance Originelle, ces combattants cherchent à se forger une renommée que la mort elle-même ne saurait effacer. Féru d'arts, de récits légendaires et de joutes épiques, le Forgelance se laissera convaincre de brandir son arme à vos côtés par des promesses de gloire et de postérité. Véritables fers de lance des combats, ils sont les premiers au contact pour percer les défenses adverses.",
            'farce' : "Les farces sont des lanciers qui ne croient qu'en eux-mêmes ! Tirant leurs pouvoirs de la Lance Originelle, ces combattants cherchent à se forger une renommée que la mort elle-même ne saurait effacer. Féru d'arts, de récits légendaires et de joutes épiques, le Forgelance se laissera convaincre de brandir son arme à vos côtés par des promesses de gloire et de postérité. Véritables fers de lance des combats, ils sont les premiers au contact pour percer les défenses adverses."
            }

# Titre du streamlit
st.title('Comparateur et similitude entre idées')

# Explication du fonctionnement
st.header('Début des tests !')

# Zone d'expression libre
test_phrase = st.text_area("Tape le texte","")

# Supprimer les stop words de la phrase de test
# cleaned_test_phrase = suppr_stopwords(test_phrase)
# Test

# Initialisation du dictionnaire des similarités
similarities_dict = {}

# Encoder la phrase de test
encoded_test = phrases_encoder(test_phrase)

# Boucle pour encoder les phrases et les ajouter dans similarities_dict
for clef, valeur in dict_perso.items():
    encoded_value = phrases_encoder(valeur)  
    enc_clef = "enc_" + clef
    similarity = F.cosine_similarity(encoded_test, encoded_value).item() 
    similarities_dict[enc_clef] = similarity
    st.write(f"Similarité avec '{clef}': {similarity:.4f}")

# Trouver la phrase avec la similarité maximale
max_phrase = max(similarities_dict, key=similarities_dict.get)
max_similarity = similarities_dict[max_phrase]

# Afficher la phrase correspondante avec la similarité maximale
st.write(f"Similarité avec phrase '{max_phrase}': {max_similarity:.4f}")

# Afficher le top 3 des meilleures correspondances
top_3 = dict(sorted(similarities_dict.items(), key=lambda item: item[1], reverse=True)[2:3])
top_2 = dict(sorted(similarities_dict.items(), key=lambda item: item[1], reverse=True)[1:2])
top_1 = dict(sorted(similarities_dict.items(), key=lambda item: item[1], reverse=True)[0:1])

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
                     ("Non défini", "Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )
    with col2 :
        st.selectbox("Rôle secondaire :",
                     ("Non défini", "Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )
    with col3 :
        st.selectbox("Rôle tertiaire :",
                     ("Non défini", "Dégâts", "Amélioration", "Soins", "Entrave", "Placement", "Protection", "Tank", "Invocation"),
                    )
