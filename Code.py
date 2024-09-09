import streamlit as st
from transformers import CamembertModel, CamembertTokenizer
import torch

# Charger le modèle et le tokenizer CamemBERT
model = CamembertModel.from_pretrained("camembert-base")
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Fonction pour encoder une phrase
def encode_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Tes phrases
phrase1 = "Le chat mange une souris."
phrase2 = "Le chien dort dans le jardin."

# Titre du streamlit
st.title('Comparateur et similitude entre idées')

# Explication du fonctionnement
st.header('Début des tests !')

# Zone d'expression libre
test_phrase = st.text_area("Tape le texte","")

# Encoder les phrases
enc1 = encode_sentence(phrase1)
enc2 = encode_sentence(phrase2)
enc_test = encode_sentence(test_phrase)

# Calculer la similarité (cosine similarity)
similarity1 = torch.nn.functional.cosine_similarity(enc_test, enc1)
similarity2 = torch.nn.functional.cosine_similarity(enc_test, enc2)

# Résultat de la comparaison
st.write(f"Similarité avec phrase 1 ('{phrase1}'): {similarity1.item():.4f}")
st.write(f"Similarité avec phrase 2 ('{phrase2}'): {similarity2.item():.4f}")
