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
test_phrase = "Un animal dort."

# Encoder les phrases
enc1 = encode_sentence(phrase1)
enc2 = encode_sentence(phrase2)
enc_test = encode_sentence(test_phrase)

# Calculer la similarité (cosine similarity)
similarity1 = torch.nn.functional.cosine_similarity(enc_test, enc1)
similarity2 = torch.nn.functional.cosine_similarity(enc_test, enc2)

print(f"Similarité phrase 1: {similarity1.item()}")
print(f"Similarité phrase 2: {similarity2.item()}")

st.title('Titre')
st.header('En tête')

st.text_area("Tape le texte","")
