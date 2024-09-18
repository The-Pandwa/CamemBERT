import streamlit as st
from transformers import CamembertModel, CamembertTokenizer
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Charger le modèle et le tokenizer CamemBERT
model = CamembertModel.from_pretrained("camembert/camembert-base-ccnet-4gb")
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base-ccnet-4gb")

# Fonction pour encoder une phrase
def phrases_encoder(phrase):
    inputs = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Fonction pour supprimer les stop words
def suppr_stopwords(text):
    stop_words = set(stopwords.words('french'))
    words = word_tokenize(text, language='french')
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Dictionnaire des persos
dict_perso = {
    'eca' : "Les Ecaflips sont des guerriers joueurs qui se fourrent dans les endroits où l'on peut gagner gros, et perdre beaucoup… Un Ecaflip bien dans sa peau parie sans arrêt, pour tout et pour rien. Mais attention, il prend le jeu très au sérieux et ira même jusqu'à risquer sa vie sur un jet de dés pour tenter de remporter la mise…",
    'eni' : "Les Eniripsas sont des guérisseurs qui soignent d'un simple Mot. Ils utilisent le pouvoir de la parole pour soulager les souffrances de leurs alliés, mais parfois aussi pour blesser leurs ennemis. Certains Eniripsas sont même devenus de véritables arpenteurs du verbe, des rôdeurs des langues oubliées.",
    'iop' : "Les Iops sont des guerriers fonceurs et sans reproche ! Une chose est sûre : les Iops savent faire parler les armes. D'ailleurs, se retrouver pris dans une bagarre au moins une fois par jour est pour eux un signe de bonne santé. Leur tempérament impétueux fait des Iops des paladins de l'extrême, capables du meilleur... comme du pire ! Iop est le Dieu de la bravoure et du courage, il est l'un des Douze Dieux Majeurs.Il accorde sa force à tous ceux qui ont la baston dans le sang ! Il méprise tellement les lâches que ses disciples, les Iops sont tous de véritables brutes qui foncent têtes baissées dans les combats… même ceux perdus d’avance ! Les Iops sont des guerriers fonceurs et sans reproche ! Une chose est sûre : les Iops savent faire parler les armes. D'ailleurs, se retrouver pris dans une bagarre au moins une fois par jour est pour eux un signe de bonne santé. Leur tempérament impétueux fait des Iops des paladins de l'extrême, capables du meilleur... comme du pire ! Ils sont réputés pour être des chevaliers courageux qui laissent parler leur poing. Le Chaos d'Ogrest n'a pas non plus changé leur attitude. Fidèles à leur Dieu, les Iops sont des guerriers fougueux et de redoutables protecteurs. Lorsqu'il s'agit de défendre leurs opinions, les Iops n'ont aucun mal à faire savoir à leurs ennemis à quel point ils peuvent être féroces. Prenez-en note!",
    'cra' : "Les Crâs sont des archers aussi fiers que précis ! Ils font des alliés précieux contre les adeptes de la mêlée franche. Restant à distance, décochant leurs traits empennés dans le moindre orifice laissé sans surveillance, ils ne laissent aucun répit à leurs adversaires ! Les Crâs sont des habitants du Monde des Douze vénérant la déesse Crâ. C'est une espèce spécialisée dans le combat à l'arc. Les Crâs sont des archers aussi fiers que précis ! Ils font des alliés précieux contre les adeptes de la mêlée franche. Restant à distance, décochant leurs traits empennés dans le moindre orifice laissé sans surveillance, ils ne laissent aucun répit à leurs adversaires !",
    'feca' : "Les Fécas sont de loyaux protecteurs toujours sur la défensive. Ils sont appréciés dans les groupes d'aventuriers pour leurs armures élémentaires et leur capacité à encaisser les coups durs. Ils sont également maîtres dans l’art des signes magiques : quand il va y avoir du grabuge, les Fécas sortent leurs glyphes !",
    'sacri' : "Les Sacrieurs sont des berserkers qui décuplent leurs forces dès qu'ils sont frappés ! N'ayant pas peur de recevoir des coups, ni de s'exposer aux blessures, ils seront souvent en première ligne, prêts à verser le premier sang ! Le Sacrieur est vraiment le compagnon idéal pour vos longues soirées guerrières…",
    'sadi' : "Les Sadidas sont des invocateurs qui empoisonnent la vie de leurs ennemis ! Apprivoiser les Ronces pour en faire des armes terrifiantes, confectionner des poupées de guerre et de soins, voilà qui satisfait tout disciple Sadida digne de ce nom.",
    'osa' : "Les Osamodas sont des dompteurs nés ! Ils ont le pouvoir d'invoquer des créatures et sont de remarquables dresseurs. Une rumeur prétend qu'ils taillent leurs vêtements dans la peau de leurs ennemis, mais allez donc leur demander ce qu'il en est… Si vous êtes de son côté, un Osamodas sera aux petits soins pour vous. Dans le cas contraire, peut-être terminerez-vous votre vie sous la forme d'une botte ou d'un bonnet fourré.",
    'enu' : "Les Enutrofs sont des chasseurs de trésor avides de kamas, qui malgré leur grand âge courent comme des dragodindes à la vue d'un coffre bien rempli. Ils sont experts dans l’art de ralentir leurs ennemis : ils peuvent ainsi les harceler avant de les assommer à grands coups de pelle le moment venu !",
    'sram' : "Les Srams sont des assassins qui aiment les bourses, rebondies de préférence. Trousser les pans d'une tunique, tâter le fond d'une poche, faire preuve de doigté, palper enfin des bijoux tant convoités avant de poser un piège ou d'asséner un coup mortel, voilà la vie d'un disciple de Sram !",
    'xel' : "Les Xélors sont des mages qui maîtrisent le temps et toutes les mécaniques qui donnent l'heure : carillons, horloges, et pendules leur obéissent au doigt et à l'œil. Les Xélors jouent donc avec le temps pour ralentir un ennemi ou se téléporter où bon leur semble.",
    'panda' : "Les Pandawas sont des guerriers adeptes des arts martiaux qui savent faire des folies de leurs corps ! Ils peuvent même en faire avec le corps des autres… Le Pandawa sait comment soulever les foules, il porte ses alliés sur ses épaules pour mieux les protéger. Quant à ses ennemis, il les enverra valser dans le décor, avant de fêter sa victoire avec une bonne rasade de lait de bambou !",
    'roub' : "Membres d’un clan créé à l’origine par Raval et la famille Smisse, les Roublards ont finalement prêté allégeance au dieu Dralbour, qui n’est autre que Sram. Maîtres de l’entourloupe, du coup fourré, des bombes à retardement et des pistolets… les Roublards ne sont à l’aise qu’en terrain miné.",
    'zozo' : "Les Zobals portent des masques magiques qui leur permettent de changer de tête comme de chemise. Tour à tour collants comme de la glu, enragés comme des psychopathes ou partisans de la retraite stratégique, ils s’adaptent à la situation, mais gare aux troubles de la personnalité ! La légende dit que ces êtres imprévisibles seraient bénis par Sadida lui-même.",
    'steam' : "Les Steamers ont plus d’une tourelle dans leur boîte à outils. En fins tacticiens, ils utilisent la technomagie pour prendre l’avantage sur le terrain. Ces marinventeurs aux scaphandres rutilants vouent un culte au Grand Oktapodas, le protecteur des océans. Les Steamers sont aussi à l'aise sous les eaux que sur le champ de bataille. Harponneurs, gardiens, maîtres des mouvements sur le terrain : ces explorateurs des profondeurs sont changeants comme l'océan. Leurs tourelles technomagiques bombardent tous ceux qui passent à leur portée, aussi bien les amis que les ennemis. Prenez garde à ne pas finir noyé sous un déluge de technomagie ! Après le Chaos d'Ogrest, les Steamers restent patients et prudents, et attendent le meilleur moment pour agir... Au travers des siècles, ils ne cessent de s'améliorer, tant au niveau intellectuel que physique. Autrefois athées et belliqueux, ils sont désormais les ambassadeurs d'un peuple qui prône la paix. Leur mission : trouver des mines de Stasili et ainsi préparer le retour des leurs...",
    'elio' : "Apparus par accident, les Eliotropes sont des reflets de leur créateur, le Roi-Dieu. Ils se déplacent à la vitesse de l'éclair, disparaissant en un clin d'œil pour réapparaître plus loin. Tout comme les Eliatropes, ils connaissent les secrets du Wakfu.",
    'hupper' : "Les Huppermages sont capables de combiner le Feu, l’Air, l’Eau et la Terre pour lancer des sortilèges aux multiples effets. Maîtres des runes élémentaires, ils vénèrent la Balance Krosmique, une force mystérieuse qui tend à maintenir l’équilibre dans l'univers.",
    'ougi' : "Quand on est un barbare avec un caractère de chienchien, il faut savoir montrer les crocs pour se faire respecter. Le regard fier, le poil brillant, la truffe humide : c'est ainsi que l'Ouginak traverse les épreuves et triomphe de l'adversité. Traquer ses proies sans relâche, laisser éclater sa rage pour écraser ses ennemis, les voir mourir devant soi et entendre les miaulements terrorisés de leurs chachas… Voilà une vie digne d'être vécue !",
    'forge' : "Les Forgelances sont des lanciers qui ne croient qu'en eux-mêmes ! Tirant leurs pouvoirs de la Lance Originelle, ces combattants cherchent à se forger une renommée que la mort elle-même ne saurait effacer. Féru d'arts, de récits légendaires et de joutes épiques, le Forgelance se laissera convaincre de brandir son arme à vos côtés par des promesses de gloire et de postérité. Véritables fers de lance des combats, ils sont les premiers au contact pour percer les défenses adverses."
}

# Import fichier CSV
list_perso = pd.read_csv(r'Perso.csv')
st.write(list_perso)

# Titre du streamlit
st.title('Comparateur et similitude entre idées')

# Explication du fonctionnement
st.header('Début des tests !')

# Zone d'expression libre
test_phrase = st.text_area("Tape le texte","")

# Supprimer les stop words de la phrase de test
cleaned_test_phrase = suppr_stopwords(test_phrase)

# Initialisation du dictionnaire des similarités
similarities_dict = {}

# Encoder la phrase de test
encoded_test = phrases_encoder(cleaned_test_phrase)

# Boucle pour encoder les phrases et les ajouter dans similarities_dict
for clef, valeur in dict_perso.items():
    cleaned_value = suppr_stopwords(valeur)
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
