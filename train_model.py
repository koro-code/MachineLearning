import pandas as pd # pour manipuler les données sous forme de DataFrame.
import re # pour les expressions régulières.
import ast # pour convertir une chaîne de caractères en liste.

from sklearn.feature_extraction.text import TfidfVectorizer # pour la vectorisation des textes.
from sklearn.linear_model import LogisticRegression # le model
from sklearn.pipeline import Pipeline # pour créer un pipeline de traitement.
from sklearn.model_selection import train_test_split # pour séparer les données en train/test.
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # pour mesurer les performances du modèle (précision, rappel, f1-score, etc.).

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1) CHARGER LE DATASET
# ------------------------------
csv_file = 'tweets.csv'
df = pd.read_csv(csv_file)

# ------------------------------
# 2) CREER LA CIBLE (LABEL) GLOBAL
#    -> majoritaire "YES"/"NO"
# ------------------------------
def is_sexist(evals_str):
    # ex: "['YES','NO','YES']"
    evals_list = ast.literal_eval(evals_str)
    nb_yes = sum(e == 'YES' for e in evals_list)
    nb_no  = sum(e == 'NO'  for e in evals_list)
    return 1 if nb_yes >= nb_no else 0

df['label'] = df['evaluation'].apply(is_sexist)

# ------------------------------
# 3) GARDER UNIQUEMENT COLONNES UTILES POUR L'ENTRAINEMENT GLOBAL
# ------------------------------
# -> On garde 'tweet', 'label', mais ON CONSERVE aussi
#    'gender_annotators', 'age_annotators', 'evaluation'
#    pour plus tard (extraction par annotateur au test).
df = df[['tweet', 
         'label',
         'gender_annotators',
         'age_annotators',
         'evaluation']]

# ------------------------------
# 4) NETTOYAGE DES TWEETS
# ------------------------------
# préparer le texte pour le traitement du langage (enlever URLs, mentions, caractères spéciaux)
def clean_tweet(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# on crée une nouvelle colonne tweet_clean qui contient la version nettoyée du tweet.
df['tweet_clean'] = df['tweet'].apply(clean_tweet)

# ------------------------------
# 5) SEPARATION TRAIN / TEST
# ------------------------------
X = df['tweet_clean']
y = df['label']

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X,
    y,
    df,  # On split également le DataFrame complet 
    test_size=0.2, # 20% des données dans le set de test
    random_state=42,
    stratify=y
)

# df_train contiendra les mêmes lignes que X_train (et X_test -> df_test).

# ------------------------------
# 6) PIPELINE
# ------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()), # Le TF-IDF évalue l'importance d’un mot dans un document en tenant compte de sa fréquence dans le corpus.
    ('clf', LogisticRegression(max_iter=1000)) # on crée un classifieur de type Régression Logistique
])

# Ainsi, quand on fera .fit(X_train, y_train), il appliquera TfidfVectorizer sur X_train, puis entraînera la LogisticRegression sur le résultat

# ------------------------------
# 7) ENTRAINEMENT
# ------------------------------
print("Entraînement du modèle...")
pipeline.fit(X_train, y_train) # On entraîne le modèle sur les données d'entraînement
print("Entraînement terminé.")

# ------------------------------
# 8) PREDICTION SUR LE SET DE TEST
# ------------------------------
print("Prédiction sur les données de test...")
y_pred = pipeline.predict(X_test) # Le pipeline applique la même transformation TfidfVectorizer sur X_test (en utilisant le vocabulaire appris).

# ------------------------------
# 9) EVALUATION GLOBALE
# ------------------------------
acc = accuracy_score(y_test, y_pred) # on calcule la proportion de prédictions correctes.
report = classification_report(y_test, y_pred, target_names=['Non Sexiste', 'Sexiste']) # montre la précision (precision), rappel (recall), f1-score, et support (nombre d’exemples) pour chaque classe

print(f"Accuracy globale: {acc:.4f}")
print("Rapport de classification:\n", report)

# ------------------------------
# 10) MATRICE DE CONFUSION GLOBALE
# ------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non Sexiste', 'Sexiste'],
            yticklabels=['Non Sexiste', 'Sexiste'])
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.title("Matrice de Confusion Globale (tweet entier)")
plt.show()