import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

# Pour afficher la matrice de confusion
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# 1. Charger le dataset
csv_file = 'tweets.csv'  # Assurez-vous que ce fichier est dans le même dossier
df = pd.read_csv(csv_file)

# 2. Créer la cible (label)
def is_sexist(evals_str):
    evals_list = ast.literal_eval(evals_str)
    nb_yes = sum(e == 'YES' for e in evals_list)
    nb_no = sum(e == 'NO' for e in evals_list)
    return 1 if nb_yes >= nb_no else 0
df['label'] = df['evaluation'].apply(is_sexist)

# 3. Garder uniquement les colonnes utiles
df = df[['tweet', 'label']]

# 4. Nettoyer les tweets
def clean_tweet(text):
    text = re.sub(r'https?://\S+', '', text)          # Enlever les URL
    text = re.sub(r'@\w+', '', text)                  # Enlever les mentions @
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # Enlever les caractères non-alphabétiques
    text = text.lower()                                # Mettre en minuscule
    text = re.sub(r'\s+', ' ', text).strip()          # Supprimer les espaces multiples
    return text

df['tweet_clean'] = df['tweet'].apply(clean_tweet)

# 5. Séparation train/test
X = df['tweet_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,       # 20% des données pour le test
    random_state=42,
    stratify=y           # pour garder la même proportion de classes
)

# 6. Créer le pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),               # Vectorisation TF-IDF
    ('clf', LogisticRegression(max_iter=1000))  # Modèle de régression logistique
])

# 7. Entraîner le modèle
print("Entraînement du modèle...")
pipeline.fit(X_train, y_train)
print("Entraînement terminé.")

# 8. Prédictions
print("Prédiction sur les données de test...")
y_pred = pipeline.predict(X_test)

# 9. Évaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Non Sexiste', 'Sexiste'])

print(f"Accuracy: {accuracy:.4f}\n")
print("Rapport de classification:\n", report)

# 10. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Affichage graphique de la matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non Sexiste', 'Sexiste'],
            yticklabels=['Non Sexiste', 'Sexiste'])
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.title("Matrice de Confusion")
plt.show()

# Optionnel : Sauvegarder le modèle entraîné
import joblib

model_filename = 'modele_detecteur_sexisme.pkl'
joblib.dump(pipeline, model_filename)
print(f"Modèle sauvegardé sous le nom {model_filename}")
