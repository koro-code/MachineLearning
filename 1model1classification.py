import pandas as pd
import re
import ast

# --- S'il vous manque ce package, faire pip install imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# 1) Charger le dataset
# ------------------------------------------------
csv_file = 'tweets.csv'
df = pd.read_csv(csv_file)

# ------------------------------------------------
# 2) Déterminer si un tweet est jugé sexiste
#    (majorité de 'YES')
# ------------------------------------------------
def is_majority_yes(eval_str):
    evals = ast.literal_eval(eval_str)
    nb_yes = sum(e == 'YES' for e in evals)
    nb_no = sum(e == 'NO' for e in evals)
    return nb_yes >= nb_no

df['is_sexist'] = df['evaluation'].apply(is_majority_yes)

# ------------------------------------------------
# 3) Majorité d'intention (DIRECT/REPORTED/JUDGEMENTAL)
# ------------------------------------------------
def majority_intention(types_str):
    """
    Parmi la liste 'evaluation_type', renvoie la catégorie majoritaire 
    parmi DIRECT, REPORTED, JUDGEMENTAL. Ignore les '-'. 
    Retourne None si aucune majorité ne se dégage.
    """
    types_list = ast.literal_eval(types_str)
    counts = {'DIRECT': 0, 'REPORTED': 0, 'JUDGEMENTAL': 0}
    for t in types_list:
        if t in counts:
            counts[t] += 1
    
    max_cat = max(counts, key=counts.get)
    max_val = counts[max_cat]
    
    if max_val == 0:
        return None  # Pas d'étiquette claire
    return max_cat

# ------------------------------------------------
# 4) Créer la colonne 'intention_label' avec 4 classes
#    - DIRECT / REPORTED / JUDGEMENTAL (pour les sexistes)
#    - NON_SEXIST (pour les non-sexistes)
# ------------------------------------------------
def get_intention_label(row):
    if row['is_sexist']:
        # On récupère la majorité d'intention pour les tweets sexistes
        label = majority_intention(row['evaluation_type'])
        return label
    else:
        # Pour les tweets non sexistes
        return "NON_SEXIST"

df['intention_label'] = df.apply(get_intention_label, axis=1)

# On retire les tweets sexistes qui n'ont pas d'intention claire (None)
df = df[df['intention_label'].notnull()].copy()

# ------------------------------------------------
# 5) Nettoyage du texte
# ------------------------------------------------
def clean_tweet(text):
    text = re.sub(r'https?://\S+', '', text)   # enlever les URL
    text = re.sub(r'@\w+', '', text)           # enlever les @mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # enlever la ponctuation et les chiffres
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['tweet_clean'] = df['tweet'].apply(clean_tweet)

# ------------------------------------------------
# 6) Séparation train / test
# ------------------------------------------------
X = df['tweet_clean']
y = df['intention_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # pour mieux équilibrer dans train/test
)

# ------------------------------------------------
# 7) Pipeline avec sous-échantillonnage et LogisticRegression
# ------------------------------------------------
#  - TfidfVectorizer pour transformer les textes
#  - RandomUnderSampler pour équilibrer les classes en se basant sur la plus petite
#  - LogisticRegression avec class_weight='balanced'

pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('under', RandomUnderSampler(random_state=42)),  # <<--- sous-échantillonnage
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

print("Entraînement du modèle (4 classes : DIRECT / REPORTED / JUDGEMENTAL / NON_SEXIST)...")
pipeline.fit(X_train, y_train)
print("Entraînement terminé.")

# ------------------------------------------------
# 8) Évaluation
# ------------------------------------------------
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

print("Rapport de classification (PREC, RECALL, F1) :")
print(classification_report(y_test, y_pred, digits=4))

classes = ["DIRECT", "REPORTED", "JUDGEMENTAL", "NON_SEXIST"]
cm = confusion_matrix(y_test, y_pred, labels=classes)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes,
            yticklabels=classes)
plt.title("Matrice de confusion (4 classes)")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()
