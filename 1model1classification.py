import pandas as pd
import re
import ast

# --- S'il vous manque ce package, faire pip install imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

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
# 2) Filtrer les tweets réellement sexistes
#    (majoritairement "YES")
# ------------------------------------------------
def is_majority_yes(eval_str):
    evals = ast.literal_eval(eval_str)
    nb_yes = sum(e == 'YES' for e in evals)
    nb_no = sum(e == 'NO' for e in evals)
    return nb_yes >= nb_no

df['is_sexist'] = df['evaluation'].apply(is_majority_yes)
df_sexist_only = df[df['is_sexist'] == True].copy()

# ------------------------------------------------
# 3) Extraire la "majorité" d'intention 
#    (DIRECT / REPORTED / JUDGEMENTAL)
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
    
    # si 0 => pas de DIRECT/REPORTED/JUDGEMENTAL dans ce tweet
    if max_val == 0:
        return None
    return max_cat

df_sexist_only['intention_label'] = df_sexist_only['evaluation_type'].apply(majority_intention)

# On retire les tweets sans intention claire
df_sexist_only = df_sexist_only[df_sexist_only['intention_label'].notnull()].copy()

# ------------------------------------------------
# 4) Nettoyage du texte
# ------------------------------------------------
def clean_tweet(text):
    text = re.sub(r'https?://\S+', '', text)   # enlever les URL
    text = re.sub(r'@\w+', '', text)           # enlever les @mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # enlever ponctuation/chiffres
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_sexist_only['tweet_clean'] = df_sexist_only['tweet'].apply(clean_tweet)

# ------------------------------------------------
# 5) Séparation train / test
# ------------------------------------------------
X = df_sexist_only['tweet_clean']
y = df_sexist_only['intention_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # pour mieux équilibrer dans train/test
)

# ------------------------------------------------
# 6) Pipeline avec oversampling et LogisticRegression
# ------------------------------------------------
#  - TfidfVectorizer pour transformer les textes
#  - RandomOverSampler pour rééquilibrer les classes
#  - LogisticRegression avec class_weight='balanced'

pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('over', RandomOverSampler(random_state=42)),  # <<--- suréchantillonnage
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

print("Entraînement du modèle (3 classes : DIRECT / REPORTED / JUDGEMENTAL)...")
pipeline.fit(X_train, y_train)
print("Entraînement terminé.")

# ------------------------------------------------
# 7) Évaluation
# ------------------------------------------------
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

print("Rapport de classification (PREC, RECALL, F1) :")
print(classification_report(y_test, y_pred, digits=4))

classes = ["DIRECT", "REPORTED", "JUDGEMENTAL"]
cm = confusion_matrix(y_test, y_pred, labels=classes)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes,
            yticklabels=classes)
plt.title("Matrice de confusion (3 classes sexistes)")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()
