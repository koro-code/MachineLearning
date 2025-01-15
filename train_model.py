import pandas as pd
import re
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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
def clean_tweet(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
    test_size=0.2,
    random_state=42,
    stratify=y
)

# df_train contiendra les mêmes lignes que X_train (et X_test -> df_test).

# ------------------------------
# 6) PIPELINE
# ------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# ------------------------------
# 7) ENTRAINEMENT
# ------------------------------
print("Entraînement du modèle...")
pipeline.fit(X_train, y_train)
print("Entraînement terminé.")

# ------------------------------
# 8) PREDICTION SUR LE SET DE TEST
# ------------------------------
print("Prédiction sur les données de test...")
y_pred = pipeline.predict(X_test)

# ------------------------------
# 9) EVALUATION GLOBALE
# ------------------------------
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Non Sexiste', 'Sexiste'])

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

# =============================================================
#         PARTIE SPECIFIQUE: MATRICES DE CONFUSION
#         PAR ANNOTATEUR (en tenant compte de l'âge et du sexe)
# =============================================================

# ------------------------------
# 11) AJOUTER LES PREDICTIONS AU DF DE TEST
# ------------------------------
df_test = df_test.copy()
df_test['y_pred'] = y_pred  # la prédiction binaire (0/1) pour chaque tweet

# ------------------------------
# 12) PARSER LES COLONNES POUR AVOIR 1 LIGNE PAR ANNOTATEUR
# ------------------------------
df_test['gender_list'] = df_test['gender_annotators'].apply(ast.literal_eval)
df_test['age_list']    = df_test['age_annotators'].apply(ast.literal_eval)
df_test['eval_list']   = df_test['evaluation'].apply(ast.literal_eval)

def explode_annotators(row):
    new_rows = []
    for i in range(len(row['gender_list'])):  # 6 en général
        sex_i     = row['gender_list'][i]
        age_i     = row['age_list'][i]
        eval_i    = row['eval_list'][i]   # 'YES' ou 'NO'
        y_pred_i  = row['y_pred']         # (0 ou 1) => pareil pour les 6

        # Convertir 'YES'/'NO' en 1/0
        real_label_i = 1 if eval_i == 'YES' else 0

        new_rows.append({
            'tweet_clean':    row['tweet_clean'],
            'annotator_index': i,   # 0..5
            'gender':         sex_i,
            'age_range':      age_i,
            'eval_annotator': real_label_i,
            'pred_model':     y_pred_i
        })
    return new_rows

all_rows = []
for idx, row in df_test.iterrows():
    exploded = explode_annotators(row)
    all_rows.extend(exploded)

df_test_exploded = pd.DataFrame(all_rows)

# ------------------------------
# 13) Évaluation par annotateur (accuracy & classification_report)
# ------------------------------
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

for i in range(6):
    subset = df_test_exploded[df_test_exploded['annotator_index'] == i]
    true_labels = subset['eval_annotator']  # liste de 0/1
    pred_labels = subset['pred_model']      # liste de 0/1 (prédits par le modèle)

    # Accuracy et rapport
    acc_anno = accuracy_score(true_labels, pred_labels)
    report_anno = classification_report(
        true_labels, 
        pred_labels, 
        target_names=['Non Sexiste', 'Sexiste']
    )

    print(f"\n=== Annotateur {i+1} ===")
    print(f"Accuracy: {acc_anno:.4f}")
    print("Rapport de classification:\n", report_anno)

    # On peut aussi afficher la matrice de confusion
    cm_anno = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm_anno, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Non Sexiste','Sexiste'],
                yticklabels=['Non Sexiste','Sexiste'])
    plt.xlabel("Prédiction Modèle")
    plt.ylabel("Label Annotateur")
    plt.title(f"Matrice de Confusion - Annotateur {i+1}")
    plt.show()

# -----------------------------------------------------
# 14) Optionnel : MATRICE DE CONFUSION (TOUS ANNOTATEURS)
# -----------------------------------------------------
all_true = df_test_exploded['eval_annotator']
all_pred = df_test_exploded['pred_model']
cm_all = confusion_matrix(all_true, all_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm_all, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Non Sexiste','Sexiste'],
            yticklabels=['Non Sexiste','Sexiste'])
plt.xlabel("Prédiction Modèle")
plt.ylabel("Label Annotateur")
plt.title("Matrice de Confusion - Tous Annotateurs Confondus")
plt.show()
