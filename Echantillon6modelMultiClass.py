import pandas as pd
import re
import ast
import numpy as np

# Pour la division train/test
from sklearn.model_selection import train_test_split, GridSearchCV

# Pour le pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Utilitaires pour l'équilibrage
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

# Pour les métriques
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------
# 0) Lecture du CSV et nettoyage de base
# ----------------------------------------------
csv_file = "tweets.csv"
df = pd.read_csv(csv_file)

def clean_tweet(text):
    # Supprime les URLs
    text = re.sub(r'https?://\S+', '', text)
    # Supprime les mentions @
    text = re.sub(r'@\w+', '', text)
    # Conserve seulement caractères alphabétiques et espaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Passe en minuscules
    text = text.lower()
    # Supprime espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["tweet_clean"] = df["tweet"].apply(clean_tweet)

# ----------------------------------------------
# 1) Conversion en listes Python (colonnes)
# ----------------------------------------------
df['gender_list']     = df['gender_annotators'].apply(ast.literal_eval)
df['age_list']        = df['age_annotators'].apply(ast.literal_eval)
df['eval_list']       = df['evaluation'].apply(ast.literal_eval)
df['eval_type_list']  = df['evaluation_type'].apply(ast.literal_eval)

# On crée un ID unique pour chaque tweet
df['tweet_id'] = df.index

# ----------------------------------------------
# 2) Explosion en 6 lignes (une par annotateur)
# ----------------------------------------------
def explode_annotators(row):
    """
    Pour chaque tweet (row), on crée 6 lignes, une par annotateur.
    On va construire un label_4class par annotateur:
      - NON_SEXISTE si eval_annotator == 0 ou type_annotator == '-'
      - DIRECT/REPORTED/JUDGEMENTAL sinon
    """
    new_rows = []
    for i in range(len(row['gender_list'])):
        sex_i  = row['gender_list'][i]
        age_i  = row['age_list'][i]
        eval_i = row['eval_list'][i]          # 'YES' ou 'NO'
        type_i = row['eval_type_list'][i]     # 'DIRECT','REPORTED','JUDGEMENTAL' ou '-'

        # Construction du label en 4 classes :
        #  - NON_SEXISTE si NO ou type = '-'
        #  - autrement, DIRECT / REPORTED / JUDGEMENTAL
        if eval_i == 'NO' or type_i == '-':
            label_4class = 'NON_SEXISTE'
        else:
            label_4class = type_i

        new_rows.append({
            'tweet_id':            row['tweet_id'],
            'annotator_index':     i,
            'gender':              sex_i,
            'age_range':           age_i,
            'tweet_clean':         row['tweet_clean'],
            # eval_annotator et eval_type_annotator (si on veut toujours les conserver)
            'eval_annotator':      1 if eval_i == 'YES' else 0,
            'eval_type_annotator': type_i,
            # Notre nouveau label 4 classes
            'label_4class':        label_4class
        })
    return new_rows

all_rows = []
for _, row in df.iterrows():
    all_rows.extend(explode_annotators(row))

df_exploded = pd.DataFrame(all_rows)

# ----------------------------------------------
# 3) Split train / test (au niveau du tweet_id)
# ----------------------------------------------
unique_ids = df_exploded['tweet_id'].unique()
train_ids, test_ids = train_test_split(
    unique_ids,
    test_size=0.2,
    random_state=42
)

df_exploded_train = df_exploded[df_exploded['tweet_id'].isin(train_ids)].copy()
df_exploded_test  = df_exploded[df_exploded['tweet_id'].isin(test_ids)].copy()

# ----------------------------------------------
# 4) Entraînement Multi-Classes (4 classes) par annotateur
#    (NON_SEXISTE / DIRECT / REPORTED / JUDGEMENTAL)
# ----------------------------------------------
# On élargit la grille de recherche et on va
# utiliser un Pipeline qui inclut un sur-échantillonnage
param_grid_4class = {
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
    'tfidf__max_df': [0.9, 1.0],
    'clf__C': [0.01, 0.1, 1, 10, 100]
}

models_per_annotator_4class = {}

for i in range(6):
    # Sélection des lignes d'entraînement pour l'annotateur i
    subset_train_i = df_exploded_train[df_exploded_train['annotator_index'] == i]
    if len(subset_train_i) < 5:
        # Trop peu d'exemples pour cet annotateur
        continue

    X_i_train = subset_train_i['tweet_clean']
    y_i_train = subset_train_i['label_4class']

    pipeline_i = ImbPipeline([
        ('tfidf', TfidfVectorizer(
            # Si vos tweets sont en français, passez stop_words='french' 
            sublinear_tf=True,
            stop_words='english',
            min_df=2
        )),
        # Sur-échantillonnage aléatoire des classes minoritaires
        ('ros', RandomOverSampler(random_state=42)),
        ('clf', LogisticRegression(
            max_iter=2000,
            solver='lbfgs',
            random_state=42
        ))
    ])

    gs = GridSearchCV(
        pipeline_i,
        param_grid=param_grid_4class,
        # Utiliser un scoring adapté aux données déséquilibrées
        scoring='f1_macro',
        cv=3,     # ou plus si vous avez assez de données
        n_jobs=-1
    )
    gs.fit(X_i_train, y_i_train)

    models_per_annotator_4class[i] = gs.best_estimator_

# ----------------------------------------------
# 5) Calcul du "vrai" label agrégé en 4 classes
#    pour chaque tweet du test
# ----------------------------------------------
def majority_4classes(evals_list, types_list):
    """
    Retourne la classe agrégée parmi 4 classes:
    - 'NON_SEXISTE'
    - 'DIRECT'
    - 'REPORTED'
    - 'JUDGEMENTAL'
    On considère que:
      * si nb(NO) >= nb(YES), => 'NON_SEXISTE'
      * sinon => majorité parmi DIRECT/REPORTED/JUDGEMENTAL
    """
    nb_yes = sum(e == 'YES' for e in evals_list)
    nb_no  = sum(e == 'NO' for e in evals_list)
    if nb_no >= nb_yes:
        return 'NON_SEXISTE'
    else:
        selected_types = []
        for e, t in zip(evals_list, types_list):
            if e == 'YES' and t in ['DIRECT','REPORTED','JUDGEMENTAL']:
                selected_types.append(t)
        if len(selected_types) == 0:
            return 'NON_SEXISTE'
        freq = pd.Series(selected_types).value_counts()
        return freq.idxmax()

df_test_only = df[df['tweet_id'].isin(test_ids)].copy()
df_test_only['true_4class'] = '-'

for idx in df_test_only.index:
    evals  = ast.literal_eval(df_test_only.loc[idx, 'evaluation'])
    types  = ast.literal_eval(df_test_only.loc[idx, 'evaluation_type'])
    df_test_only.loc[idx, 'true_4class'] = majority_4classes(evals, types)

# ----------------------------------------------------------------
# Ré-équilibrage du jeu de test (optionnel)
# ----------------------------------------------------------------
# Vous pouvez le faire si vous tenez absolument à tester
# sur un jeu équilibré en sortie, mais ce n'est pas
# obligatoire. Laisser le test tel quel est souvent préférable.
class_counts = df_test_only['true_4class'].value_counts()
min_count = class_counts.min()

df_test_only = (
    df_test_only
    .groupby('true_4class', group_keys=False)
    .apply(lambda x: x.sample(n=min_count, random_state=42))
    .reset_index(drop=True)
)

# ----------------------------------------------
# 6) Prédiction finale agrégée (4 classes)
# ----------------------------------------------
all_4classes = ['NON_SEXISTE','DIRECT','REPORTED','JUDGEMENTAL']

y_true_all = []
y_pred_all = []

for tweet_id in df_test_only['tweet_id'].unique():
    row_any = df_test_only[df_test_only['tweet_id'] == tweet_id].iloc[0]
    true_label_4 = row_any['true_4class']
    tweet_text   = row_any['tweet_clean']

    sum_probas = np.array([0.0, 0.0, 0.0, 0.0])
    nb_models = 0

    for i in range(6):
        if i in models_per_annotator_4class:
            model_i = models_per_annotator_4class[i]
            class_order = model_i.named_steps['clf'].classes_
            probas_4 = model_i.predict_proba([tweet_text])[0]

            # On ajoute aux sommes de proba
            order_map = {c: idx for idx, c in enumerate(class_order)}
            for j, c in enumerate(all_4classes):
                if c in order_map:
                    sum_probas[j] += probas_4[order_map[c]]
            nb_models += 1

    if nb_models == 0:
        pred_4 = '-'
    else:
        avg_probas = sum_probas / nb_models
        idx_max = np.argmax(avg_probas)
        pred_4 = all_4classes[idx_max]

    y_true_all.append(true_label_4)
    y_pred_all.append(pred_4)

# ----------------------------------------------
# 7) Évaluation finale 4 classes
# ----------------------------------------------
final_true = []
final_pred = []

for t, p in zip(y_true_all, y_pred_all):
    if t in all_4classes:
        final_true.append(t)
        final_pred.append(p if p in all_4classes else '-')

print("\n=== Évaluation finale agrégée (4 classes) ===")
print(classification_report(final_true, final_pred, labels=all_4classes, zero_division=0))

cm = confusion_matrix(final_true, final_pred, labels=all_4classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_4classes,
            yticklabels=all_4classes)
plt.title("Matrice de Confusion - 4 Classes (Agrégées)")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.tight_layout()
plt.show()
