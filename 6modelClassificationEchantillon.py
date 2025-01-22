import pandas as pd
import re
import ast
import numpy as np

# Pour la division train/test
from sklearn.model_selection import train_test_split, GridSearchCV

# Pour la pipeline (avec oversampling)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Pour les métriques
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    new_rows = []
    for i in range(len(row['gender_list'])):
        sex_i  = row['gender_list'][i]
        age_i  = row['age_list'][i]
        eval_i = row['eval_list'][i]          # 'YES' ou 'NO'
        type_i = row['eval_type_list'][i]     # 'DIRECT','REPORTED','JUDGEMENTAL' ou '-'
        
        label_i = 1 if eval_i == 'YES' else 0
        new_rows.append({
            'tweet_id':            row['tweet_id'],
            'annotator_index':     i,
            'gender':              sex_i,
            'age_range':           age_i,
            'tweet_clean':         row['tweet_clean'],
            'eval_annotator':      label_i,               # 0 ou 1
            'eval_type_annotator': type_i                 # 'DIRECT','REPORTED','JUDGEMENTAL' ou '-'
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
# 4) Entraînement BINAIRE par annotateur
#    (Sexiste vs Non sexiste) - OVERSAMPLING
# ----------------------------------------------
param_grid_binary = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.01, 0.1, 1, 10, 100],
}

models_per_annotator_binary = {}

for i in range(6):
    subset_train_i = df_exploded_train[df_exploded_train['annotator_index'] == i]
    X_i_train = subset_train_i['tweet_clean']
    y_i_train = subset_train_i['eval_annotator']
    
    # Important: on place le TfidfVectorizer AVANT le RandomOverSampler
    pipeline_i = Pipeline([
        ('tfidf', TfidfVectorizer(
            sublinear_tf=True,
            stop_words='english', 
            min_df=2
        )),
        ('oversample', RandomOverSampler(random_state=42)),
        ('clf', LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ))
    ])
    
    gs = GridSearchCV(
        pipeline_i,
        param_grid=param_grid_binary,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )
    gs.fit(X_i_train, y_i_train)
    
    models_per_annotator_binary[i] = gs.best_estimator_

# ----------------------------------------------
# 5) Entraînement MULTI-CLASSES par annotateur
#    (DIRECT / REPORTED / JUDGEMENTAL)
#    + OVERSAMPLING
# ----------------------------------------------
df_exploded_sexist = df_exploded[
    (df_exploded['eval_annotator'] == 1) & 
    (df_exploded['eval_type_annotator'] != '-')
].copy()

df_exploded_sexist_train = df_exploded_sexist[df_exploded_sexist['tweet_id'].isin(train_ids)]
df_exploded_sexist_test  = df_exploded_sexist[df_exploded_sexist['tweet_id'].isin(test_ids)]

param_grid_multi = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.01, 0.1, 1, 10, 100],
}

models_per_annotator_multi = {}
type_classes = ['DIRECT','REPORTED','JUDGEMENTAL']

for i in range(6):
    subset_train_i = df_exploded_sexist_train[df_exploded_sexist_train['annotator_index'] == i]
    if len(subset_train_i) < 5:
        # Trop peu d'exemples => skip
        continue
    
    X_i_train = subset_train_i['tweet_clean']
    y_i_train = subset_train_i['eval_type_annotator']
    
    pipeline_i = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('oversample', RandomOverSampler(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42))
    ])
    
    gs = GridSearchCV(
        pipeline_i,
        param_grid=param_grid_multi,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )
    gs.fit(X_i_train, y_i_train)
    
    models_per_annotator_multi[i] = gs.best_estimator_

# ----------------------------------------------
# 6) Prédiction finale agrégée (4 classes)
#    "NON_SEXISTE" + "DIRECT" + "REPORTED" + "JUDGEMENTAL"
# ----------------------------------------------
df_test_only = df[df['tweet_id'].isin(test_ids)].copy()

y_true_all = []
y_pred_all = []

def majority_eval_type(evals_list, types_list):
    """Renvoie la classe majoritaire (DIRECT/REPORTED/JUDGEMENTAL) 
       pour les annotateurs qui disent 'YES'."""
    selected = []
    for e, t in zip(evals_list, types_list):
        if e == 'YES' and t != '-':
            selected.append(t)
    if len(selected) == 0:
        return '-'
    freq = pd.Series(selected).value_counts()
    return freq.idxmax()

# On calcule le 'vrai' label agrégé (multi-classes + "NON_SEXISTE")
df_test_only['true_type'] = '-'
for idx in df_test_only.index:
    evals = ast.literal_eval(df_test_only.loc[idx, 'evaluation'])
    types = ast.literal_eval(df_test_only.loc[idx, 'evaluation_type'])
    
    nb_yes = sum(e == 'YES' for e in evals)
    nb_no  = sum(e == 'NO'  for e in evals)
    if nb_yes >= nb_no:
        df_test_only.loc[idx, 'true_type'] = majority_eval_type(evals, types)
    else:
        df_test_only.loc[idx, 'true_type'] = 'NON_SEXISTE'

for tweet_id in df_test_only['tweet_id'].unique():
    row_any = df_test_only[df_test_only['tweet_id'] == tweet_id].iloc[0]
    true_type_label = row_any['true_type']  # DIRECT / REPORTED / JUDGEMENTAL / NON_SEXISTE
    tweet_text      = row_any['tweet_clean']
    
    # 1) Moyenne des probas "Sexiste" sur les 6 modèles binaires
    probas_yes = []
    for i in range(6):
        p_yes = models_per_annotator_binary[i].predict_proba([tweet_text])[0,1]
        probas_yes.append(p_yes)
    mean_yes = np.mean(probas_yes)
    
    # 2) Si < 0.5 => "NON_SEXISTE"
    if mean_yes < 0.5:
        pred_type = "NON_SEXISTE"
    else:
        # 3) Sinon, on agrège les probas DIRECT/REPORTED/JUDGEMENTAL
        sum_probas = np.array([0.0, 0.0, 0.0])
        nb_models = 0
        for i in range(6):
            if i in models_per_annotator_multi:
                model_multi = models_per_annotator_multi[i]
                class_order = model_multi.classes_
                p_multi = model_multi.predict_proba([tweet_text])[0]
                
                # indices de classes
                idx_direct       = list(class_order).index('DIRECT')
                idx_reported     = list(class_order).index('REPORTED')
                idx_judgemental  = list(class_order).index('JUDGEMENTAL')
                
                sum_probas[0] += p_multi[idx_direct]
                sum_probas[1] += p_multi[idx_reported]
                sum_probas[2] += p_multi[idx_judgemental]
                nb_models += 1
        
        if nb_models == 0:
            pred_type = '-'
        else:
            avg_probas = sum_probas / nb_models
            idx_max = np.argmax(avg_probas)
            pred_type = ['DIRECT','REPORTED','JUDGEMENTAL'][idx_max]
    
    y_true_all.append(true_type_label)
    y_pred_all.append(pred_type)

# ----------------------------------------------
# 7) Évaluation finale 4 classes
#    (DIRECT / REPORTED / JUDGEMENTAL / NON_SEXISTE)
# ----------------------------------------------
all_labels = ['DIRECT','REPORTED','JUDGEMENTAL','NON_SEXISTE']
final_true = []
final_pred = []

for t, p in zip(y_true_all, y_pred_all):
    if t in all_labels:
        final_true.append(t)
        final_pred.append(p if p in all_labels else '-')

print("\n=== Évaluation finale agrégée (4 classes) ===")
print(classification_report(final_true, final_pred, labels=all_labels, zero_division=0))

cm = confusion_matrix(final_true, final_pred, labels=all_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_labels,
            yticklabels=all_labels)
plt.title("Matrice de Confusion - 4 Classes Agrégées")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.tight_layout()
plt.show()
