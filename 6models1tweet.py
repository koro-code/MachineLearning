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
    text = re.sub(r'https?://\S+', '', text)   # Supprime les URLs
    text = re.sub(r'@\w+', '', text)           # Supprime les mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Caractères alpha
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()   # Espaces multiples
    return text

df["tweet_clean"] = df["tweet"].apply(clean_tweet)

# ----------------------------------------------
# 1) Explosion en 6 lignes (une par annotateur)
# ----------------------------------------------
df['gender_list'] = df['gender_annotators'].apply(ast.literal_eval)
df['age_list']    = df['age_annotators'].apply(ast.literal_eval)
df['eval_list']   = df['evaluation'].apply(ast.literal_eval)

df['tweet_id'] = df.index  # ou df["ID"] si dispo

def explode_annotators(row):
    new_rows = []
    for i in range(len(row['gender_list'])):
        sex_i  = row['gender_list'][i]
        age_i  = row['age_list'][i]
        eval_i = row['eval_list'][i]  # 'YES'/'NO'
        label_i = 1 if eval_i == 'YES' else 0
        
        new_rows.append({
            'tweet_id':        row['tweet_id'],
            'annotator_index': i,  # 0..5
            'gender':          sex_i,
            'age_range':       age_i,
            'tweet_clean':     row['tweet_clean'],
            'eval_annotator':  label_i
        })
    return new_rows

all_rows = []
for idx, row in df.iterrows():
    all_rows.extend(explode_annotators(row))

df_exploded = pd.DataFrame(all_rows)

# ----------------------------------------------
# 2) Split train / test (au niveau du tweet)
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
# 3) Hyperparamétrage + Entraînement
#    -> 1 pipeline par annotateur
# ----------------------------------------------
param_grid = {
    # TfidfVectorizer
    'tfidf__ngram_range': [(1,1), (1,2)],
    
    # LogisticRegression
    'clf__C': [0.01, 0.1, 1, 10, 100],
    # On peut aussi tuner 'clf__penalty': ['l2', 'l1'] si on veut 
    # (nécessite solver approprié)
}

models_per_annotator = {}
results_gs = {}  # pour garder le résumé GridSearch

for i in range(6):
    print(f"=== Annotateur {i} ===")
    
    # Sous-ensemble d'entraînement pour l'annotateur i
    subset_train_i = df_exploded_train[df_exploded_train['annotator_index'] == i]
    X_i_train = subset_train_i['tweet_clean']
    y_i_train = subset_train_i['eval_annotator']
    
    # Pipeline de base
    pipeline_i = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    # Grid Search
    gs = GridSearchCV(
        pipeline_i, 
        param_grid=param_grid, 
        scoring='accuracy', 
        cv=3,               # cross-validation
        verbose=1,
        n_jobs=-1           # utilise tous les CPU dispo
    )
    
    gs.fit(X_i_train, y_i_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best score (CV) = {gs.best_score_:.3f}")
    
    # Récupérer le meilleur modèle et le stocker
    best_model_i = gs.best_estimator_
    models_per_annotator[i] = best_model_i
    
    # On mémorise les résultats pour info
    results_gs[i] = (gs.best_params_, gs.best_score_)
    print()

# ----------------------------------------------
# 4) Évaluation par annotateur (confusion matrix)
#    -> on évalue chaque modèle i sur le sous-ensemble test correspondant
# ----------------------------------------------
for i in range(6):
    print(f"\n=== Évaluation du Modèle Annotateur {i} ===")
    best_model_i = models_per_annotator[i]
    
    # Extraire la portion du test qui concerne l'annotateur i
    subset_test_i = df_exploded_test[df_exploded_test['annotator_index'] == i]
    X_i_test = subset_test_i['tweet_clean']
    y_i_test = subset_test_i['eval_annotator']
    
    y_i_pred = best_model_i.predict(X_i_test)
    
    # Métriques
    acc_i = accuracy_score(y_i_test, y_i_pred)
    report_i = classification_report(y_i_test, y_i_pred, target_names=['Non Sexiste','Sexiste'])
    print(f"Accuracy Annotateur {i} : {acc_i:.3f}")
    print("Classification Report:\n", report_i)
    
    # Matrice de confusion
    cm_i = confusion_matrix(y_i_test, y_i_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm_i, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non Sexiste','Sexiste'],
                yticklabels=['Non Sexiste','Sexiste'])
    plt.title(f"Matrice de Confusion - Modèle Annotateur {i}")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité Annotateur")
    plt.tight_layout()
    plt.show()

# ----------------------------------------------
# 5) Agrégation au niveau TWEET 
#    -> moyenne des 6 probas "YES"
# ----------------------------------------------

# 5.1) Calculer la vérité majoritaire (>=3 => YES)
def is_sexist_majority(evals_str):
    nb_yes = sum(e == 'YES' for e in ast.literal_eval(evals_str))
    nb_no  = sum(e == 'NO'  for e in ast.literal_eval(evals_str))
    return 1 if nb_yes >= nb_no else 0

df['label_global'] = df['evaluation'].apply(is_sexist_majority)
df_test_only = df[df['tweet_id'].isin(test_ids)].copy()  # tweets du test

y_pred_global = []
y_true_global = []

for tweet_id in df_test_only['tweet_id'].unique():
    # Le "vrai" label majoritaire
    row_any = df_test_only[df_test_only['tweet_id'] == tweet_id].iloc[0]
    true_label = row_any['label_global']
    
    # On récupère le texte (nettoyé)
    tweet_text = row_any['tweet_clean']
    
    # Moyenne des probas
    probas_yes = []
    for i in range(6):
        model_i = models_per_annotator[i]
        p_yes = model_i.predict_proba([tweet_text])[0,1]
        probas_yes.append(p_yes)
    
    mean_yes = np.mean(probas_yes)
    final_pred = 1 if mean_yes >= 0.5 else 0
    
    y_pred_global.append(final_pred)
    y_true_global.append(true_label)

# ----------------------------------------------
# 6) Évaluation globale (moyenne des 6 modèles)
# ----------------------------------------------
acc_global = accuracy_score(y_true_global, y_pred_global)
report_global = classification_report(y_true_global, y_pred_global, target_names=['Non Sexiste','Sexiste'])

print("\n=== Évaluation au niveau TWEET (Moyenne des 6 modèles) ===")
print(f"Accuracy: {acc_global:.4f}")
print("Classification Report:\n", report_global)

cm_global = confusion_matrix(y_true_global, y_pred_global)
plt.figure(figsize=(5,4))
sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non Sexiste','Sexiste'],
            yticklabels=['Non Sexiste','Sexiste'])
plt.title("Matrice de Confusion - Niveau Tweet (Moyenne)")
plt.xlabel("Prédictions (Moyenne 6 modèles)")
plt.ylabel("Vérité (Majorité Humaine)")
plt.tight_layout()
plt.show()
