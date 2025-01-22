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
    # Supprime les URLs
    text = re.sub(r'https?://\S+', '', text)
    # Supprime les mentions @
    text = re.sub(r'@\w+', '', text)
    # Garde seulement caractères alphabétiques et espaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Passe en minuscules
    text = text.lower()
    # Supprime espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["tweet_clean"] = df["tweet"].apply(clean_tweet)

# ----------------------------------------------
# 1) Conversion en listes Python et ajout des colonnes
# ----------------------------------------------
df['gender_list']     = df['gender_annotators'].apply(ast.literal_eval)
df['age_list']        = df['age_annotators'].apply(ast.literal_eval)
df['eval_list']       = df['evaluation'].apply(ast.literal_eval)
df['eval_type_list']  = df['evaluation_type'].apply(ast.literal_eval)

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
            'eval_annotator':      label_i,
            'eval_type_annotator': type_i
        })
    return new_rows

all_rows = []
for idx, row in df.iterrows():
    all_rows.extend(explode_annotators(row))

df_exploded = pd.DataFrame(all_rows)

# ----------------------------------------------
# 3) Split train / test (au niveau du tweet_id)
# ----------------------------------------------
unique_ids = df_exploded['tweet_id'].unique()
train_ids, test_ids = train_test_split(
    unique_ids,
    test_size=0.2,
    random_state=42  # pour la reproductibilité
)

df_exploded_train = df_exploded[df_exploded['tweet_id'].isin(train_ids)].copy()
df_exploded_test  = df_exploded[df_exploded['tweet_id'].isin(test_ids)].copy()

# ----------------------------------------------
# 4) Entraînement binaire (Sexiste vs Non) 
#    pour chacun des 6 annotateurs
# ----------------------------------------------
# On ajoute sublinear_tf, stop_words='english', min_df=2 et class_weight='balanced'
param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.01, 0.1, 1, 10, 100],
}

models_per_annotator = {}
results_gs = {}

for i in range(6):
    print(f"=== Annotateur {i} ===")
    
    subset_train_i = df_exploded_train[df_exploded_train['annotator_index'] == i]
    X_i_train = subset_train_i['tweet_clean']
    y_i_train = subset_train_i['eval_annotator']
    
    pipeline_i = Pipeline([
        ('tfidf', TfidfVectorizer(
            sublinear_tf=True,
            stop_words='english',  # si vos tweets sont en anglais
            min_df=2
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced', 
            random_state=42
        ))
    ])
    
    gs = GridSearchCV(
        pipeline_i,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    gs.fit(X_i_train, y_i_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best score (CV) = {gs.best_score_:.3f}")
    
    best_model_i = gs.best_estimator_
    models_per_annotator[i] = best_model_i
    results_gs[i] = (gs.best_params_, gs.best_score_)
    print()

# ----------------------------------------------
# 5) Évaluation binaire par annotateur
# ----------------------------------------------
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

for i in range(6):
    print(f"\n=== Évaluation du Modèle Annotateur {i} ===")
    best_model_i = models_per_annotator[i]
    
    subset_test_i = df_exploded_test[df_exploded_test['annotator_index'] == i]
    X_i_test = subset_test_i['tweet_clean']
    y_i_test = subset_test_i['eval_annotator']
    
    y_i_pred = best_model_i.predict(X_i_test)
    
    acc_i = accuracy_score(y_i_test, y_i_pred)
    print(f"Accuracy Annotateur {i} : {acc_i:.3f}")
    print("Classification Report:\n", 
          classification_report(y_i_test, y_i_pred, 
                                target_names=['Non Sexiste','Sexiste'],
                                zero_division=0))
    
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
# 6) Agrégation binaire au niveau du TWEET 
#    (moyenne des probas des 6 modèles)
# ----------------------------------------------
def is_sexist_majority(evals_str):
    # nb_yes >= nb_no => 1 (sexiste), sinon 0 (non sexiste).
    nb_yes = sum(e == 'YES' for e in ast.literal_eval(evals_str))
    nb_no  = sum(e == 'NO'  for e in ast.literal_eval(evals_str))
    return 1 if nb_yes >= nb_no else 0

df['label_global'] = df['evaluation'].apply(is_sexist_majority)
df_test_only = df[df['tweet_id'].isin(test_ids)].copy()

y_pred_global = []
y_true_global = []

for tweet_id in df_test_only['tweet_id'].unique():
    row_any = df_test_only[df_test_only['tweet_id'] == tweet_id].iloc[0]
    true_label = row_any['label_global']
    tweet_text = row_any['tweet_clean']
    
    # Moyenne des probas "YES"
    probas_yes = []
    for i in range(6):
        model_i = models_per_annotator[i]
        p_yes = model_i.predict_proba([tweet_text])[0,1]
        probas_yes.append(p_yes)
    
    mean_yes = np.mean(probas_yes)
    final_pred = 1 if mean_yes >= 0.5 else 0
    
    y_pred_global.append(final_pred)
    y_true_global.append(true_label)

acc_global = accuracy_score(y_true_global, y_pred_global)
print("\n=== Évaluation au niveau TWEET (moyenne des 6 modèles) ===")
print(f"Accuracy: {acc_global:.4f}")
print("Classification Report:\n",
      classification_report(y_true_global, y_pred_global, 
                            target_names=['Non Sexiste','Sexiste'],
                            zero_division=0))

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

# ----------------------------------------------
# 7) Classification multi-classes : DIRECT / REPORTED / JUDGEMENTAL
#    uniquement pour les tweets jugés SEXISTES par l'annotateur
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

models_per_annotator_type = {}

for i in range(6):
    print(f"=== Annotateur {i} (multi-classes) ===")
    
    subset_train_i = df_exploded_sexist_train[df_exploded_sexist_train['annotator_index'] == i]
    if len(subset_train_i) < 5:
        print("Peu ou pas de tweets sexistes pour cet annotateur => pas d'entraînement.")
        continue
    
    X_i_train = subset_train_i['tweet_clean']
    y_i_train = subset_train_i['eval_type_annotator']
    
    # PAS de multi_class explicit -> on supprime le warning
    pipeline_i = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])
    
    gs = GridSearchCV(
        pipeline_i,
        param_grid=param_grid_multi,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    gs.fit(X_i_train, y_i_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best score (CV) = {gs.best_score_:.3f}")
    
    best_model_i = gs.best_estimator_
    models_per_annotator_type[i] = best_model_i
    print()

# ----------------------------------------------
# 8) Évaluation multi-classes par annotateur
# ----------------------------------------------
type_classes = ['DIRECT','REPORTED','JUDGEMENTAL']

for i in range(6):
    if i not in models_per_annotator_type:
        continue
    
    print(f"\n=== Évaluation multi-classes (Annotateur {i}) ===")
    model_i = models_per_annotator_type[i]
    
    # Filtre: labels corrects (DIRECT/REPORTED/JUDGEMENTAL)
    subset_test_i = df_exploded_sexist_test[
        (df_exploded_sexist_test['annotator_index'] == i) &
        (df_exploded_sexist_test['eval_type_annotator'].isin(type_classes))
    ]
    if len(subset_test_i) == 0:
        print("Aucun tweet sexiste test pour cet annotateur => skip")
        continue
    
    X_i_test = subset_test_i['tweet_clean']
    y_i_test = subset_test_i['eval_type_annotator']
    
    y_i_pred = model_i.predict(X_i_test)
    # Juste au cas où le classif sortirait un label inconnu :
    y_i_pred = [lab if lab in type_classes else 'DIRECT' for lab in y_i_pred]
    
    acc_i = accuracy_score(y_i_test, y_i_pred)
    print(f"Accuracy (multi-classe) Annotateur {i} : {acc_i:.3f}")
    
    print(classification_report(
        y_i_test,
        y_i_pred,
        labels=type_classes,
        target_names=type_classes,
        zero_division=0  # pour éviter les warnings si classe absente
    ))
    
    cm_i = confusion_matrix(y_i_test, y_i_pred, labels=type_classes)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm_i, annot=True, fmt='d', cmap='Blues',
                xticklabels=type_classes,
                yticklabels=type_classes)
    plt.title(f"Matrice de Confusion - Type - Annotateur {i}")
    plt.xlabel("Prédit")
    plt.ylabel("Vérité Annotateur")
    plt.tight_layout()
    plt.show()

# ----------------------------------------------
# 9) Agrégation multi-classes au niveau du tweet
# ----------------------------------------------
def majority_eval_type(evals_list, types_list):
    """
    Construit la classe majoritaire (DIRECT/REPORTED/JUDGEMENTAL)
    pour les annotateurs qui disent 'YES'.
    """
    selected = []
    for e, t in zip(evals_list, types_list):
        if e == 'YES' and t != '-':
            selected.append(t)
    if len(selected) == 0:
        return '-'  # Aucun type déterminé
    freq = pd.Series(selected).value_counts()
    return freq.idxmax()

def compute_majority_type(row):
    evals = ast.literal_eval(row['evaluation'])
    types = ast.literal_eval(row['evaluation_type'])
    return majority_eval_type(evals, types)

df['label_global_type'] = df.apply(compute_majority_type, axis=1)
df_test_only['label_global_type'] = df_test_only.apply(compute_majority_type, axis=1)

y_true_multi = []
y_pred_multi = []

for tweet_id in df_test_only['tweet_id'].unique():
    row_any = df_test_only[df_test_only['tweet_id'] == tweet_id].iloc[0]
    true_type_label   = row_any['label_global_type']  # DIRECT / REPORTED / JUDGEMENTAL / '-'
    true_label_binary = row_any['label_global']       # 0 ou 1
    
    tweet_text = row_any['tweet_clean']
    
    # Agrégation binaire
    probas_yes = [models_per_annotator[i].predict_proba([tweet_text])[0,1]
                  for i in range(6)]
    mean_yes = np.mean(probas_yes)
    pred_binary = 1 if mean_yes >= 0.5 else 0
    
    # Si pas sexiste => "NON_SEXISTE"
    if pred_binary == 0:
        pred_type = "NON_SEXISTE"
    else:
        sum_probas = np.array([0.0, 0.0, 0.0])
        nb_models = 0
        for i in range(6):
            if i in models_per_annotator_type:
                model_multi = models_per_annotator_type[i]
                p_multi = model_multi.predict_proba([tweet_text])[0]
                # Vérifier l'ordre
                class_order = model_multi.classes_
                direct_idx = list(class_order).index('DIRECT')
                reported_idx = list(class_order).index('REPORTED')
                judgemental_idx = list(class_order).index('JUDGEMENTAL')
                sum_probas[0] += p_multi[direct_idx]
                sum_probas[1] += p_multi[reported_idx]
                sum_probas[2] += p_multi[judgemental_idx]
                nb_models += 1
        
        if nb_models == 0:
            pred_type = '-'
        else:
            avg_probas = sum_probas / nb_models
            idx_max = np.argmax(avg_probas)
            pred_type = ['DIRECT','REPORTED','JUDGEMENTAL'][idx_max]
    
    y_true_multi.append(true_type_label)
    y_pred_multi.append(pred_type)

# Filtrons seulement les cas où le vrai label n'est pas '-'
# (et ajoutons "NON_SEXISTE" si on veut l'inclure).
final_true = []
final_pred = []
for t, p in zip(y_true_multi, y_pred_multi):
    # On conserve si la vérité est DIRECT/REPORTED/JUDGEMENTAL ou NON_SEXISTE
    if t in type_classes or t == "NON_SEXISTE":
        final_true.append(t)
        final_pred.append(p)

print("\n=== Évaluation Agrégée Multi-Classes (y compris NON_SEXISTE) ===")
all_labels = type_classes + ["NON_SEXISTE"]
print(classification_report(final_true, final_pred, labels=all_labels, zero_division=0))

cm = confusion_matrix(final_true, final_pred, labels=all_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_labels,
            yticklabels=all_labels)
plt.title("Matrice de Confusion - Multi-Classes (Agrégé)")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.tight_layout()
plt.show()
