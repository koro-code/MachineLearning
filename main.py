import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import re

# Téléchargement des stopwords si ce n'est pas déjà fait
nltk.download('stopwords')

# Prétraitement des tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)  # URLs
    tweet = re.sub(r'\@\w+|\#', '', tweet)  # Mentions et hashtags
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Ponctuation
    tweet = tweet.lower()  # Minuscule
    tokens = [word for word in tweet.split() if word not in stopwords.words('english')]
    return " ".join(tokens)

# Chargement des données
df = pd.read_csv("tweets.csv")

# Nettoyage des tweets
df['cleaned_tweet'] = df['tweet'].apply(preprocess_tweet)

# Extraction de la colonne de classification binaire
# Utilise la première valeur majoritaire dans la liste `evaluation`
df['binary_label'] = df['evaluation'].apply(lambda x: eval(x)[0])  # Convertion de string en liste puis extraction

# Vérification des labels pour la tâche binaire
print(df['binary_label'].value_counts())

# Représentation TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['cleaned_tweet'])
y = df['binary_label']

# Division en ensembles d'entraînement/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle SVM pour la tâche 1
model = SVC(probability=True, kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Prédictions pour la tâche 1
y_pred = model.predict(X_test)
print("Classification binaire (Task 1):")
print(classification_report(y_test, y_pred))

# --- Tâche 2 : Classification multicatégorie ---
# Filtrage des tweets classés comme sexistes
df_task2 = df[df['binary_label'] == 'YES'].copy()

# Extraction des labels multicatégories
df_task2['multiclass_label'] = df_task2['evaluation_type'].apply(lambda x: eval(x)[0])  # Extraction de la première étiquette

# Représentation TF-IDF pour les tweets sexistes
X_task2 = vectorizer.transform(df_task2['cleaned_tweet'])
y_task2 = df_task2['multiclass_label']

# Division en ensembles d'entraînement/test pour la tâche 2
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_task2, y_task2, test_size=0.2, random_state=42)

# Modèle SVM pour la tâche 2
model_task2 = SVC(probability=True, kernel='linear', random_state=42)
model_task2.fit(X_train_2, y_train_2)

# Prédictions pour la tâche 2
y_pred_2 = model_task2.predict(X_test_2)
print("Classification multicatégorie (Task 2):")
print(classification_report(y_test_2, y_pred_2))
