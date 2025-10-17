import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import re
import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    df = pd.read_csv('data/sentiment_tweets.csv')
    df = df.dropna(subset=['text', 'label'])  # boş değerleri at
    df['label'] = df['label'].astype(int)    # label tipini integer yap
    df['text_clean'] = df['text'].apply(clean_text)

    X_train, X_val, y_train, y_val = train_test_split(
    df['text_clean'], df['label'],
    test_size=0.2,
    random_state=42
)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=200))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    print(classification_report(y_val, preds))

    joblib.dump(model, 'model.pkl')
    print("✅ Model kaydedildi: model.pkl")

if __name__ == "__main__":
    main()