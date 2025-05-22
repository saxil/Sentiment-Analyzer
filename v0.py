import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

data = {
    'text': [
        "I love this movie!", 
        "This was the worst movie I've ever seen", 
        "Absolutely fantastic! Highly recommend.", 
        "It was okay, not great but not terrible.",
        "I would never watch this again."
    ],
    'sentiment': [1, 0, 1, 1, 0]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: ' '.join(
    word for word in x.lower().split() if word not in stop_words and word not in string.punctuation
))

X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

X_test_tfidf = tfidf.transform(X_test)
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_sentiment(text):
    text = ' '.join(word for word in text.lower().split() if word not in stop_words and word not in string.punctuation)
    feature = tfidf.transform([text])
    prediction = model.predict(feature)
    return "Positive" if prediction[0] == 1 else "Negative"

new_review = "I had a wonderful experience!"
print(f"The sentiment of the review: '{new_review}' is {predict_sentiment(new_review)}")