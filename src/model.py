import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import exact_pattern_to_tag, labels, texts
from preprocessing import clean_texts

vectoriser = TfidfVectorizer(ngram_range=(1, 2))
cleaned_texts = [clean_texts(text) for text in texts]
X = vectoriser.fit_transform(cleaned_texts)
y = labels

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X,y)

def predict_intent_details(text):
    cleaned = clean_texts(text)

    if not cleaned:
        return "fallback", 0.0, 0.0

    if cleaned in exact_pattern_to_tag:
        return exact_pattern_to_tag[cleaned], 1.0, 1.0

    vector = vectoriser.transform([cleaned])

    probs = model.predict_proba(vector)[0]
    index = np.argmax(probs)

    sorted_probs = np.sort(probs)
    confidence = float(probs[index])
    intent = str(model.classes_[index])
    margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else confidence

    return intent, confidence, margin


def predict_intent_with_confidence(text):
    intent, confidence, _ = predict_intent_details(text)
    return intent, confidence

