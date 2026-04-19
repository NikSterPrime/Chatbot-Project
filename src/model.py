import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import exact_pattern_to_tag, labels, texts
from preprocessing import clean_texts

vectoriser = TfidfVectorizer(
    ngram_range=(1, 3),
    sublinear_tf=True,
    max_df=0.95,
)
cleaned_texts = [clean_texts(text) for text in texts]
X = vectoriser.fit_transform(cleaned_texts)
y = labels

model = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    C=2.0,
    solver="lbfgs",
)
model.fit(X,y)


def _rank_intents_from_cleaned(cleaned_text):
    vector = vectoriser.transform([cleaned_text])
    probs = model.predict_proba(vector)[0]
    sorted_indices = np.argsort(probs)[::-1]
    return [(str(model.classes_[idx]), float(probs[idx])) for idx in sorted_indices]

def predict_intent_details(text):
    cleaned = clean_texts(text)

    if not cleaned:
        return "fallback", 0.0, 0.0

    if cleaned in exact_pattern_to_tag:
        return exact_pattern_to_tag[cleaned], 1.0, 1.0

    ranked = _rank_intents_from_cleaned(cleaned)
    intent, confidence = ranked[0]
    margin = confidence - ranked[1][1] if len(ranked) > 1 else confidence

    return intent, confidence, margin


def predict_intent_rankings(text, top_k=3):
    cleaned = clean_texts(text)

    if not cleaned:
        return [("fallback", 0.0)]

    if cleaned in exact_pattern_to_tag:
        return [(exact_pattern_to_tag[cleaned], 1.0)]

    ranked = _rank_intents_from_cleaned(cleaned)
    return ranked[:top_k]


def predict_intent_with_confidence(text):
    intent, confidence, _ = predict_intent_details(text)
    return intent, confidence

