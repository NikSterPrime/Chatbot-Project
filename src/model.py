from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import label
from preprocessing import clean_texts

vectoriser = TfidfVectorizer()
X = []
y = []
def vectoriseText(texts):
    X = vectoriser.fit_transform(texts)
    y = label

    print(X.shape)

model = LogisticRegression()
model.fit(X,y)

def predict_intent(text):
    text = clean_texts(text)
    vector = vectoriser.transform([text])
    prediction = model.predict(vector)
    return prediction[0]

print(predict_intent("Hello"))
print(predict_intent("I feel sad"))

