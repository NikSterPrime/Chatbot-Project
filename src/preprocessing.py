import string

CONTRACTIONS = {
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "i'm": "i am",
    "im": "i am",
    "you're": "you are",
    "youre": "you are",
    "it's": "it is",
    "thats": "that is",
    "that's": "that is",
    "what's": "what is",
    "whats": "what is",
    "don't": "do not",
    "dont": "do not",
    "didn't": "did not",
    "didnt": "did not",
    "isn't": "is not",
    "isnt": "is not",
    "i've": "i have",
    "ive": "i have",
    "i'll": "i will",
    "ill": "i will",
}


def _expand_contractions(text):
    words = text.split()
    normalized = [CONTRACTIONS.get(word, word) for word in words]
    return " ".join(normalized)

def clean_texts(text):
    text = text.lower().strip()
    text = _expand_contractions(text)
    text = "".join([char for char in text if char not in string.punctuation])
    return " ".join(text.split())
