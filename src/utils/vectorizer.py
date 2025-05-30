from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_vectorizer(name, max_features):
    if name == "tfidf":
        return TfidfVectorizer(max_features=max_features)
    elif name == "count":
        return CountVectorizer(max_features=max_features)
    else:
        raise ValueError(f"Unsupported vectorizer type: {name}")
