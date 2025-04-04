from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text: str, top_n: int = 5):
    """
    Extracts the top_n keywords from the given text using TF-IDF.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tfidf_matrix.toarray().flatten().argsort()[::-1]
    top_keywords = [feature_array[i] for i in tfidf_sorting[:top_n]]
    return top_keywords

