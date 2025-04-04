import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class VectorDatabase:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the vector database with a pre-trained embedding model.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []  # To store article metadata in the same order as the index

    def build_index(self, articles: List[Dict[str, Any]], text_key: str = None) -> None:
        """
        Builds a FAISS index from a list of articles.
        
        Parameters:
          articles (List[Dict[str, Any]]): List of articles fetched from NewsAPI.
          text_key (str): Optional. If provided, uses this key from the article as the text to embed.
                          Otherwise, concatenates title and description.
        """
        texts = []
        self.documents = []  # Reset document list
        for article in articles:
            if text_key and text_key in article:
                text = article[text_key]
            else:
                title = article.get("title", "")
                description = article.get("description", "")
                text = f"{title}. {description}"
            texts.append(text)
            self.documents.append(article)
        
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype='float32'))

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the FAISS index with a given text and returns the top k matching articles.
        
        Parameters:
          query_text (str): The text query.
          k (int): The number of top results to return.
        
        Returns:
          List[Dict[str, Any]]: List of article metadata dictionaries for the top k matches.
        """
        if self.index is None:
            raise ValueError("Index has not been built. Call build_index() first.")
        
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)
        distances, indices = self.index.search(np.array(query_embedding, dtype='float32'), k)
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

