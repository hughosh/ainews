import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from topic_extraction import extract_keywords

class VectorDatabase:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the vector database with a pre-trained SBERT model.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []  # List of enriched article dictionaries.
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

    def build_index(self, articles: List[Dict[str, Any]]) -> None:
        """
        Builds a FAISS index from a list of articles.
        Each stored document has the following format:
          {
             "title": "Election Results Announced",
             "source": "CNN",
             "timestamp": "2025-04-04T12:00:00",
             "topic": "Politics",
             "keywords": ["election", "results", "president"],
             "vector": [0.123, 0.456, ...]
          }
        """
        vectors = []
        self.documents = []  # Reset the document list.
        
        for article in articles:
            title = article.get("title", "No Title")
            source = (article.get("source", {}).get("name") 
                      if isinstance(article.get("source"), dict) else article.get("source", "Unknown Source"))
            published_at_str = article.get("publishedAt", "")
            timestamp = published_at_str  # Assume ISO formatted string.
            description = article.get("description", "")
            
            # Combine title and description for text analysis.
            text = f"{title}. {description}"
            # Extract keywords using TF-IDF.
            keywords = extract_keywords(text, top_n=5)
            # Derive topic as the top keyword if available.
            topic = keywords[0] if keywords else "Unknown Topic"
            # Get embedding vector from the text.
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            vector = embedding.astype('float32')
            vectors.append(vector)
            
            # Create the enriched document.
            doc = {
                "title": title,
                "source": source,
                "timestamp": timestamp,
                "topic": topic,
                "keywords": keywords,
                "vector": vector.tolist()
            }
            self.documents.append(doc)
        
        if vectors:
            vectors_np = np.array(vectors, dtype='float32')
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors_np)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the FAISS index with the given text and returns the top k matching documents.
        """
        if self.index is None:
            raise ValueError("Index has not been built. Call build_index() first.")
        
        query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True).astype('float32')
        query_embedding = np.array([query_embedding])
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

