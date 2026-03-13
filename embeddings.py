"""
Embedding module for fast semantic retrieval.
"""

from sentence_transformers import SentenceTransformer

class EmbeddingModel:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Small embedding model for low latency (<10ms per embedding).
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Generate embeddings.
        """
        return self.model.encode(texts, show_progress_bar=False)