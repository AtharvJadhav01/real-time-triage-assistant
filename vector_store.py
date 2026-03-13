"""
Vector database using FAISS for fast similarity search.
"""

import faiss
import numpy as np


class VectorStore:

    def __init__(self, dim):
        """
        Initialize FAISS index.
        """
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, docs):
        """
        Add embeddings + metadata.
        """
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)

        self.metadata.extend(docs)

    def search(self, query_embedding, k=10):
        """
        Return top k most similar documents.
        """
        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results