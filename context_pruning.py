"""
Intelligent Context Pruning module.

Filters irrelevant medical records before sending to LLM
to reduce token usage and latency.
"""

from sklearn.metrics.pairwise import cosine_similarity


class ContextPruner:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def prune(self, query, docs, max_docs=3):
        """
        Select the most relevant documents.

        Steps:
        1. Embed query
        2. Embed retrieved docs
        3. Rank by cosine similarity
        4. Apply metadata filtering
        """

        query_emb = self.embedding_model.encode([query])[0]

        doc_texts = [d["text"] for d in docs]
        doc_embs = self.embedding_model.encode(doc_texts)

        scores = cosine_similarity([query_emb], doc_embs)[0]

        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        # Metadata filtering example
        filtered = [
            doc for score, doc in ranked
            if "dental" not in doc.get("tags", [])
        ]

        return filtered[:max_docs]