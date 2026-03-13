"""
Main orchestration pipeline.
"""

from embeddings import EmbeddingModel
from vector_store import VectorStore
from context_pruning import ContextPruner
from reasoning_engine import ReasoningEngine


class TriageAssistant:

    def __init__(self):

        self.embedder = EmbeddingModel()

        self.vector_store = VectorStore(dim=384)

        self.pruner = ContextPruner(self.embedder)

        self.reasoner = ReasoningEngine()

    def ingest(self, docs):

        texts = [d["text"] for d in docs]

        embeddings = self.embedder.encode(texts)

        self.vector_store.add(embeddings, docs)

    def query(self, user_input):

        # Step 1: Embed query
        query_emb = self.embedder.encode([user_input])[0]

        # Step 2: Retrieve candidates
        retrieved = self.vector_store.search(query_emb)

        # Step 3: Intelligent Context Pruning
        pruned_context = self.pruner.prune(user_input, retrieved)

        # Step 4: LLM reasoning
        response = self.reasoner.generate_action(
            user_input,
            pruned_context
        )

        return response