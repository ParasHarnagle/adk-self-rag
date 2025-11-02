# from __future__ import annotations

# import logging
# from typing import Any, Callable, Optional, TypeAlias

# from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
# from google.adk.tools.tool_context import ToolContext
# from pinecone import Pinecone

# logger = logging.getLogger(__name__)

# EmbedderFn: TypeAlias = Callable[[str], list[float]]


# class PineconeIndexRetrieval(BaseRetrievalTool):
#     """
#     A robust tool for retrieving documents from a Pinecone index using vector similarity.
#     """

#     def __init__(
#         self,
#         *,
#         name: str,
#         description: str,
#         pinecone: Pinecone,
#         index_name: str,
#         namespace: str,
#         embedder: EmbedderFn,
#         top_k: int = 10,
#         key_text: str = "text",
#     ):
#         super().__init__(name=name, description=description)
#         self.pinecone = pinecone
#         self.index = self.pinecone.Index(index_name)
#         self.index_name = index_name
#         self.namespace = namespace
#         self.embedder = embedder
#         self.top_k = top_k
#         self.key_text = key_text

#     async def run_async(
#         self,
#         *,
#         args: dict[str, Any],
#         tool_context: Optional[ToolContext] = None,
#     ) -> list[str]:
#         query = args.get("query")

#         if not isinstance(query, str) or not query.strip():
#             logger.warning("Invalid or missing query")
#             raise ValueError("Query must be a non-empty string.")

#         logger.info(f"Running Pinecone retrieval for query: {query!r}")

#         try:
#             vector = self.embedder(query)
#         except Exception as e:
#             logger.error("Failed to generate embedding vector", exc_info=True)
#             raise RuntimeError("Embedder failed to generate a valid vector.") from e

#         if not isinstance(vector, list) or not all(
#             isinstance(x, float) for x in vector
#         ):
#             raise TypeError("Embedder must return a list of floats.")

#         logger.debug(f"Embedding vector generated (dim={len(vector)})")

#         try:
#             results = self.index.query(
#                 vector=vector,
#                 top_k=self.top_k,
#                 namespace=self.namespace,
#                 include_metadata=True,
#             )
#         except Exception as e:
#             logger.error("Pinecone query failed", exc_info=True)
#             raise RuntimeError("Failed to query Pinecone index.") from e

#         matches = results.get("matches", [])
#         texts = [
#             match["metadata"][self.key_text]
#             for match in matches
#             if isinstance(match.get("metadata", {}).get(self.key_text), str)
#         ]

#         logger.info(f"Retrieved {len(texts)} results from Pinecone.")

#         return texts




import faiss
import numpy as np
import os
import pickle
from typing import List

class FaissIndexRetrieval:
    def __init__(self,
                 name: str,
                 description: str,
                 index_path: str,
                 embedder,
                 top_k: int = 5):
        self.name = name
        self.description = description
        self.index_path = index_path
        self.embedder = embedder
        self.top_k = top_k

        # Load FAISS index and metadata
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        self.index = faiss.read_index(index_path)

        # optional: load metadata (doc ids, texts)
        meta_path = os.path.splitext(index_path)[0] + "_meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = None

    def retrieve(self, query: str) -> List[dict]:
        """Retrieve top_k documents given a query string"""
        query_vector = np.array([self.embedder(query)], dtype=np.float32)
        print("Searching FAISS index...")
        distances, indices = self.index.search(query_vector, self.top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata[idx] if self.metadata else {"id": idx}
            results.append({
                "score": float(dist),
                "metadata": meta
            })

        return results
