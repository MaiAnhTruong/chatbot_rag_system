#/home/truong/IDT/RAG/chatbot_rag_system/src/retriever/service.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from domain.schemas import RetrievalResult
from embeddings.base import Embeddings
from vector.base import VectorStore

class RetrieverService:
    def __init__(self, emb: Embeddings, store: VectorStore):
        self.emb = emb
        self.store = store

    def retrieve(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        qv = self.emb.encode_query(query)
        return self.store.similarity_search(qv, k=top_k, metadata_filter=metadata_filter)
