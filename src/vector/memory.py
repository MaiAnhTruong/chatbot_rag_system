from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from domain.schemas import RetrievalResult
from vector.base import VectorStore
import hashlib, random

class InMemoryVectorStore(VectorStore):
    def __init__(self, dim: int = 16):
        self._docs: List[Tuple[str, Dict[str, Any], list[float]]] = []
        self._dim = dim

    def _hash_vec(self, text: str) -> list[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rnd = random.Random(seed)
        return [rnd.uniform(-1, 1) for _ in range(self._dim)]

    def upsert(self, chunks: List[Tuple[str, Dict[str, Any]]]) -> None:
        for text, md in chunks:
            self._docs.append((text, md, self._hash_vec(text)))

    def similarity_search(self, query_vec: list[float], k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        # nếu query_vec không đúng dim (do fallback), cứ dùng hash vec
        if not query_vec or len(query_vec) != self._dim:
            query_vec = self._hash_vec("fallback-query")
        def dot(a,b): return sum(x*y for x,y in zip(a,b))
        docs = self._docs
        if metadata_filter:
            docs = [d for d in docs if all(d[1].get(k) == v for k, v in metadata_filter.items())]
        ranked = sorted(docs, key=lambda d: dot(d[2], query_vec), reverse=True)[:max(1,k)]
        return [RetrievalResult(page_content=t, metadata=md, score=None) for (t, md, _) in ranked]
