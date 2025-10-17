from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from domain.schemas import RetrievalResult

class VectorStore:
    def upsert(self, chunks: List[Tuple[str, Dict[str, Any]]]) -> None:
        raise NotImplementedError
    def similarity_search(self, query_vec: list[float], k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        raise NotImplementedError
