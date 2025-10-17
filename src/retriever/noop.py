from __future__ import annotations
from typing import List, Dict, Any, Optional
from domain.schemas import RetrievalResult

class NoopRetriever:
    """Retriever rỗng: luôn trả danh sách kết quả trống."""
    def retrieve(self, query: str, top_k: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        return []
