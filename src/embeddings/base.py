from __future__ import annotations
from typing import List

class Embeddings:
    def encode_texts(self, texts: List[str]) -> List[list[float]]:
        raise NotImplementedError
    def encode_query(self, text: str) -> list[float]:
        raise NotImplementedError
