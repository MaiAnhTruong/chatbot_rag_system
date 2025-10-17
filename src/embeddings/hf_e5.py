# src/embeddings/hf_e5.py
from __future__ import annotations
from typing import List, Optional
from embeddings.base import Embeddings
from app.settings import SETTINGS

class HFE5Embeddings(Embeddings):
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or SETTINGS.EMBEDDING_MODEL
        from sentence_transformers import SentenceTransformer
        try:
            self.model = SentenceTransformer(self.model_name, device=device or "cuda")
        except Exception:
            # fallback CPU nếu không có CUDA
            self.model = SentenceTransformer(self.model_name, device="cpu")
        self.model.max_seq_length = 512

    def encode_texts(self, texts: List[str]) -> List[list[float]]:
        payload = [f"passage: {t}" for t in texts]
        embs = self.model.encode(payload, normalize_embeddings=True, convert_to_numpy=True).tolist()
        return embs

    def encode_query(self, text: str) -> list[float]:
        q = f"query: {text}"
        emb = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        return emb
