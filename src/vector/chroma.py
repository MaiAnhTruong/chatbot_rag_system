# src/vector/chroma.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from app.settings import SETTINGS
from domain.schemas import RetrievalResult
from vector.base import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name: str, embedder):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self._client = chromadb.PersistentClient(
            path=SETTINGS.CHROMA_DIR,
            settings=ChromaSettings(allow_reset=False)
        )
        self._embedder = embedder
        self._col = self._client.get_or_create_collection(name=collection_name)

    def upsert(self, chunks: List[Tuple[str, Dict[str, Any]]]) -> None:
        if not chunks:
            return
        texts = [t for t, _ in chunks]
        metadatas = [md for _, md in chunks]
        ids = [md.get("id") or f"doc-{i}" for i, md in enumerate(metadatas)]
        embeddings = self._embedder.encode_texts(texts)
        self._col.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings)

    def similarity_search(
        self,
        query_vec: list[float],
        k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        res = self._col.query(
            query_embeddings=[query_vec],
            n_results=max(1, k),
            where=metadata_filter if metadata_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        # Chroma dùng distance; metric mặc định thường là cosine distance (0..2).
        # Giữ nguyên "distance" để client tự hiểu, tránh tính "similarity" sai.
        out: List[RetrievalResult] = []
        for txt, md, dist in zip(docs, metas, dists):
            score = None if dist is None else float(1.0 - dist)  # optional similarity heuristic
            out.append(RetrievalResult(page_content=txt, metadata=md or {}, score=score))
        return out
