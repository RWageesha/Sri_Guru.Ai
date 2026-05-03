import math
import os
import re
from typing import List

import requests


class RAGRetriever:
    def __init__(
        self,
        data_dir="data",
        base_url="http://localhost:11434",
        embedding_model="nomic-embed-text",
    ):
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.docs, self.doc_texts = self._load_docs(data_dir)
        self._use_embeddings = False
        self._doc_embeddings: List[List[float]] = []
        self._doc_tokens = [self._tokenize(text) for text in self.doc_texts]

        if self.doc_texts:
            self._build_embeddings(self.doc_texts)

    def _load_docs(self, data_dir):
        docs = []
        doc_texts = []
        for fname in os.listdir(data_dir):
            with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
                text = f.read()
                docs.append({"filename": fname, "text": text})
                doc_texts.append(text)
        return docs, doc_texts

    def _build_embeddings(self, doc_texts):
        embeddings = []
        for text in doc_texts:
            vector = self._embed_text(text)
            if not vector:
                self._use_embeddings = False
                return
            embeddings.append(vector)
        self._doc_embeddings = embeddings
        self._use_embeddings = True

    def _embed_text(self, text: str) -> List[float]:
        url = f"{self.base_url.rstrip('/')}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text[:8000],
        }
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except requests.RequestException:
            return []

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return {w for w in words if len(w) >= 2}

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _keyword_score(self, query: str, doc_tokens: set[str]) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens or not doc_tokens:
            return 0.0
        return len(query_tokens & doc_tokens) / len(query_tokens)

    def retrieve(self, query, top_k=2):
        if self._use_embeddings:
            query_emb = self._embed_text(query)
            if query_emb:
                scored = [
                    (idx, self._cosine(query_emb, emb))
                    for idx, emb in enumerate(self._doc_embeddings)
                ]
                scored.sort(key=lambda item: item[1], reverse=True)
                return [self.docs[i] for i, _ in scored[:top_k]]

        scored = [
            (idx, self._keyword_score(query, tokens))
            for idx, tokens in enumerate(self._doc_tokens)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [self.docs[i] for i, _ in scored[:top_k]]
