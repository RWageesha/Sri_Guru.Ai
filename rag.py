import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGRetriever:
    def __init__(self, data_dir="data", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs, self.doc_texts = self._load_docs(data_dir)
        self.index, self.embeddings = self._build_index(self.doc_texts)

    def _load_docs(self, data_dir):
        docs = []
        doc_texts = []
        for fname in os.listdir(data_dir):
            with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
                text = f.read()
                docs.append({"filename": fname, "text": text})
                doc_texts.append(text)
        return docs, doc_texts

    def _build_index(self, doc_texts):
        embeddings = self.model.encode(doc_texts, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings

    def retrieve(self, query, top_k=2):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        return [self.docs[i] for i in I[0]]
