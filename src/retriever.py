import re
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, chunks: list, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.chunks = chunks
        print(f"🔄 Loading embedding model ({model_name})...")
        self.encoder = SentenceTransformer(model_name)
        
        print("🧠 Building dense FAISS index...")
        texts = [c["chunk_text"] for c in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
        
        self.dimension = embeddings.shape[1]
        self.faiss_index = IndexFlatIP_compat(self.dimension, embeddings)
        
        print("📚 Building sparse BM25 index...")
        # Clean regex tokenization to remove trailing punctuation/newlines
        tokenized_corpus = [re.findall(r"\w+", doc.lower()) for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query: str, top_k: int = 10):
        # BGE recommendation: prefix queries with instruction text
        prefixed_query = f"Represent this sentence for searching: {query}"
        query_vector = self.encoder.encode([prefixed_query], normalize_embeddings=True).astype("float32")
        scores, indices = self.faiss_index.search(query_vector, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1: continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def sparse_search(self, query: str, top_k: int = 10):
        tokenized_query = re.findall(r"\w+", query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0.0:  # Drop zero or negative scores
                break
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    def hybrid_search_rrf(self, query: str, top_k: int = 10, k_param: int = 60):
        fetch_k = max(top_k * 3, 30)
        dense_res = self.dense_search(query, top_k=fetch_k)
        sparse_res = self.sparse_search(query, top_k=fetch_k)
        
        rrf_scores = {}
        chunk_map = {}

        for rank, chunk in enumerate(dense_res):
            c_id = (chunk["doc_id"], chunk["char_start"], chunk["char_end"])
            chunk_map[c_id] = chunk
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (k_param + (rank + 1)))

        for rank, chunk in enumerate(sparse_res):
            c_id = (chunk["doc_id"], chunk["char_start"], chunk["char_end"])
            chunk_map[c_id] = chunk
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (k_param + (rank + 1)))

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for c_id, score in sorted_chunks[:top_k]:
            chunk = chunk_map[c_id].copy()
            chunk["score"] = float(score)
            results.append(chunk)
            
        return results

def IndexFlatIP_compat(dimension, embeddings):
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index