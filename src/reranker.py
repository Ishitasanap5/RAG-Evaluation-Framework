from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initializes the cross-encoder reranker model.
        """
        print(f"Loading cross-encoder reranker model ({model_name})...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list, top_k: int = 5):
        """
        Takes a list of retrieved chunks, scores them against the query using 
        the cross-encoder, and returns the top_k re-sorted chunks.
        """
        if not chunks:
            return []

        # Prepare pairs of [query, chunk_text] for the cross-encoder
        pairs = [[query, chunk["chunk_text"]] for chunk in chunks]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Attach scores to chunks
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
            
        # Sort descending by rerank score
        reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_chunks[:top_k]