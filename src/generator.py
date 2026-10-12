import ollama


class RAGGenerator:
    def __init__(self, model_name: str = "llama3.2"):
        """Local generation via ollama. Pull the model first: `ollama pull llama3.2`."""
        self.model_name = model_name

    def generate_answer(self, query: str, retrieved_chunks: list) -> str:
        context = ""
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context += f"\n--- [Chunk {i}] (Doc: {chunk['doc_id']}) ---\n{chunk['chunk_text']}\n"

        prompt = f"""You are a precise technical documentation assistant. Answer the question using ONLY the context chunks below.
If the answer is not in the context, say you do not know. Cite chunk numbers like [Chunk 1] for every fact.

CONTEXT CHUNKS:
{context}

QUESTION:
{query}

ANSWER:"""
        # No silent fallback: if ollama is down or the model is missing, this raises.
        res = ollama.chat(model=self.model_name, options={"temperature": 0},
                          messages=[{"role": "user", "content": prompt}])
        return res["message"]["content"].strip()