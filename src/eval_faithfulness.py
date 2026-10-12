import json
import ollama

JUDGE_MODEL = "llama3.2"   # a 3B judge is noisy; use llama3.1:8b or qwen2.5:7b if you can


def check_faithfulness(answer: str, retrieved_chunks: list, model: str = JUDGE_MODEL):
    """Return a score in [0, 1], or None if the judge failed. Never fabricates a score."""
    context = "\n\n".join(f"[Chunk {i}] {c['chunk_text']}" for i, c in enumerate(retrieved_chunks, start=1))
    prompt = f"""You are a strict judge. Is every factual claim in the ANSWER supported by the CONTEXT?
Score 1.0 if all claims are supported, 0.5 if some are unsupported, 0.0 if mostly unsupported or invented.
Return JSON: {{"score": <number between 0 and 1>}}

CONTEXT:
{context}

ANSWER:
{answer}
"""
    try:
        res = ollama.chat(model=model, format="json", options={"temperature": 0},
                          messages=[{"role": "user", "content": prompt}])
        score = float(json.loads(res["message"]["content"])["score"])
        return min(1.0, max(0.0, score))
    except Exception as ex:
        print(f"   [judge failed] {ex!r}")
        return None


def mean_faithfulness(scores: list):
    valid = [s for s in scores if s is not None]
    print(f"judge succeeded on {len(valid)}/{len(scores)} answers")
    return sum(valid) / len(valid) if valid else None