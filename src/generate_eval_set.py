import json
import os
import random
import re
from collections import Counter
from pathlib import Path

GEN_MODEL = os.environ.get("EVAL_GEN_MODEL", "llama3.2")   # try llama3.1:8b / qwen2.5:7b if your machine can
MAX_PER_DOC = 3

BANNED = re.compile(
    r"\b(snippet|passage|provided|given|above|below|the text|this text|this document|"
    r"this project|this section|this example|json|the documentation|search engine|"
    r"this|these|those|that function|the function|a function|the model)\b", re.I)

STOP_LEN = 5

TERM_RE = re.compile(
    r"`([^`]+)`|(--[\w-]+)|\b([A-Za-z]+_\w+)\b|\b([a-z]+[A-Z]\w*|[A-Z][a-z]+[A-Z]\w*)\b|\b(\w+\(\))")


def candidate_paragraphs(text: str, lo: int = 250, hi: int = 900):
    """Prose paragraphs (outside code fences) -> (start, end). Gold spans are real paragraphs, not random windows."""
    fences = [(m.start(), m.end()) for m in re.finditer(r"```[\s\S]*?```", text)]
    out = []
    for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
        para = m.group().rstrip()
        s, e = m.start(), m.start() + len(para)
        if not (lo <= len(para) <= hi):
            continue
        if any(s < fe and e > fs for fs, fe in fences):
            continue
        if para.startswith(("```", "|", "<", "!", "#", "---", "///", "//")):
            continue
        if "{*" in para or "![" in para or para.count("|") > 4 or para.count("<") > 3:
            continue
        if sum(c.isalpha() or c == " " for c in para) / len(para) < 0.75:
            continue
        out.append((s, e))
    return out


def valid_question(q: str) -> bool:
    return 25 <= len(q) <= 200 and q.endswith("?") and not BANNED.search(q)


def specific_enough(question: str, para: str, doc_freq, max_df: int = 8) -> bool:
    """Reject generic questions: at least one word from the question must be in the gold paragraph
    AND rare in the corpus (appears in <= max_df docs), so retrieval has something to latch onto."""
    p = para.lower()
    toks = set(re.findall(r"[a-z_][a-z_0-9]{%d,}" % (STOP_LEN - 1), question.lower()))
    return any(t in p and doc_freq(t) <= max_df for t in toks)


def classify(question: str, para: str, doc_freq) -> str:
    """exact-term = question contains a literal identifier/command that is in the gold paragraph AND rare in the corpus."""
    for m in TERM_RE.finditer(question):
        term = next(g for g in m.groups() if g).lower()
        if term in para.lower() and doc_freq(term) <= 5:
            return "exact-term"
    return "conceptual"


def generate_synthetic_eval_set(output_path: str = "eval_set.json", target_count: int = 80):
    import ollama  # lazy import

    random.seed(42)
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path("data/processed").glob("*.json"))]
    if not docs:
        print("No processed files found!")
        return

    lowered = [d["full_text"].lower() for d in docs]
    df_cache = {}
    def doc_freq(term):
        if term not in df_cache:
            df_cache[term] = sum(term in t for t in lowered)
        return df_cache[term]

    pool = [(d["doc_id"], s, e, d["full_text"][s:e]) for d in docs for s, e in candidate_paragraphs(d["full_text"])]
    random.shuffle(pool)
    print(f"{len(docs)} docs, {len(pool)} candidate paragraphs. Generating up to {target_count} questions with {GEN_MODEL}...")

    items, seen, per_doc, rejects = [], set(), Counter(), Counter()
    for doc_id, s, e, para in pool:
        if len(items) >= target_count:
            break
        if per_doc[doc_id] >= MAX_PER_DOC:
            continue

        prompt = f"""You write evaluation questions for a documentation search engine.
Read the passage and write ONE question that a user could ask WITHOUT having seen the passage, and that this passage answers.
Rules: be self-contained; name the specific feature, command or concept instead of using pronouns;
never say "the passage", "the text", "snippet", "provided" or "given"; not a yes/no question; under 25 words.
Return JSON: {{"question": "..."}}

PASSAGE:
{para}
"""
        try:
            res = ollama.chat(model=GEN_MODEL, format="json", options={"temperature": 0.3},
                              messages=[{"role": "user", "content": prompt}])
            q = json.loads(res["message"]["content"]).get("question", "").strip()
        except Exception as ex:
            rejects["call/parse error"] += 1
            if rejects["call/parse error"] <= 3:
                print(f"   [x] {doc_id}: {ex!r}")
            continue

        key = re.sub(r"\W+", " ", q.lower()).strip()
        if not valid_question(q):
            rejects["failed validation"] += 1
        elif not specific_enough(q, para, doc_freq):
            rejects["too generic"] += 1
        elif key in seen:
            rejects["duplicate"] += 1
        else:
            seen.add(key)
            per_doc[doc_id] += 1
            items.append({"question": q, "type": classify(q, para, doc_freq),
                          "gold_doc_id": doc_id, "gold_start": s, "gold_end": e})
            print(f"   [+] {len(items)}/{target_count} {doc_id}: {q}")

    Path(output_path).write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"\nWrote {len(items)} questions to {output_path}")
    print("types:", dict(Counter(i['type'] for i in items)), "| docs:", len(per_doc), "| rejected:", dict(rejects))


if __name__ == "__main__":
    generate_synthetic_eval_set()