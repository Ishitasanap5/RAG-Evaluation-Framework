import json
from pathlib import Path
import pandas as pd

from chunkers import fixed_size_chunker, recursive_chunker, heading_based_chunker
from retriever import HybridRetriever
from reranker import Reranker
from eval_harness import EvalHarness

def run_pipeline_experiment():
    print("🚀 Loading processed corpus for experiments...")
    processed_dir = Path("data/processed")
    documents = []
    
    for json_file in processed_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            documents.append(json.load(f))

    eval_path = Path("eval_set.json")
    if not eval_path.exists():
        print("⚠️ eval_set.json not found! Run generate_eval_set.py first.")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    chunker_configs = [
        ("fixed", lambda text, doc_id: fixed_size_chunker(text, doc_id, chunk_size=512, overlap=64)),
        ("recursive", lambda text, doc_id: recursive_chunker(text, doc_id, max_size=512, overlap=64)),
        ("heading", lambda text, doc_id: heading_based_chunker(text, doc_id))
    ]

    retriever_types = ["dense", "hybrid"]
    use_reranker_options = [False, True]

    results_table = []
    error_analysis_logs = []

    print("\n🔬 Starting Fully Optimized Experiment Grid Sweep...")
    
    # Load reranker once globally outside loops
    global_reranker = Reranker()

    for strategy_name, chunker_func in chunker_configs:
        print(f"\n--- Running Chunker Strategy: {strategy_name} ---")
        
        all_chunks = []
        for doc in documents:
            chunks = chunker_func(doc["full_text"], doc["doc_id"])
            all_chunks.extend(chunks)
            
        print(f"   Generated {len(all_chunks)} total chunks.")
        retriever = HybridRetriever(all_chunks)

        for ret_type in retriever_types:
            for use_rerank in use_reranker_options:
                config_name = f"chunker_{strategy_name}__retriever_{ret_type}__rerank_{use_rerank}"
                print(f"   Evaluating config: {config_name}")

                retrieved_results_per_query = []
                for idx, item in enumerate(eval_dataset):
                    q = item["question"]
                    
                    # Fetch wider candidate pool before reranking down to 10
                    if ret_type == "dense":
                        candidates = retriever.dense_search(q, top_k=30)
                    else:
                        candidates = retriever.hybrid_search_rrf(q, top_k=30)

                    if use_rerank and global_reranker:
                        final_chunks = global_reranker.rerank(q, candidates, top_k=10)
                    else:
                        final_chunks = candidates[:10]

                    retrieved_results_per_query.append(final_chunks)

                    is_hit_5 = any(EvalHarness.check_span_overlap(c, item["gold_doc_id"], item["gold_start"], item["gold_end"]) for c in final_chunks[:5])
                    error_analysis_logs.append({
                        "config": config_name,
                        "query_idx": idx,
                        "question": q,
                        "type": item.get("type", "general"),
                        "hit_at_5": int(is_hit_5)
                    })

                raw_metrics = EvalHarness.evaluate_retrieval(retrieved_results_per_query, eval_dataset, k_values=[1, 3, 5, 10])
                
                r5_mean, r5_low, r5_high = EvalHarness.bootstrap_confidence_interval(raw_metrics["recall@5"])
                r10_mean, r10_low, r10_high = EvalHarness.bootstrap_confidence_interval(raw_metrics["recall@10"])
                mrr_mean, mrr_low, mrr_high = EvalHarness.bootstrap_confidence_interval(raw_metrics["mrr_raw"])

                # Extract per-type mean hit rates from error analysis for this config
                df_errors = pd.DataFrame(error_analysis_logs)
                config_errors = df_errors[df_errors["config"] == config_name]
                type_means = config_errors.groupby("type")["hit_at_5"].mean().to_dict()

                row = {
                    "chunker": strategy_name,
                    "retriever": ret_type,
                    "reranker": use_rerank,
                    "recall@5": round(r5_mean, 3),
                    "recall@5_ci": f"[{round(r5_low, 3)}, {round(r5_high, 3)}]",
                    "recall@10": round(r10_mean, 3),
                    "recall@10_ci": f"[{round(r10_low, 3)}, {round(r10_high, 3)}]",
                    "mrr": round(mrr_mean, 3),
                    "mrr_ci": f"[{round(mrr_low, 3)}, {round(mrr_high, 3)}]",
                    "exact_term_recall@5": round(type_means.get("exact-term", 0.0), 3),
                    "conceptual_recall@5": round(type_means.get("conceptual", 0.0), 3)
                }
                results_table.append(row)

    df = pd.DataFrame(results_table)
    df.to_csv("results_with_ci.csv", index=False)
    
    df_errors = pd.DataFrame(error_analysis_logs)
    df_errors.to_csv("error_analysis.csv", index=False)

    print("\n🎉 Fully optimized experiments completed! Results saved to 'results_with_ci.csv' and 'error_analysis.csv'.")
    print(df[['chunker', 'retriever', 'reranker', 'recall@5', 'recall@10', 'mrr', 'exact_term_recall@5', 'conceptual_recall@5']])

if __name__ == "__main__":
    run_pipeline_experiment()