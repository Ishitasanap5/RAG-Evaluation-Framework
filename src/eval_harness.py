import numpy as np

class EvalHarness:
    @staticmethod
    def check_span_overlap(retrieved_chunk: dict, gold_doc_id: str, gold_start: int, gold_end: int, min_fraction: float = 0.5) -> bool:
        """
        Fairer overlap rule: A retrieved chunk is a hit if it matches doc_id 
        and its intersection with the gold span covers at least `min_fraction` (50%) 
        of the smaller of the two spans (gold length vs chunk length).
        """
        if retrieved_chunk["doc_id"] != gold_doc_id:
            return False
        
        r_start = retrieved_chunk["char_start"]
        r_end = retrieved_chunk["char_end"]
        
        overlap_start = max(r_start, gold_start)
        overlap_end = min(r_end, gold_end)
        
        intersection = max(0, overlap_end - overlap_start)
        gold_length = max(1, gold_end - gold_start)
        chunk_length = max(1, r_end - r_start)
        
        return intersection >= min_fraction * min(gold_length, chunk_length)

    @staticmethod
    def evaluate_retrieval(retrieved_results_per_query: list, eval_dataset: list, k_values=[1, 3, 5, 10]):
        recalls = {k: [] for k in k_values}
        mrr_list = []
        type_recalls = {}

        for q_item, retrieved_chunks in zip(eval_dataset, retrieved_results_per_query):
            g_doc = q_item["gold_doc_id"]
            g_start = q_item["gold_start"]
            g_end = q_item["gold_end"]
            q_type = q_item.get("type", "general")

            if q_type not in type_recalls:
                type_recalls[q_type] = {k: [] for k in k_values}

            first_hit_rank = None
            for rank, chunk in enumerate(retrieved_chunks, start=1):
                if EvalHarness.check_span_overlap(chunk, g_doc, g_start, g_end):
                    first_hit_rank = rank
                    break

            mrr_list.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

            for k in k_values:
                top_k_chunks = retrieved_chunks[:k]
                hit_in_k = any(EvalHarness.check_span_overlap(c, g_doc, g_start, g_end) for c in top_k_chunks)
                recalls[k].append(1.0 if hit_in_k else 0.0)
                type_recalls[q_type][k].append(1.0 if hit_in_k else 0.0)

        results = {f"recall@{k}": recalls[k] for k in k_values}
        results["mrr_raw"] = mrr_list
        results["type_breakdown"] = type_recalls
        return results

    @staticmethod
    def bootstrap_confidence_interval(metric_values, n_bootstrap=1000, confidence=0.95):
        values = np.array(metric_values)
        if len(values) == 0:
            return 0.0, 0.0, 0.0
        
        boot_means = []
        n = len(values)
        rng = np.random.default_rng(42)
        
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=n, replace=True)
            boot_means.append(np.mean(sample))
            
        alpha = (1.0 - confidence) / 2.0
        lower = np.percentile(boot_means, alpha * 100)
        upper = np.percentile(boot_means, (1.0 - alpha) * 100)
        
        return float(np.mean(values)), float(lower), float(upper)