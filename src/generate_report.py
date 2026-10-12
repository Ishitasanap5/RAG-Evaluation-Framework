import numpy as np
import pandas as pd

BASELINE = "chunker_fixed__retriever_dense__rerank_False"


def paired_bootstrap(piv, a, b, n=10000, seed=0):
    d = (piv[a] - piv[b]).to_numpy()
    boots = np.random.default_rng(seed).choice(d, (n, len(d))).mean(axis=1)
    return d.mean(), np.percentile(boots, [2.5, 97.5])


def report():
    df = pd.read_csv("results_with_ci.csv")
    err = pd.read_csv("error_analysis.csv")
    piv = err.pivot(index="query_idx", columns="config", values="hit_at_5")

    print(f"--- RAG EXPERIMENT SUMMARY ({len(piv)} questions) ---")
    print(df.to_string(index=False))

    best = df.loc[df["recall@5"].idxmax()]
    print("\nHighest recall@5 (point estimate; see paired comparison before calling it a winner):")
    print(f"   {best['chunker']} / {best['retriever']} / rerank={best['reranker']}")
    print(f"   recall@5 {best['recall@5']} {best['recall@5_ci']} | mrr {best['mrr']} {best['mrr_ci']}")

    print(f"\nPaired difference in recall@5 vs {BASELINE}  (* = 95% CI excludes 0)")
    for c in piv.columns:
        if c == BASELINE:
            continue
        m, ci = paired_bootstrap(piv, c, BASELINE)
        star = "*" if ci[0] > 0 or ci[1] < 0 else " "
        print(f" {star} {c:52} {m:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    dead = piv.index[piv.sum(axis=1) == 0]
    print(f"\n{len(dead)} of {len(piv)} questions were missed by EVERY config - check these for bad questions/gold spans:")
    q = err.drop_duplicates("query_idx").set_index("query_idx")
    for i in dead:
        print(f"   [{i}] ({q.loc[i, 'type']}) {q.loc[i, 'question']}")


if __name__ == "__main__":
    report()