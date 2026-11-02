# Independent check of the reported numbers (second implementation of the metrics, plain NumPy, different code path).
# It rebuilds the index, runs the seven queries and recomputes every metric of results/validation_metrics.csv and
# results/test_queries_metrics.csv, using relevance_judgments.csv for the labels of our own queries.
# Run from this folder after the notebook:  python check_metrics.py
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "project_progress", "part_1"))
sys.path.insert(0, HERE)
import preprocessing as pp                                    # Part 1, unchanged
from indexing import InvertedIndex, SearchEngine

corpus = pd.read_pickle(os.path.join(ROOT, "data", "processed_corpus.pkl"))
labels = pd.read_csv(os.path.join(ROOT, "data", "validation_labels.csv"))
engine = SearchEngine(InvertedIndex(corpus.pid.tolist(), corpus.all_terms.tolist()), pp.preprocess_text)


def metrics(rel, total_relevant, ks=(5, 10, 20)):
    """rel: 0/1 array of the ranking in rank order; total_relevant: R."""
    out, n = {}, len(rel)
    cum, ranks = np.cumsum(rel), np.arange(1, n + 1)
    prec = cum / ranks
    for k in ks:
        hits = cum[min(k, n) - 1] if n else 0
        p, r = hits / k, hits / total_relevant
        out.update({f"P@{k}": p, f"R@{k}": r, f"F1@{k}": 0 if p + r == 0 else 2 * p * r / (p + r),
                    f"AP@{k}": (prec[:k] * rel[:k]).sum() / min(k, total_relevant)})
        dcg = (rel[:k] / np.log2(np.arange(2, min(k, n) + 2))).sum()
        ideal = np.zeros(k)
        ideal[:min(k, total_relevant)] = 1
        out[f"NDCG@{k}"] = dcg / (ideal / np.log2(np.arange(2, k + 2))).sum()
    out["AP"] = (prec * rel).sum() / total_relevant
    out["RR"] = 1 / (np.argmax(rel) + 1) if rel.any() else 0.0
    return out


def compare(recomputed, saved, name):
    bad = [(k, v, saved[k]) for k, v in recomputed.items() if abs(v - saved[k]) > 5e-4]
    print(f"{name}: {'ok' if not bad else 'MISMATCH ' + str(bad)}")
    return not bad


ok = True
saved_val = pd.read_csv(os.path.join(HERE, "results", "validation_metrics.csv"), index_col=0)
saved_val.index = saved_val.index.astype(str)
for q, text in {1: "women full sleeve sweatshirt cotton", 2: "men slim jeans blue"}.items():
    judged = labels[labels.query_id == q].set_index("pid").labels
    ranking = [pid for pid, _ in engine.search(text) if pid in judged.index]            # condensed list
    ok &= compare(metrics(np.array([judged[p] for p in ranking]), int(judged.sum())), saved_val.loc[str(q)], f"validation query {q}")

judgments = pd.read_csv(os.path.join(HERE, "relevance_judgments.csv"))
saved_test = pd.read_csv(os.path.join(HERE, "results", "test_queries_metrics.csv"), index_col=0)
queries = pd.read_csv(os.path.join(HERE, "results", "test_queries.csv")).set_index("id")["query"].to_dict()
for tid, text in queries.items():
    d = judgments[judgments.query_id == tid]
    relevant = set(d[d.label == 1].pid)
    ranking = [pid for pid, _ in engine.search(text)]
    assert set(d[d.returned_by_AND].pid) == set(ranking), "the judgments file does not list the AND results"
    ok &= compare(metrics(np.array([int(p in relevant) for p in ranking]), len(relevant)), saved_test.loc[tid], f"test query {tid}")
print("ALL MATCH" if ok else "SOME MISMATCH")
