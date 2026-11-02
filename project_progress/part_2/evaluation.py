# Evaluation metrics for ranked retrieval (Part 2): P@K, R@K, AP@K, F1@K, MAP, MRR and NDCG@K.
#
# Conventions used everywhere in this file
#   - A ranking is a list of document ids (pids), best first.
#   - Relevance judgments ("qrels") are a dict {pid: label} with binary labels (1 = relevant, 0 = not relevant).
#     A document that is not in the dict has not been judged and is treated as NOT relevant.
#   - R is the number of relevant documents of the query in the judgments (not in the ranking).

import math
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


def relevant_set(qrels: Dict[str, int]) -> set:
    """
    :param qrels: {pid: label} of one query
    :return: set of pids with label > 0
    """
    return {pid for pid, label in qrels.items() if label > 0}


def condense(ranked: Sequence[str], judged: Iterable[str]) -> List[str]:
    """
    "Condensed list" (Sakai, 2007): remove from a ranking every document that has no judgment, keeping
    the order of the rest. Use it when only a sample of the collection was judged, so that unjudged documents
    are not counted as non-relevant.
    :param ranked: ranking (best first)
    :param judged: pids that have a judgment
    :return: ranking restricted to judged documents
    """
    judged = set(judged)
    return [pid for pid in ranked if pid in judged]


def precision_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    """
    P@K = (number of relevant documents in the first K positions) / K.
    If the ranking has fewer than K documents the denominator is still K (missing positions are misses).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for pid in ranked[:k] if pid in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    """
    R@K = (number of relevant documents in the first K positions) / R. It is 0 if the query has no relevant document.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return sum(1 for pid in ranked[:k] if pid in relevant) / len(relevant)


def f1_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    """
    F1@K = 2 * P@K * R@K / (P@K + R@K) (harmonic mean), 0 when both are 0.
    """
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def average_precision_at_k(ranked: Sequence[str], relevant: set, k: Optional[int] = None) -> float:
    """
    AP@K = 1 / min(K, R) * sum_{i=1..K} P@i * rel_i , where rel_i is 1 if the document at position i is relevant.
    The sum only has terms at the positions of relevant documents. Dividing by min(K, R) (and not by K) means that
    a perfect ranking has AP@K = 1 even when R < K or R > K.
    With k=None the whole ranking is used and the denominator is R: this is the classic Average Precision
    (the mean of AP over queries is the MAP).
    """
    if not relevant:
        return 0.0
    cutoff = len(ranked) if k is None else k
    hits, total = 0, 0.0
    for i, pid in enumerate(ranked[:cutoff], start=1):
        if pid in relevant:
            hits += 1
            total += hits / i
    denominator = len(relevant) if k is None else min(k, len(relevant))
    return total / denominator


def reciprocal_rank(ranked: Sequence[str], relevant: set) -> float:
    """
    RR = 1 / (position of the first relevant document), 0 if there is none. The mean over queries is the MRR.
    """
    for i, pid in enumerate(ranked, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """
    DCG@K = sum_{i=1..K} gain_i / log2(i + 1)  (linear gain; with binary labels it equals the (2^rel - 1) version).
    """
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(ranked: Sequence[str], qrels: Dict[str, int], k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K. IDCG@K is the DCG@K of the ideal ranking, i.e. all the judged documents sorted by
    decreasing label. 0 if the query has no relevant document.
    """
    gains = [qrels.get(pid, 0) for pid in ranked]
    ideal = sorted(qrels.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg_at_k(gains, k) / idcg


def interpolated_pr_curve(ranked: Sequence[str], relevant: set):
    """
    Interpolated precision-recall curve (used only for the figures). One point per relevant document found:
    recall = hits / R and precision = hits / rank; interpolated precision at recall r is the maximum precision at any
    recall >= r, so the curve never increases with recall.
    :return: (recall values, interpolated precision values)
    """
    hits, recall, precision = 0, [], []
    for i, pid in enumerate(ranked, start=1):
        if pid in relevant:
            hits += 1
            recall.append(hits / len(relevant))
            precision.append(hits / i)
    for j in range(len(precision) - 2, -1, -1):
        precision[j] = max(precision[j], precision[j + 1])
    return recall, precision


def evaluate_query(ranked: Sequence[str], qrels: Dict[str, int], ks: Sequence[int] = (5, 10, 20)) -> Dict[str, float]:
    """
    All the per-query metrics.
    :param ranked: ranking of the query (best first)
    :param qrels: {pid: label} of the query
    :param ks: cut-offs
    :return: dict with P@k, R@k, F1@k, AP@k and NDCG@k for every k, plus AP (whole ranking) and RR
    """
    relevant = relevant_set(qrels)
    out = {}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"R@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"F1@{k}"] = f1_at_k(ranked, relevant, k)
        out[f"AP@{k}"] = average_precision_at_k(ranked, relevant, k)
        out[f"NDCG@{k}"] = ndcg_at_k(ranked, qrels, k)
    out["AP"] = average_precision_at_k(ranked, relevant)
    out["RR"] = reciprocal_rank(ranked, relevant)
    return out


def evaluate_run(run: Dict, qrels: Dict, ks: Sequence[int] = (5, 10, 20)) -> pd.DataFrame:
    """
    Evaluate a whole run (several queries). Rows = queries, plus a final row "mean" with the average over queries:
    the mean of AP is the MAP, the mean of RR is the MRR, and the mean of AP@k / NDCG@k / ... are their macro averages.
    :param run: {query_id: ranking}
    :param qrels: {query_id: {pid: label}}
    :param ks: cut-offs
    :return: DataFrame (index = query ids + "mean", columns = metrics)
    """
    rows = {qid: evaluate_query(run[qid], qrels[qid], ks) for qid in qrels}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.loc["mean"] = table.mean()
    return table
