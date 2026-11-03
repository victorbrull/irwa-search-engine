# Sanity tests for indexing.py and evaluation.py. Run with `pytest test_part2.py` or `python test_part2.py`.
# The metrics are checked against values computed by hand, and the index / TF-IDF against a dense brute-force
# implementation on random corpora.

import math
import random

import numpy as np

from evaluation import (average_precision_at_k, condense, evaluate_query, evaluate_run, f1_at_k, interpolated_pr_curve,
                        ndcg_at_k,
                        precision_at_k, recall_at_k, reciprocal_rank, relevant_set)
from indexing import Bm25Ranker, InvertedIndex, SearchEngine, TfidfRanker

# ---------------------------------------------------------------------------------------------------- metrics

RANKED = ["a", "b", "c", "d", "e"]
QRELS = {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "z": 1}   # z is relevant but never retrieved: R = 4
REL = relevant_set(QRELS)


def test_precision_recall_f1():
    assert math.isclose(precision_at_k(RANKED, REL, 1), 1.0)
    assert math.isclose(precision_at_k(RANKED, REL, 3), 2 / 3)
    assert math.isclose(precision_at_k(RANKED, REL, 5), 3 / 5)
    assert math.isclose(precision_at_k(RANKED, REL, 10), 3 / 10)      # short ranking: the missing positions are misses
    assert math.isclose(recall_at_k(RANKED, REL, 5), 3 / 4)
    p, r = 3 / 5, 3 / 4
    assert math.isclose(f1_at_k(RANKED, REL, 5), 2 * p * r / (p + r))
    assert f1_at_k(["x", "y"], REL, 2) == 0.0


def test_average_precision_and_rr():
    # relevant at positions 1, 3, 5 -> precisions 1, 2/3, 3/5
    assert math.isclose(average_precision_at_k(RANKED, REL), (1 + 2 / 3 + 3 / 5) / 4)
    assert math.isclose(average_precision_at_k(RANKED, REL, 3), (1 + 2 / 3) / 3)      # min(K, R) = 3
    assert math.isclose(average_precision_at_k(RANKED, REL, 5), (1 + 2 / 3 + 3 / 5) / 4)
    assert math.isclose(reciprocal_rank(RANKED, REL), 1.0)
    assert math.isclose(reciprocal_rank(["x", "b", "c"], REL), 1 / 3)
    assert reciprocal_rank(["x", "y"], REL) == 0.0


def test_ndcg():
    dcg = 1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)
    assert math.isclose(ndcg_at_k(RANKED, QRELS, 5), dcg / idcg)
    perfect = ["a", "c", "e", "z", "b", "d"]
    assert math.isclose(ndcg_at_k(perfect, QRELS, 4), 1.0)
    assert ndcg_at_k(RANKED, {"a": 0, "b": 0}, 5) == 0.0                              # no relevant document


def test_perfect_and_empty_rankings():
    qrels = {"a": 1, "b": 1, "c": 0}
    out = evaluate_query(["a", "b", "c"], qrels, ks=(2, 3))
    assert out["P@2"] == out["R@2"] == out["F1@2"] == out["AP@2"] == out["NDCG@2"] == 1.0
    assert out["AP"] == out["RR"] == 1.0
    out = evaluate_query([], qrels, ks=(2,))
    assert all(v == 0.0 for v in out.values())


def test_interpolated_pr_curve():
    recall, precision = interpolated_pr_curve(RANKED, REL)            # relevant at ranks 1, 3, 5 (R = 4)
    assert recall == [1 / 4, 2 / 4, 3 / 4]
    r2, p2 = interpolated_pr_curve(["x", "a", "y", "c"], {"a", "c"})   # raw precisions 1/2, 2/4
    assert r2 == [1 / 2, 1.0] and p2 == [1 / 2, 1 / 2]
    r3, p3 = interpolated_pr_curve(["x", "a", "c"], {"a", "c"})          # raw 1/2, 2/3 -> interpolated 2/3, 2/3
    assert p3 == [2 / 3, 2 / 3]
    assert precision == [1.0, 2 / 3, 3 / 5]                           # raw 1, 2/3, 3/5: already non-increasing


def test_condense_keeps_order():
    assert condense(["u1", "a", "u2", "b", "u3"], {"a", "b", "c"}) == ["a", "b"]


def test_map_and_mrr_are_means_over_queries():
    qrels = {"q1": {"a": 1, "b": 0}, "q2": {"x": 1, "y": 1}}
    run = {"q1": ["b", "a"], "q2": ["x", "y"]}
    table = evaluate_run(run, qrels, ks=(2,))
    assert math.isclose(table.loc["mean", "AP"], (0.5 + 1.0) / 2)       # MAP
    assert math.isclose(table.loc["mean", "RR"], (0.5 + 1.0) / 2)       # MRR


def test_against_scikit_learn_if_available():
    try:
        from sklearn.metrics import average_precision_score, ndcg_score
    except ImportError:
        return
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(3, 25)
        pids = [f"d{i}" for i in range(n)]
        labels = {p: rng.randint(0, 1) for p in pids}
        if sum(labels.values()) == 0:
            labels[pids[0]] = 1
        scores = [rng.random() for _ in pids]
        ranked = [p for _, p in sorted(zip(scores, pids), key=lambda x: -x[0])]
        y_true, y_score = [[labels[p] for p in pids]], [scores]
        for k in (1, 5, n):
            assert math.isclose(ndcg_at_k(ranked, labels, k), ndcg_score(y_true, y_score, k=k), abs_tol=1e-9)
        assert math.isclose(average_precision_at_k(ranked, relevant_set(labels)),
                            average_precision_score(y_true[0], y_score[0]), abs_tol=1e-9)


# ----------------------------------------------------------------------------------------- index and TF-IDF

def _dense_tfidf(term_lists, query_terms):
    """Brute-force cosine TF-IDF with a dense matrix (independent of the sparse implementation)."""
    vocab = sorted({t for terms in term_lists for t in terms})
    pos = {t: i for i, t in enumerate(vocab)}
    n = len(term_lists)
    tf = np.zeros((n, len(vocab)))
    for d, terms in enumerate(term_lists):
        for t in terms:
            tf[d, pos[t]] += 1
    df = (tf > 0).sum(axis=0)
    idf = np.log2(n / df)
    w = np.where(tf > 0, 1 + np.log2(np.where(tf > 0, tf, 1)), 0) * idf
    q = np.zeros(len(vocab))
    for t in query_terms:
        if t in pos:
            q[pos[t]] += 1
    qw = np.where(q > 0, 1 + np.log2(np.where(q > 0, q, 1)), 0) * idf
    norms = np.linalg.norm(w, axis=1)
    scores = np.zeros(n)
    for d in range(n):
        if norms[d] > 0 and np.linalg.norm(qw) > 0:
            scores[d] = (w[d] @ qw) / (norms[d] * np.linalg.norm(qw))
    return scores


def test_toy_corpus():
    docs = {"d0": "red cotton shirt", "d1": "blue cotton jean", "d2": "blue jean slim jean", "d3": "red shirt"}
    index = InvertedIndex(list(docs), [t.split() for t in docs.values()])
    engine = SearchEngine(index, lambda q: q.split())
    results = engine.search("blue jean")
    assert [p for p, _ in results] == ["d1", "d2"]                    # AND: only d1 and d2 contain both terms
    assert engine.search("blue shirt") == []                          # nothing contains both
    assert engine.search("blue unknownterm") == []                    # out-of-vocabulary term -> empty with AND
    assert math.isclose(index.idf("slim"), math.log2(4 / 1))
    dense = _dense_tfidf([t.split() for t in docs.values()], ["blue", "jean"])
    for pid, score in results:
        assert math.isclose(score, dense[list(docs).index(pid)])


def test_random_corpora_match_brute_force():
    rng = random.Random(1)
    vocab = [f"t{i}" for i in range(30)]
    for _ in range(20):
        term_lists = [[rng.choice(vocab) for _ in range(rng.randint(1, 12))] for _ in range(60)]
        pids = [f"p{i}" for i in range(60)]
        index = InvertedIndex(pids, term_lists)
        ranker = TfidfRanker(index)
        query = [rng.choice(vocab) for _ in range(rng.randint(1, 3))]
        expected_docs = [d for d, terms in enumerate(term_lists) if set(query) <= set(terms)]
        assert index.conjunctive(query) == expected_docs
        dense = _dense_tfidf(term_lists, query)
        for doc_id, score in ranker.score(query, expected_docs):
            assert math.isclose(score, dense[doc_id], abs_tol=1e-12)


def test_bm25_matches_brute_force():
    rng = random.Random(2)
    vocab = [f"t{i}" for i in range(25)]
    for _ in range(10):
        term_lists = [[rng.choice(vocab) for _ in range(rng.randint(1, 15))] for _ in range(50)]
        index = InvertedIndex([f"p{i}" for i in range(50)], term_lists)
        ranker = Bm25Ranker(index, k1=1.2, b=0.75)
        query = [rng.choice(vocab) for _ in range(rng.randint(1, 3))]
        n, avgdl = len(term_lists), sum(map(len, term_lists)) / len(term_lists)
        for doc_id, score in ranker.score(query, list(range(n))):
            expected = 0.0
            for t in set(query):
                df = sum(1 for terms in term_lists if t in terms)
                tf = term_lists[doc_id].count(t)
                if df and tf:
                    idf = math.log2(1 + (n - df + 0.5) / (df + 0.5))
                    expected += idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * len(term_lists[doc_id]) / avgdl))
            assert math.isclose(score, expected, abs_tol=1e-12)


def test_min_match_is_a_soft_and():
    docs = {"d0": "red cotton shirt", "d1": "blue cotton jean", "d2": "blue jean slim", "d3": "red shirt"}
    index = InvertedIndex(list(docs), [t.split() for t in docs.values()])
    engine = SearchEngine(index, lambda q: q.split())
    assert {p for p, _ in engine.search("blue cotton jean")} == {"d1"}                     # AND
    assert {p for p, _ in engine.search("blue cotton jean", min_match=2)} == {"d1", "d2"}         # at least 2 of 3 terms (d0 has only 'cotton')
    assert {p for p, _ in engine.search("blue cotton jean", min_match=3)} == {"d1"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
