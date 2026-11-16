# Tests for ranking.py (Part 3). Run with `python test_part3.py` or `python -m pytest test_part3.py`.
# Our score and word2vec + cosine are checked against direct implementations of their formulas, BM25 against a value
# computed by hand, and the pipeline against the definition of a conjunctive query.

import math
import random

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

import ranking as rk
from indexing import Bm25Ranker, InvertedIndex

# --------------------------------------------------------------------------------------------------- toy corpus

TITLES = [["men", "denim", "jacket"], ["men", "jean"], ["women", "jacket"], ["men", "jacket"], ["men", "shirt"]]
BODIES = [["blue", "denim", "men"], ["denim", "jacket", "pocket", "men", "men"], ["denim", "men", "gift"],
          ["denim", "denim", "wash"], ["cotton"]]
DOCS = [t + b for t, b in zip(TITLES, BODIES)]
PIDS = ["A", "B", "C", "D", "E"]
RATINGS = [4.0, None, 5.0, 2.0, 3.0]
DISCOUNTS = [50.0, 20.0, None, 0.0, 10.0]
OUT_OF_STOCK = [False, False, True, False, False]
QUERY = ["men", "denim", "jacket"]


def _index():
    return InvertedIndex(PIDS, DOCS)


def _random_corpus(rng, n_docs=60, vocab=25):
    words = [f"w{i}" for i in range(vocab)]
    docs = [[rng.choice(words) for _ in range(rng.randint(3, 30))] for _ in range(n_docs)]
    titles = [doc[: rng.randint(1, 4)] for doc in docs]
    ratings = [None if rng.random() < 0.2 else round(rng.uniform(1, 5), 1) for _ in docs]
    discounts = [None if rng.random() < 0.2 else float(rng.randint(0, 80)) for _ in docs]
    stock = [rng.random() < 0.2 for _ in docs]
    return words, docs, titles, ratings, discounts, stock


# -------------------------------------------------------------------------------------------------------- BM25

def test_bm25_hand_computed():
    # 3 documents, query "a": df = 2, lengths 2, 4, 3 -> avgdl = 3
    index = InvertedIndex(["x", "y", "z"], [["a", "b"], ["a", "a", "c", "d"], ["b", "c", "d"]])
    idf = math.log2(1 + (3 - 2 + 0.5) / (2 + 0.5))
    k1, b = 1.2, 0.75
    expected_x = idf * 1 * (k1 + 1) / (1 + k1 * (1 - b + b * 2 / 3))
    expected_y = idf * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 4 / 3))
    scores = dict(Bm25Ranker(index, k1, b).score(["a"], [0, 1]))
    assert math.isclose(scores[0], expected_x) and math.isclose(scores[1], expected_y)
    # b = 0: no length normalisation, only the saturated term frequency counts
    scores = dict(Bm25Ranker(index, k1, 0.0).score(["a"], [0, 1]))
    assert math.isclose(scores[1] / scores[0], (2 * (k1 + 1) / (2 + k1)) / ((k1 + 1) / (1 + k1)))


# --------------------------------------------------------------------------------------------------- our score

def test_custom_score_equals_formula():
    rng = random.Random(7)
    for _ in range(5):
        words, docs, titles, ratings, discounts, stock = _random_corpus(rng)
        index = InvertedIndex([str(i) for i in range(len(docs))], docs)
        params = rk.CustomScoreParams(title_boost=rng.uniform(0, 2), rating_weight=rng.uniform(0, 1),
                                      discount_weight=rng.uniform(0, 1), out_of_stock_factor=rng.uniform(0.1, 1))
        ranker = rk.CustomRanker(index, titles, ratings, discounts, stock, params)
        mean_rating = np.mean([r for r in ratings if r is not None])
        query = rng.sample(words, 2)
        candidates = index.conjunctive(query)
        bm25 = dict(Bm25Ranker(index).score(query, candidates))
        expected = {}
        for d in candidates:
            title = len(set(query) & set(titles[d])) / 2
            rating = ((ratings[d] if ratings[d] is not None else mean_rating) - 1) / 4
            discount = (discounts[d] if discounts[d] is not None else 0.0) / 100
            factor = params.out_of_stock_factor if stock[d] else 1.0
            expected[d] = (bm25[d] * (1 + params.title_boost * title)
                           * (1 + params.rating_weight * rating + params.discount_weight * discount) * factor)
        got = ranker.score(query, candidates)
        assert [d for d, _ in got] == sorted(expected, key=lambda d: (-expected[d], d))
        assert all(math.isclose(s, expected[d], rel_tol=1e-9) for d, s in got)
        for d, s in got[:3]:
            assert math.isclose(ranker.components(query, d)["score"], s, rel_tol=1e-9)


def test_custom_score_without_weights_is_bm25():
    index = _index()
    params = rk.CustomScoreParams(title_boost=0, rating_weight=0, discount_weight=0, out_of_stock_factor=1)
    ranker = rk.CustomRanker(index, TITLES, RATINGS, DISCOUNTS, OUT_OF_STOCK, params)
    candidates = index.conjunctive(QUERY)
    assert candidates == [0, 1, 2, 3]
    expected = Bm25Ranker(index).score(QUERY, candidates)
    got = ranker.score(QUERY, candidates)
    assert [d for d, _ in got] == [d for d, _ in expected]
    assert all(math.isclose(a[1], b[1]) for a, b in zip(got, expected))


def test_custom_score_factors():
    index = _index()
    ranker = rk.CustomRanker(index, TITLES, RATINGS, DISCOUNTS, OUT_OF_STOCK)
    # title coverage: A has the three terms in the title, B one, C one, D two
    assert [ranker.title_coverage(QUERY, d) for d in range(4)] == [1.0, 1 / 3, 1 / 3, 2 / 3]
    # missing rating -> mean of the rated products (3.5); missing discount -> 0
    assert math.isclose(ranker.mean_rating, 3.5)
    assert math.isclose(ranker.rating[1], (3.5 - 1) / 4) and ranker.discount[2] == 0.0
    assert math.isclose(ranker.rating[0], 0.75) and math.isclose(ranker.discount[0], 0.5)
    # out of stock: the whole score is multiplied by gamma
    comp = ranker.components(QUERY, 2)
    in_stock = rk.CustomRanker(index, TITLES, RATINGS, DISCOUNTS, [False] * 5)
    assert comp["stock"] == 0.5 and math.isclose(comp["score"], 0.5 * in_stock.components(QUERY, 2)["score"])
    # a better rating never lowers the score
    better = rk.CustomRanker(index, TITLES, [4.0, None, 5.0, 4.5, 3.0], DISCOUNTS, OUT_OF_STOCK)
    assert better.components(QUERY, 3)["score"] > ranker.components(QUERY, 3)["score"]


# -------------------------------------------------------------------------------------------- word2vec + cosine

def _toy_vectors():
    wv = KeyedVectors(vector_size=3)
    wv.add_vectors(["men", "denim", "jacket", "jean", "women"],
                   np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32))
    return wv


def test_text_vector_is_the_average():
    wv = _toy_vectors()
    assert np.allclose(rk.text_vector(wv, ["men", "denim", "jacket"]), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(rk.text_vector(wv, ["men", "men", "denim"]), [2 / 3, 1 / 3, 0])       # repetitions count
    assert np.allclose(rk.text_vector(wv, ["men", "unknown"]), [1, 0, 0])                   # unknown words are skipped
    assert np.allclose(rk.text_vector(wv, ["unknown"]), [0, 0, 0])
    weighted = rk.text_vector(wv, ["men", "denim"], {"men": 1.0, "denim": 3.0})
    assert np.allclose(weighted, [0.25, 0.75, 0])


def test_word2vec_cosine_brute_force():
    wv = _toy_vectors()
    docs = [["men", "denim", "jacket"], ["men", "men", "jean"], ["women", "women", "jacket", "unknown"], ["unknown"]]
    ranker = rk.Word2VecRanker(wv, docs)
    query = ["denim", "jacket"]
    q = np.array([0, 0.5, 0.5])
    expected = {}
    for d, terms in enumerate(docs):
        known = [wv[t] for t in terms if t in wv.key_to_index]
        v = np.mean(known, axis=0) if known else np.zeros(3)
        denominator = np.linalg.norm(v) * np.linalg.norm(q)
        expected[d] = float(v @ q / denominator) if denominator else 0.0
    got = ranker.score(query, [0, 1, 2, 3])
    assert [d for d, _ in got] == sorted(expected, key=lambda d: (-expected[d], d))
    assert all(math.isclose(s, expected[d], abs_tol=2e-6) for d, s in got)        # scores have 6 decimals
    assert all(s == 0.0 for _, s in ranker.score(["unknown"], [0, 1]))                        # query without vector
    assert ranker.score(query, []) == []


def test_remove_first_component():
    rng = np.random.default_rng(0)
    wv = KeyedVectors(vector_size=8)
    words = [f"w{i}" for i in range(30)]
    wv.add_vectors(words, (rng.normal(size=(30, 8)) + 2.0).astype(np.float32))                # common direction
    docs = [list(rng.choice(words, size=6)) for _ in range(40)]
    ranker = rk.Word2VecRanker(wv, docs, remove_first_component=True)
    assert np.allclose(ranker.doc_vectors @ ranker.component, 0, atol=1e-5)
    assert abs(float(ranker.query_vector(docs[0]) @ ranker.component)) < 1e-5


def test_training_is_reproducible():
    rng = random.Random(3)
    words = [f"w{i}" for i in range(15)]
    docs = [[rng.choice(words) for _ in range(12)] for _ in range(80)]
    a = rk.train_word2vec(docs, vector_size=10, epochs=3)
    b = rk.train_word2vec(docs, vector_size=10, epochs=3)
    assert a.index_to_key == b.index_to_key and np.array_equal(a.vectors, b.vectors)
    assert rk.stable_hash("denim") == rk.stable_hash("denim") == 1662847616  # CRC32, the same in every session
    model = rk.train_doc2vec(docs, vector_size=10, epochs=3)
    ranker = rk.Doc2VecRanker(model, infer_epochs=5)
    assert np.array_equal(ranker.query_vector(["w1", "w2"]), ranker.query_vector(["w1", "w2"]))
    assert len(ranker.score(["w1", "w2"], [0, 5, 9])) == 3


# ----------------------------------------------------------------------------------------------------- pipeline

def test_all_methods_rank_the_same_and_set():
    corpus = pd.DataFrame({"pid": PIDS, "all_terms": DOCS, "title_terms": TITLES, "average_rating": RATINGS,
                           "discount": DISCOUNTS, "out_of_stock": OUT_OF_STOCK})
    engines = rk.build_engines(corpus, str.split)
    assert set(engines) == set(rk.METHODS)
    expected = {pid for pid, terms in zip(PIDS, DOCS) if set(QUERY) <= set(terms)}           # definition of the AND
    for name, engine in engines.items():
        results = engine.search("men denim jacket")
        assert {pid for pid, _ in results} == expected == {"A", "B", "C", "D"}, name
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), name
        assert engine.search("men denim jacket", top_k=2) == results[:2], name
        assert engine.search("men zzzz") == [] and engine.search("") == [], name            # unknown term, empty query
    # our score: the out-of-stock product (C) is below the three products in stock
    assert [pid for pid, _ in engines["custom"].search("men denim jacket")][-1] == "C"
    try:
        rk.build_engines(corpus, str.split, methods=("pagerank",))
        raise AssertionError("an unknown method must be rejected")
    except ValueError:
        pass


if __name__ == "__main__":
    for _name, _test in sorted(globals().items()):
        if _name.startswith("test_") and callable(_test):
            _test()
            print("ok", _name)
