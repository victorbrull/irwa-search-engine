# Ranking functions of Part 3: our own score, word2vec + cosine, and the representations compared with word2vec.
#
# The retrieval pipeline is the one of Part 2 (project_progress/part_2/indexing.py, used WITHOUT any change):
#     query -> preprocess_text (Part 1) -> conjunctive (AND) retrieval on the inverted index -> ranking of the matches.
# TF-IDF + cosine (`TfidfRanker`) and BM25 (`Bm25Ranker`) are already implemented in Part 2 and are imported from there.
# This file adds the rankers that are new in Part 3. All of them have the same interface as the Part 2 rankers,
#     ranker.score(query_terms, candidates) -> [(doc_id, score), ...]   sorted by decreasing score, ties by doc id,
# so any of them can be given to the Part 2 `SearchEngine`.

import os
import sys
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _part in ("part_1", "part_2"):                      # make the Part 1 and Part 2 modules importable from anywhere
    _path = os.path.normpath(os.path.join(_HERE, "..", _part))
    if _path not in sys.path:
        sys.path.append(_path)

from indexing import Bm25Ranker, InvertedIndex, SearchEngine, TfidfRanker  # noqa: E402  (Part 2, unchanged)

Ranking = List[Tuple[int, float]]

# Names accepted by `build_engines` / `search.py --method`.
METHODS = ("tfidf", "bm25", "custom", "word2vec")
METHOD_NAMES = {
    "tfidf": "TF-IDF + cosine",
    "bm25": "BM25",
    "custom": "Our score",
    "word2vec": "word2vec + cosine",
}


def _sort(scored: Sequence[Tuple[int, float]]) -> Ranking:
    """Decreasing score; ties are broken by doc id so that every ranking is deterministic (same rule as Part 2)."""
    return sorted(scored, key=lambda x: (-x[1], x[0]))


# ------------------------------------------------------------------------------------------------- our score

@dataclass(frozen=True)
class CustomScoreParams:
    """
    Parameters of our score (see `CustomRanker`). The defaults are the values used in the report.
    :param title_boost: lambda, weight of the share of query terms that appear in the title
    :param rating_weight: alpha, weight of the average rating (scaled to [0, 1])
    :param discount_weight: beta, weight of the discount (scaled to [0, 1])
    :param out_of_stock_factor: gamma, the score of a product that cannot be bought is multiplied by this factor
    :param k1: BM25 term-frequency saturation
    :param b: BM25 length normalisation
    """
    title_boost: float = 1.0
    rating_weight: float = 0.5
    discount_weight: float = 0.25
    out_of_stock_factor: float = 0.5
    k1: float = 1.2
    b: float = 0.75


class CustomRanker:
    """
    Our score: text relevance, corrected by where the query terms appear, by the quality of the offer and by its
    availability.

        score(q, d) = BM25(q, d) * (1 + lambda * title(q, d)) * (1 + alpha * rating(d) + beta * discount(d)) * stock(d)

        title(q, d)  = (number of distinct query terms that appear in the title of d) / (number of distinct query terms)
        rating(d)    = (average_rating - 1) / 4          in [0, 1]; a product without rating gets the collection mean
        discount(d)  = discount / 100                    in [0, 1]; a product without discount information gets 0
        stock(d)     = 1 if the product is in stock, gamma (< 1) if it is out of stock

    The text part decides which products are about the query; the numeric fields can only rescale that score inside
    a bounded range (at most a factor 1 + alpha + beta), so a popular product cannot overtake a much better match.
    """

    def __init__(self, index: InvertedIndex, title_terms: Sequence[Sequence[str]], ratings: Sequence[Optional[float]],
                 discounts: Sequence[Optional[float]], out_of_stock: Sequence[bool],
                 params: CustomScoreParams = CustomScoreParams()):
        """
        :param index: inverted index of Part 2 (built on the merged field `all_terms`)
        :param title_terms: pre-processed title of every document (same order as the index)
        :param ratings: average rating (1..5) of every document, None / NaN when the product has no rating
        :param discounts: discount in % (0..100) of every document, None / NaN when it is unknown
        :param out_of_stock: True when the product is out of stock
        :param params: weights of the formula
        """
        if not (len(title_terms) == len(ratings) == len(discounts) == len(out_of_stock) == index.num_docs):
            raise ValueError("every field must have one value per document of the index")
        self.index = index
        self.params = params
        self.bm25 = Bm25Ranker(index, k1=params.k1, b=params.b)
        self.title_sets = [frozenset(terms) for terms in title_terms]

        rating = np.array([np.nan if r is None else r for r in ratings], dtype=float)
        self.rating_missing = np.isnan(rating)
        self.mean_rating = float(np.nanmean(rating)) if (~self.rating_missing).any() else 3.0
        self.rating = (np.where(self.rating_missing, self.mean_rating, rating) - 1.0) / 4.0

        discount = np.array([np.nan if d is None else d for d in discounts], dtype=float)
        self.discount = np.where(np.isnan(discount), 0.0, discount) / 100.0

        self.stock = np.where(np.asarray(out_of_stock, dtype=bool), params.out_of_stock_factor, 1.0)
        # part of the score that does not depend on the query: computed once
        self.offer = (1.0 + params.rating_weight * self.rating + params.discount_weight * self.discount) * self.stock

    def title_coverage(self, query_terms: Sequence[str], doc_id: int) -> float:
        """Share of the distinct query terms that appear in the title of the document (0..1)."""
        unique = set(query_terms)
        return len(unique & self.title_sets[doc_id]) / len(unique) if unique else 0.0

    def components(self, query_terms: Sequence[str], doc_id: int) -> Dict[str, float]:
        """
        Every factor of the score of one document (used to explain a ranking).
        :return: dict with bm25, title, rating, discount, stock and the final score
        """
        bm25 = dict(self.bm25.score(query_terms, [doc_id]))[doc_id]
        title = self.title_coverage(query_terms, doc_id)
        return {"bm25": bm25, "title": title, "rating": float(self.rating[doc_id]),
                "discount": float(self.discount[doc_id]), "stock": float(self.stock[doc_id]),
                "score": bm25 * (1.0 + self.params.title_boost * title) * float(self.offer[doc_id])}

    def score(self, query_terms: Sequence[str], candidates: Sequence[int]) -> Ranking:
        """
        :param query_terms: pre-processed query terms
        :param candidates: doc ids to score (the documents returned by the AND)
        :return: [(doc_id, score)] sorted by decreasing score
        """
        unique = set(query_terms)
        scored = []
        for doc_id, bm25 in self.bm25.score(query_terms, candidates):
            title = len(unique & self.title_sets[doc_id]) / len(unique) if unique else 0.0
            scored.append((doc_id, bm25 * (1.0 + self.params.title_boost * title) * float(self.offer[doc_id])))
        return _sort(scored)


# ------------------------------------------------------------------------------------------ word2vec + cosine

def stable_hash(text: str) -> int:
    """
    Hash that does not change between Python runs. gensim uses Python's `hash()` to initialise the vectors, and
    `hash()` of a string is different in every interpreter session (hash randomisation), so with the default the same
    seed would give a different model each time.
    """
    return zlib.crc32(text.encode("utf-8"))


def train_word2vec(term_lists: Sequence[Sequence[str]], vector_size: int = 100, window: int = 5, min_count: int = 1,
                   sg: int = 0, negative: int = 10, epochs: int = 30, seed: int = 42):
    """
    Train word2vec on the pre-processed documents of the collection (one document = one "sentence").
    A single worker thread and `stable_hash` make the training reproducible.
    :param term_lists: pre-processed terms of every document
    :param vector_size: number of dimensions of the word vectors
    :param window: maximum distance between the target word and a context word
    :param min_count: words with a lower collection frequency are ignored (1 = every term of the index has a vector)
    :param sg: 0 = CBOW, 1 = skip-gram
    :param negative: number of negative samples
    :param epochs: passes over the corpus
    :param seed: random seed
    :return: gensim KeyedVectors (word -> vector)
    """
    from gensim.models import Word2Vec
    model = Word2Vec(sentences=[list(terms) for terms in term_lists], vector_size=vector_size, window=window,
                     min_count=min_count, sg=sg, negative=negative, epochs=epochs, seed=seed, workers=1,
                     hashfxn=stable_hash)
    return model.wv


def load_or_train_word2vec(path: str, term_lists: Sequence[Sequence[str]], **train_kwargs):
    """
    Load the word vectors stored in `path` (word2vec binary format); if the file does not exist, train them with
    `train_word2vec` and save them. Delete the file to train again.
    :return: (KeyedVectors, True if the vectors were trained in this call)
    """
    from gensim.models import KeyedVectors
    if os.path.exists(path):
        return KeyedVectors.load_word2vec_format(path, binary=True), False
    wv = train_word2vec(term_lists, **train_kwargs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wv.save_word2vec_format(path, binary=True)
    return wv, True


def text_vector(wv, terms: Sequence[str], term_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Vector of a piece of text: the average of the vectors of all its words (a word that appears twice counts twice).
    Words without a vector are skipped; a text without any known word gets the zero vector.
    :param wv: gensim KeyedVectors
    :param terms: pre-processed terms of the text
    :param term_weights: None = plain average (the representation asked in the statement). With a dict
                         {term: weight} the average is weighted (used only for the comparison of section 3).
    :return: array of `wv.vector_size` dimensions
    """
    rows = [wv.key_to_index[t] for t in terms if t in wv.key_to_index]
    if not rows:
        return np.zeros(wv.vector_size, dtype=np.float32)
    vectors = wv.vectors[rows]
    if term_weights is None:
        return vectors.mean(axis=0)
    weights = np.array([term_weights.get(wv.index_to_key[r], 0.0) for r in rows], dtype=np.float32)
    total = float(weights.sum())
    return vectors.mean(axis=0) if total == 0 else (vectors * weights[:, None]).sum(axis=0) / total


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Divide every row by its Euclidean norm (rows of zeros stay zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _cosines(unit_rows: np.ndarray, unit_query: np.ndarray) -> List[float]:
    """
    Cosine between every (already normalised) row and the normalised query. The result is rounded to 6 decimals:
    products with the same text get exactly the same score on any machine, and the tie is then broken by doc id.
    """
    return np.round(unit_rows.astype(np.float64) @ unit_query.astype(np.float64), 6).tolist()


class Word2VecRanker:
    """
    word2vec + cosine. Every document is one vector, the average of the vectors of all its terms; the query is
    represented in the same way, and the score is the cosine of the two vectors:

        v(text) = (1 / |text|) * sum_{w in text} v(w)          score(q, d) = cos(v(q), v(d))

    The document vectors are computed once, when the ranker is created, and stored normalised, so scoring the
    candidates is one matrix-vector product.
    """

    def __init__(self, wv, term_lists: Sequence[Sequence[str]], term_weights: Optional[Dict[str, float]] = None,
                 remove_first_component: bool = False):
        """
        :param wv: gensim KeyedVectors
        :param term_lists: pre-processed terms of every document (same order as the index)
        :param term_weights: None = plain average. {term: weight} = weighted average (comparison of section 3)
        :param remove_first_component: subtract from every vector its projection on the first principal direction
                                       of the document vectors (second step of SIF, Arora et al. 2017)
        """
        self.wv = wv
        self.term_weights = term_weights
        matrix = np.vstack([text_vector(wv, terms, term_weights) for terms in term_lists]).astype(np.float32)
        self.component: Optional[np.ndarray] = None
        if remove_first_component:
            _, _, vt = np.linalg.svd(matrix, full_matrices=False)
            self.component = vt[0]
            matrix = matrix - np.outer(matrix @ self.component, self.component)
        self.doc_vectors = _unit_rows(matrix)

    def query_vector(self, query_terms: Sequence[str]) -> np.ndarray:
        """Vector of the query, built exactly like the document vectors."""
        vector = text_vector(self.wv, query_terms, self.term_weights)
        if self.component is not None:
            vector = vector - (vector @ self.component) * self.component
        return vector

    def score(self, query_terms: Sequence[str], candidates: Sequence[int]) -> Ranking:
        """
        :return: [(doc_id, cosine)] sorted by decreasing cosine (0 for every candidate if the query has no known word)
        """
        candidates = list(candidates)
        if not candidates:
            return []
        query = self.query_vector(query_terms)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return _sort([(doc_id, 0.0) for doc_id in candidates])
        return _sort(zip(candidates, _cosines(self.doc_vectors[candidates], query / norm)))


# ------------------------------------------------------------------- doc2vec (only for the comparison of section 3)

def train_doc2vec(term_lists: Sequence[Sequence[str]], dm: int = 0, vector_size: int = 100, window: int = 5,
                  min_count: int = 1, negative: int = 10, epochs: int = 20, seed: int = 42):
    """
    Train doc2vec (Le & Mikolov, 2014): every document gets its own vector, learned by the model instead of being
    computed as an average of word vectors.
    :param dm: 0 = PV-DBOW (the document vector is trained to predict the words of the document, word order is
               ignored), 1 = PV-DM (the document vector and the context words predict the next word)
    :return: gensim Doc2Vec model; the tag of a document is its position in `term_lists`
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    documents = [TaggedDocument(list(terms), [i]) for i, terms in enumerate(term_lists)]
    return Doc2Vec(documents, dm=dm, vector_size=vector_size, window=window, min_count=min_count,
                   negative=negative, epochs=epochs, seed=seed, workers=1, hashfxn=stable_hash)


class Doc2VecRanker:
    """
    doc2vec + cosine. The document vectors are the ones learned during training. The query is not in the training
    set, so its vector has to be inferred: the model is frozen and a new vector is fitted to the query words with a
    few gradient steps (`infer_vector`).
    """

    def __init__(self, model, infer_epochs: int = 50, seed: int = 42):
        """
        :param model: model returned by `train_doc2vec`
        :param infer_epochs: gradient passes used to infer the vector of a query
        :param seed: seed of the inference
        """
        self.model = model
        self.infer_epochs = infer_epochs
        self.seed = seed
        self.doc_vectors = _unit_rows(np.asarray(model.dv.vectors, dtype=np.float32))

    def query_vector(self, query_terms: Sequence[str]) -> np.ndarray:
        """
        Inferred vector of the query. gensim starts the inference from a vector seeded with Python's `hash()` of the
        text (different in every session) and draws the negative samples from the random state of the model (which
        changes after every call), so by default the same query gets a different vector each time. Both are fixed
        here so that the experiment can be repeated.
        """
        import gensim.models.doc2vec as d2v
        original = d2v.pseudorandom_weak_vector

        def seeded(size, seed_string=None, hashfxn=stable_hash):
            return original(size, seed_string=seed_string, hashfxn=stable_hash)

        d2v.pseudorandom_weak_vector = seeded
        try:
            self.model.random.seed(self.seed)
            return self.model.infer_vector(list(query_terms), epochs=self.infer_epochs)
        finally:
            d2v.pseudorandom_weak_vector = original

    def score(self, query_terms: Sequence[str], candidates: Sequence[int]) -> Ranking:
        candidates = list(candidates)
        if not candidates:
            return []
        query = self.query_vector(query_terms)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return _sort([(doc_id, 0.0) for doc_id in candidates])
        return _sort(zip(candidates, _cosines(self.doc_vectors[candidates], query / norm)))


# --------------------------------------------------------------------------------------------------- pipeline

def build_engines(corpus, preprocess: Callable[[str], List[str]], word2vec_path: Optional[str] = None,
                  index: Optional[InvertedIndex] = None, custom_params: CustomScoreParams = CustomScoreParams(),
                  bm25_k1: float = 1.2, bm25_b: float = 0.75,
                  methods: Sequence[str] = METHODS) -> Dict[str, SearchEngine]:
    """
    One search engine per ranking method. Every engine applies the same pipeline (pre-processing of the query,
    conjunctive AND retrieval) and only the ranking of the matching documents changes:

        engines = build_engines(corpus, pp.preprocess_text, "models/word2vec_fashion.bin")
        engines["bm25"].search("men denim jacket", top_k=20)       # [(pid, score), ...]

    :param corpus: processed corpus of Part 1 (DataFrame with pid, all_terms, title_terms, average_rating,
                   discount and out_of_stock)
    :param preprocess: `preprocess_text` of Part 1
    :param word2vec_path: file with the word vectors (trained and saved there if it does not exist). With None the
                          vectors are trained and not saved
    :param index: inverted index to reuse (built on `all_terms` if None)
    :param custom_params: weights of our score
    :param bm25_k1: k1 of the BM25 engine (term-frequency saturation)
    :param bm25_b: b of the BM25 engine (length normalisation, 0 = none, 1 = full)
    :param methods: subset of METHODS to build
    :return: {method name: SearchEngine}
    """
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown ranking method(s) {unknown}; choose from {METHODS}")
    term_lists = corpus["all_terms"].tolist()
    if index is None:
        index = InvertedIndex(corpus["pid"].tolist(), term_lists)
    rankers = {}
    if "tfidf" in methods:
        rankers["tfidf"] = TfidfRanker(index)
    if "bm25" in methods:
        rankers["bm25"] = Bm25Ranker(index, k1=bm25_k1, b=bm25_b)
    if "custom" in methods:
        rankers["custom"] = CustomRanker(index, corpus["title_terms"].tolist(), corpus["average_rating"].tolist(),
                                         corpus["discount"].tolist(), corpus["out_of_stock"].tolist(), custom_params)
    if "word2vec" in methods:
        if word2vec_path is None:
            wv = train_word2vec(term_lists)
        else:
            wv, _ = load_or_train_word2vec(word2vec_path, term_lists)
        rankers["word2vec"] = Word2VecRanker(wv, term_lists)
    return {name: SearchEngine(index, preprocess, ranker) for name, ranker in rankers.items()}
