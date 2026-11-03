# Inverted index, conjunctive (AND) retrieval and TF-IDF ranking (Part 2).
#
# The documents are the term lists produced by the Part 1 pre-processing (project_progress/part_1/preprocessing.py).
# The same `preprocess_text` function is applied to the queries, so that documents and queries use the same vocabulary.

import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class InvertedIndex:
    """
    Inverted index with term frequencies. Following the vocabulary structure of the practical labs, every term has a
    numeric term id and the index maps term_id -> postings list. A posting is (doc_id, tf), and each postings list is
    sorted by doc_id, which allows to intersect two lists with a linear merge.

        postings[term_id] = [(doc_id_1, tf_1), (doc_id_2, tf_2), ...]

    doc_id is the position of the document in `pids` (an integer is smaller and faster than the pid string).
    """

    def __init__(self, pids: Sequence[str], term_lists: Sequence[Sequence[str]]):
        """
        :param pids: document identifiers
        :param term_lists: pre-processed terms of every document (same order as `pids`)
        """
        if len(pids) != len(term_lists):
            raise ValueError("pids and term_lists must have the same length")
        self.pids: List[str] = list(pids)
        self.num_docs: int = len(self.pids)
        self.term_to_id: Dict[str, int] = {}
        self.postings: Dict[int, List[Tuple[int, int]]] = {}
        self.doc_length: List[int] = [len(terms) for terms in term_lists]

        for doc_id, terms in enumerate(term_lists):
            for term, tf in Counter(terms).items():
                term_id = self.term_to_id.setdefault(term, len(self.term_to_id))
                self.postings.setdefault(term_id, []).append((doc_id, tf))
        # doc ids are visited in increasing order, so every postings list is already sorted.

    # ---- statistics -------------------------------------------------------------------------------------------

    @property
    def vocabulary_size(self) -> int:
        return len(self.term_to_id)

    def document_frequency(self, term: str) -> int:
        """Number of documents that contain `term` (0 if the term is not in the vocabulary)."""
        term_id = self.term_to_id.get(term)
        return 0 if term_id is None else len(self.postings[term_id])

    def idf(self, term: str) -> float:
        """idf(t) = log2(N / df(t)); 0 for a term that is not in the vocabulary."""
        df = self.document_frequency(term)
        return math.log2(self.num_docs / df) if df else 0.0

    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        term_id = self.term_to_id.get(term)
        return [] if term_id is None else self.postings[term_id]

    # ---- retrieval --------------------------------------------------------------------------------------------

    @staticmethod
    def _intersect(a: List[int], b: List[int]) -> List[int]:
        """Intersection of two sorted lists of doc ids with the classic two-pointer merge, O(len(a) + len(b))."""
        i = j = 0
        out = []
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                out.append(a[i])
                i += 1
                j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return out

    def conjunctive(self, terms: Sequence[str]) -> List[int]:
        """
        Conjunctive (AND) query: ids of the documents that contain ALL the query terms.
        The postings lists are intersected from the shortest (rarest term) to the longest, so that the intermediate
        result is as small as possible. If a term is not in the vocabulary, or the query is empty, the result is empty.
        :param terms: pre-processed query terms
        :return: sorted list of doc ids
        """
        unique_terms = list(dict.fromkeys(terms))
        if not unique_terms:
            return []
        lists = [self.get_postings(t) for t in unique_terms]
        if any(len(p) == 0 for p in lists):
            return []
        lists.sort(key=len)
        result = [doc_id for doc_id, _ in lists[0]]
        for postings in lists[1:]:
            result = self._intersect(result, [doc_id for doc_id, _ in postings])
            if not result:
                break
        return result

    def disjunctive_counts(self, terms: Sequence[str]) -> Dict[int, int]:
        """
        For every document that contains at least one query term, the number of distinct query terms it contains.
        Only used for the analysis of the AND semantics (documents that miss exactly one term).
        """
        counts: Dict[int, int] = {}
        for t in dict.fromkeys(terms):
            for doc_id, _ in self.get_postings(t):
                counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts


class TfidfRanker:
    """
    TF-IDF ranking with cosine similarity (vector space model).

        weight(t, d) = (1 + log2 tf(t, d)) * idf(t),     idf(t) = log2(N / df(t))
        weight(t, q) = (1 + log2 tf(t, q)) * idf(t)
        score(q, d)  = sum_t weight(t, q) * weight(t, d) / (||q|| * ||d||)

    The norm ||d|| is computed over ALL the terms of the document (not only the query terms), so a long document that
    contains the query terms among many other terms gets a lower score than a short one that is about the query.
    """

    def __init__(self, index: InvertedIndex):
        self.index = index
        self.doc_norm = self._compute_doc_norms()

    def _doc_weight(self, tf: int, idf: float) -> float:
        return (1.0 + math.log2(tf)) * idf

    def _compute_doc_norms(self) -> List[float]:
        sq = [0.0] * self.index.num_docs
        for term, term_id in self.index.term_to_id.items():
            idf = self.index.idf(term)
            for doc_id, tf in self.index.postings[term_id]:
                sq[doc_id] += self._doc_weight(tf, idf) ** 2
        return [math.sqrt(x) for x in sq]

    def score(self, query_terms: Sequence[str], candidates: Sequence[int]) -> List[Tuple[int, float]]:
        """
        Cosine similarity between the query and every candidate document.
        :param query_terms: pre-processed query terms (repeated terms increase the query tf)
        :param candidates: doc ids to score
        :return: [(doc_id, score)] sorted by decreasing score; ties are broken by doc id (deterministic)
        """
        q_tf = Counter(query_terms)
        q_weight = {t: self._doc_weight(tf, self.index.idf(t)) for t, tf in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weight.values()))
        wanted = set(candidates)
        acc: Dict[int, float] = {doc_id: 0.0 for doc_id in wanted}
        for t, wq in q_weight.items():
            idf = self.index.idf(t)
            for doc_id, tf in self.index.get_postings(t):
                if doc_id in wanted:
                    acc[doc_id] += wq * self._doc_weight(tf, idf)
        scored = []
        for doc_id, dot in acc.items():
            denom = q_norm * self.doc_norm[doc_id]
            scored.append((doc_id, dot / denom if denom > 0 else 0.0))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored


class Bm25Ranker:
    """
    Okapi BM25 (used in the analysis of limitations, as an alternative to cosine TF-IDF).

        score(q, d) = sum_{t in q} idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl)),
        idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

    Length normalisation only changes the saturation of tf, it does not divide the whole score by the document norm.
    Same interface as `TfidfRanker.score`.
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75):
        self.index, self.k1, self.b = index, k1, b
        self.avg_len = sum(index.doc_length) / max(index.num_docs, 1)

    def score(self, query_terms: Sequence[str], candidates: Sequence[int]) -> List[Tuple[int, float]]:
        wanted = set(candidates)
        acc: Dict[int, float] = {doc_id: 0.0 for doc_id in wanted}
        for t in dict.fromkeys(query_terms):
            df = self.index.document_frequency(t)
            if df == 0:
                continue
            idf = math.log2(1.0 + (self.index.num_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in self.index.get_postings(t):
                if doc_id in wanted:
                    norm = self.k1 * (1.0 - self.b + self.b * self.index.doc_length[doc_id] / self.avg_len)
                    acc[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return sorted(acc.items(), key=lambda x: (-x[1], x[0]))


class SearchEngine:
    """
    Conjunctive retrieval + TF-IDF ranking. `preprocess` is the Part 1 function `preprocess_text`, so that the query
    goes through exactly the same steps as the documents.
    """

    def __init__(self, index: InvertedIndex, preprocess: Callable[[str], List[str]], ranker=None):
        """
        :param ranker: object with `score(query_terms, candidates)`; cosine TF-IDF by default
        """
        self.index = index
        self.ranker = ranker if ranker is not None else TfidfRanker(index)
        self.preprocess = preprocess

    def search(self, query: str, top_k: Optional[int] = None, min_match: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        :param query: raw query text
        :param top_k: keep only the best top_k results (None = all the documents that match)
        :param min_match: None (default) = conjunctive AND semantics, every term is required. An integer m returns the
                          documents that contain at least m distinct query terms (m = number of terms - 1 is a "soft
                          AND"); only used in the analysis of limitations.
        :return: [(pid, score)] sorted by decreasing score; empty if no document matches
        """
        terms = self.preprocess(query)
        if min_match is None:
            matches = self.index.conjunctive(terms)
        else:
            matches = sorted(d for d, c in self.index.disjunctive_counts(terms).items() if c >= min_match)
        ranked = self.ranker.score(terms, matches)
        if top_k is not None:
            ranked = ranked[:top_k]
        return [(self.index.pids[doc_id], score) for doc_id, score in ranked]

    def rank_pool(self, query: str, pids: Sequence[str]) -> List[Tuple[str, float]]:
        """
        Score an arbitrary set of documents (for instance a pool of judged documents), without the AND filter.
        :return: [(pid, score)] sorted by decreasing score
        """
        pos = {pid: i for i, pid in enumerate(self.index.pids)}
        terms = self.preprocess(query)
        scored = self.ranker.score(terms, [pos[p] for p in pids])
        return [(self.index.pids[doc_id], score) for doc_id, score in scored]
