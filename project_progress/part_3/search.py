# Command-line search with the ranking methods of Part 3.
#
#   python search.py "men denim jacket"                          # TF-IDF + cosine, top 20
#   python search.py "men denim jacket" --method bm25 --top 10
#   python search.py "men denim jacket" --method custom --rating-weight 1.0
#   python search.py "men denim jacket" --method all             # the four rankings, one after the other
#
# Pipeline: the query is pre-processed with the Part 1 function, the documents that contain ALL the query terms are
# retrieved with the inverted index of Part 2 (conjunctive query) and they are sorted with the selected method.

import argparse
import os
import sys

import nltk
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "fashion_products_dataset.json")
CORPUS_PATH = os.path.join(ROOT, "data", "processed_corpus.pkl")          # written by the Part 1 notebook
WORD2VEC_PATH = os.path.join(HERE, "models", "word2vec_fashion.bin")

try:
    nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

sys.path.insert(0, HERE)
import ranking as rk  # noqa: E402  (also makes the Part 1 and Part 2 modules importable)
import preprocessing as pp  # noqa: E402  (Part 1)


def load_corpus() -> pd.DataFrame:
    """
    Processed corpus of Part 1. If the pickle written by the Part 1 notebook is missing, it is created again from
    the raw dataset with the Part 1 pre-processing (~40 s).
    """
    if os.path.exists(CORPUS_PATH):
        return pd.read_pickle(CORPUS_PATH)
    if not os.path.exists(DATA_PATH):
        sys.exit(f"dataset not found: put fashion_products_dataset.json in {os.path.dirname(DATA_PATH)}")
    corpus = pp.preprocess_corpus(pd.read_json(DATA_PATH))
    corpus.to_pickle(CORPUS_PATH)
    return corpus


def results_table(corpus: pd.DataFrame, results) -> pd.DataFrame:
    """
    :param corpus: processed corpus (indexed by pid)
    :param results: [(pid, score)] returned by a search engine
    :return: one row per result with the fields shown to the user
    """
    rows = []
    for rank, (pid, score) in enumerate(results, start=1):
        doc = corpus.loc[pid]
        rows.append({"rank": rank, "score": round(score, 4), "pid": pid, "title": doc["title"],
                     "brand": doc["brand"] or "", "price": doc["selling_price"], "discount": doc["discount"],
                     "rating": doc["average_rating"], "in_stock": not doc["out_of_stock"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conjunctive (AND) search over the fashion products, ranked with "
                                                 "the selected method.")
    parser.add_argument("query", help="query text, for example \"men denim jacket\"")
    parser.add_argument("--method", choices=rk.METHODS + ("all",), default="tfidf",
                        help="ranking method: tfidf (TF-IDF + cosine), bm25, custom (our score), word2vec "
                             "(word2vec + cosine) or all (default: tfidf)")
    parser.add_argument("--top", type=int, default=20, help="number of results to show (default: 20)")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1, term-frequency saturation (default: 1.2)")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b, length normalisation (default: 0.75)")
    defaults = rk.CustomScoreParams()
    parser.add_argument("--title-boost", type=float, default=defaults.title_boost,
                        help=f"our score: weight of the query terms found in the title (default: {defaults.title_boost})")
    parser.add_argument("--rating-weight", type=float, default=defaults.rating_weight,
                        help=f"our score: weight of the average rating (default: {defaults.rating_weight})")
    parser.add_argument("--discount-weight", type=float, default=defaults.discount_weight,
                        help=f"our score: weight of the discount (default: {defaults.discount_weight})")
    parser.add_argument("--out-of-stock-factor", type=float, default=defaults.out_of_stock_factor,
                        help="our score: factor applied to the products that are out of stock "
                             f"(default: {defaults.out_of_stock_factor})")
    args = parser.parse_args()

    corpus = load_corpus()
    methods = rk.METHODS if args.method == "all" else (args.method,)
    # our score uses the same BM25 parameters as the BM25 engine
    params = rk.CustomScoreParams(title_boost=args.title_boost, rating_weight=args.rating_weight,
                                  discount_weight=args.discount_weight,
                                  out_of_stock_factor=args.out_of_stock_factor, k1=args.k1, b=args.b)
    engines = rk.build_engines(corpus, pp.preprocess_text, WORD2VEC_PATH, custom_params=params,
                               bm25_k1=args.k1, bm25_b=args.b, methods=methods)

    terms = pp.preprocess_text(args.query)
    matches = len(next(iter(engines.values())).index.conjunctive(terms))
    print(f"query: {args.query!r} -> terms {terms} -> {matches} documents contain all the terms")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    for method in methods:
        print(f"\n== {rk.METHOD_NAMES[method]} (top {args.top}) ==")
        table = results_table(corpus, engines[method].search(args.query, top_k=args.top))
        print(table.to_string(index=False) if len(table) else "no document contains all the query terms")


if __name__ == "__main__":
    main()
