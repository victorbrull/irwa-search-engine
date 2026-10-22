"""
Text pre-processing and document normalisation for the fashion products corpus (IRWA Part 1).

The same `preprocess_text` function must be applied to documents (at indexing time) and to
queries (at search time), otherwise query terms will not match index terms.

Pipeline for free text (title, description, ...):
    1. fix_encoding      -> repair mojibake such as "itâ€™s" -> "it's" (ftfy)
    2. normalize_text    -> ASCII-fold, split glued words ("look.Designed", "UnltdSlim"),
                            drop possessives ("women's" -> "women"), unify domain variants
                            ("t-shirt" / "t shirt" / "tshirts" -> "tshirt"), lowercase
    3. tokenize          -> regex tokenizer on [a-z0-9]; this also removes punctuation
    4. filter            -> stop words, 1-char tokens, product/style codes ("1005combo2")
    5. stem              -> Porter stemmer
"""
import re
import unicodedata
from typing import Any, Dict, List, Optional

import ftfy
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

# product_details keys whose values carry no descriptive meaning for retrieval:
# identifiers, washing instructions, packaging and origin (almost constant: "India").
EXCLUDED_DETAIL_KEYS = {
    "Style Code", "Fabric Care", "Sales Package", "Country of Origin",
    "Number of Contents in Sales Package", "Pack of", "Reversible",
}

# Fields returned to the user for every retrieved document (Part 1, point 2).
OUTPUT_FIELDS = [
    "pid", "title", "description", "brand", "category", "sub_category", "product_details",
    "seller", "out_of_stock", "selling_price", "discount", "actual_price", "average_rating", "url",
]

# Text fields that are tokenised and kept as separate fields of the index.
TEXT_FIELDS = ["title", "description", "brand", "category", "sub_category", "product_details", "seller"]

# Fields concatenated into the single "all_text" field (see report, Q3: category and seller are left out).
MERGED_FIELDS = ["title", "description", "brand", "sub_category", "product_details"]

_DOMAIN_SUBSTITUTIONS = [
    (re.compile(r"\bt[\s\-]?shirts?\b"), "tshirt"),
    (re.compile(r"\btees?\b"), "tshirt"),
    (re.compile(r"\bsweat[\s\-]shirts?\b"), "sweatshirt"),
    (re.compile(r"\bwom[ae]n'?s?\b"), "women"),
    (re.compile(r"\bm[ae]n'?s\b"), "men"),
    (re.compile(r"\bregulor\b"), "regular"),
]
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CODE_RE = re.compile(r"^(?=.*\d)(?=.*[a-z])[a-z0-9]+$")  # mixed letters + digits, e.g. "1005combo2"


def fix_encoding(text: str) -> str:
    """
    Repair badly decoded UTF-8 (mojibake), e.g. "itâ€™s" -> "it’s". A small number of descriptions
    (~0.3%) were decoded with the wrong encoding (sometimes twice) during crawling.
    :param text: raw text
    :return: text with encoding errors fixed
    """
    return ftfy.fix_text(text)


def normalize_text(text: str) -> str:
    """
    Normalise raw text before tokenisation.
    - Fold accents / typographic quotes to ASCII ("’" -> "'", "é" -> "e").
    - Insert a space where words are glued together by punctuation without a space
      ("look.Designed", "Comfort,Skin") or by a lower->upper case change ("UnltdSlim").
    - Lowercase and apply domain substitutions (possessives, t-shirt variants, frequent typos).
    :param text: text after `fix_encoding`
    :return: normalised lowercase text
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"([a-zA-Z])([.,;:!?/&+])([a-zA-Z])", r"\1\2 \3", text)
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)
    text = text.lower()
    for pattern, replacement in _DOMAIN_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


def tokenize(text: str) -> List[str]:
    """
    Split normalised text into alphanumeric tokens. Every character that is not [a-z0-9] acts as a
    separator, so punctuation is removed in the same step. Hyphenated compounds are split
    ("itch-free" -> "itch", "free"), consistently for documents and queries.
    :param text: normalised lowercase text
    :return: list of tokens
    """
    return _TOKEN_RE.findall(text)


def preprocess_text(text: Optional[str], stem: bool = True, remove_stopwords: bool = True,
                    remove_codes: bool = True) -> List[str]:
    """
    Full pre-processing pipeline for a piece of free text (document field or query).
    :param text: raw text (None / NaN are treated as empty)
    :param stem: apply Porter stemming
    :param remove_stopwords: remove NLTK English stop words
    :param remove_codes: remove product/style codes that mix letters and digits (e.g. "1005combo2")
    :return: list of processed terms
    """
    if text is None or (isinstance(text, float) and pd.isna(text)) or not str(text).strip():
        return []
    tokens = tokenize(normalize_text(fix_encoding(str(text))))
    terms = []
    for tok in tokens:
        if len(tok) < 2 and not tok.isdigit():
            continue
        if remove_stopwords and tok in STOP_WORDS:
            continue
        if remove_codes and _CODE_RE.match(tok):
            continue
        terms.append(STEMMER.stem(tok) if stem else tok)
    return terms


def flatten_product_details(details: Any, include_keys: bool = False) -> str:
    """
    Turn the product_details list of one-entry dicts into plain text.
    Only values are kept by default: keys such as "Fabric" or "Pattern" occur in almost every document
    and would add near-zero-IDF terms. Non-descriptive keys (EXCLUDED_DETAIL_KEYS) are dropped.
    :param details: list of {key: value} dicts (raw format) or a single dict
    :param include_keys: also emit the key names
    :return: text such as "Elastic. Side Pockets. Cotton Blend. Solid. Multicolor"
    """
    merged = details_to_dict(details)
    parts = []
    for key, value in merged.items():
        if key in EXCLUDED_DETAIL_KEYS or value is None or str(value).strip() == "":
            continue
        parts.append(f"{key} {value}" if include_keys else str(value))
    return ". ".join(parts)


def details_to_dict(details: Any) -> Dict[str, str]:
    """
    Convert product_details from the raw list-of-dicts format into a single dict.
    :param details: raw product_details value
    :return: dict {key: value}
    """
    if isinstance(details, dict):
        return dict(details)
    merged = {}
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                merged.update(item)
    return merged


def brand_from_url(brand: Optional[str], url: Optional[str]) -> Optional[str]:
    """
    Recover the full brand name. In this dataset `brand` is frequently truncated ("ECKO Unl", "Pu",
    "True Bl") while the Flipkart URL slug starts with the complete brand ("ecko-unltd-...").
    If the (non-empty) truncated brand is a prefix of the slug, the brand is extended to the next
    slug word boundary. Otherwise the original value is returned unchanged.
    NOTE: only the brand is taken from the URL; the rest of the slug is NOT used because its gender
    words disagree with title / "Ideal For" in ~50% of the products.
    :param brand: brand field
    :param url: url field
    :return: full brand name (lowercase, words separated by spaces) or the original brand
    """
    if not brand or not url:
        return brand or None
    match = re.search(r"flipkart\.com/([^/]+)/p/", url)
    if not match:
        return brand
    slug_words = match.group(1).split("-")
    target = re.sub(r"\W", "", brand.lower())
    built = ""
    for i, word in enumerate(slug_words):
        built += word
        if built.startswith(target):
            return " ".join(slug_words[: i + 1])
        if not target.startswith(built):
            break
    return brand


def parse_price(value: Any) -> Optional[float]:
    """
    "2,999" -> 2999.0 ; "" / None -> None
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).replace(",", "").strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def parse_discount(value: Any) -> Optional[float]:
    """
    "69% off" -> 69.0 ; "" / None -> None
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else None


def parse_rating(value: Any) -> Optional[float]:
    """
    "3.9" -> 3.9 ; "" / None -> None. A missing rating is kept as None (unrated), NOT 0.
    """
    return parse_price(value)


def preprocess_document(raw: Dict[str, Any], **text_kwargs) -> Dict[str, Any]:
    """
    Build the processed representation of one product.

    The output keeps:
      - every OUTPUT_FIELD in clean, typed form (what will be shown to the user),
      - one token list per text field ("<field>_terms") for field-aware indexing / weighting,
      - "all_terms": the concatenation of MERGED_FIELDS for single-field indexing,
      - numeric / boolean fields parsed as numbers (they are NOT indexed as text).
    :param raw: one record of fashion_products_dataset.json
    :param text_kwargs: options forwarded to `preprocess_text`
    :return: processed document dict
    """
    details = details_to_dict(raw.get("product_details"))
    actual = parse_price(raw.get("actual_price"))
    selling = parse_price(raw.get("selling_price"))
    discount = parse_discount(raw.get("discount"))
    if discount is None and actual and selling is not None and actual > 0:
        discount = round(100 * (actual - selling) / actual)
    if actual is None and selling is not None and discount == 0:
        actual = selling

    doc = {
        "pid": raw["pid"],
        "title": fix_encoding(raw.get("title") or ""),
        "description": fix_encoding(raw.get("description") or ""),
        "brand": raw.get("brand") or None,
        "brand_full": brand_from_url(raw.get("brand"), raw.get("url")),
        "category": raw.get("category") or None,
        "sub_category": raw.get("sub_category") or None,
        "product_details": details,
        "seller": raw.get("seller") or None,
        "out_of_stock": bool(raw.get("out_of_stock", False)),
        "selling_price": selling,
        "discount": discount,
        "actual_price": actual,
        "average_rating": parse_rating(raw.get("average_rating")),
        "url": raw.get("url"),
    }

    field_text = {
        "title": doc["title"],
        "description": doc["description"],
        "brand": doc["brand_full"],
        "category": doc["category"],
        "sub_category": doc["sub_category"],
        "product_details": flatten_product_details(details),
        "seller": doc["seller"],
    }
    for field in TEXT_FIELDS:
        doc[f"{field}_terms"] = preprocess_text(field_text[field], **text_kwargs)
    doc["all_terms"] = [t for field in MERGED_FIELDS for t in doc[f"{field}_terms"]]
    return doc


def preprocess_corpus(df: pd.DataFrame, **text_kwargs) -> pd.DataFrame:
    """
    Apply `preprocess_document` to every row of the raw corpus.
    :param df: raw corpus as loaded by pd.read_json
    :param text_kwargs: options forwarded to `preprocess_text`
    :return: DataFrame with one processed document per row, indexed by pid
    """
    records = [preprocess_document(row, **text_kwargs) for row in df.to_dict(orient="records")]
    processed = pd.DataFrame(records)
    processed.index = processed["pid"].values  # lookup by pid, while keeping pid as a normal column
    return processed
