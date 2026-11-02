# Relevance judgments for the five test queries of Part 2 (binary labels: 1 = relevant, 0 = not relevant).
#
# We act as the expert judges of the collection: a product is relevant when it satisfies EVERY attribute that the
# query asks for, judged from what a shopper sees (title and product details), NOT from whether the stemmed query terms
# appear somewhere in the text. The guidelines below are applied by `judge`, which returns the label and the reason, so
# that all the documents are judged with the same criteria and every label can be audited. The cases that the rules
# do not decide by themselves were read one by one (see MANUAL_OVERRIDES).

import re
from typing import Dict, Optional, Tuple

import pandas as pd

QUERIES = {
    1: "men black track pants polyester",
    2: "women black v neck tshirt",
    3: "men grey hooded sweatshirt",
    4: "women striped shirt",
    5: "men denim jacket",
}

GUIDELINES = {
    1: "Track pants (not a track suit) for men, black (also if black is one of several colours), made of "
       "polyester (blends and polycotton count; cotton-only or cotton blend do not).",
    2: "T-shirt with a V neck for women, black (also if black is one of several colours). T-shirts have no Color detail, so the colour is read from the title.",
    3: "Sweatshirt for men with a hood, grey (also if grey is one of several colours; sleeveless and half-sleeve "
       "hooded sweatshirts count because the query does not ask for sleeves).",
    4: "Shirt (not T-shirt, vest, sweater, or unstitched shirt fabric) for women with a striped pattern "
       "(also if the pattern list contains other patterns).",
    5: "Jacket for men made of denim: 'denim' in the title, in the fabric or in the jacket type "
       "(denim blend counts). Other jackets and other denim products (jeans, shirts) do not.",
}

# Documents whose label was decided by reading the record, because the rules gave a wrong or doubtful result.
# {(query_id, pid): (label, reason)}  -- filled in after reviewing the output of `judge` (see the notebook).
MANUAL_OVERRIDES: Dict[Tuple[int, str], Tuple[int, str]] = {}


# ---------------------------------------------------------------------------------------------- helpers

def _detail(rec, key: str) -> str:
    details = rec["product_details"] if isinstance(rec["product_details"], dict) else {}
    return str(details.get(key, "")).lower()


def _title(rec) -> str:
    return str(rec["title"]).lower()


def _color(rec) -> str:
    """Colour(s) of the product: the Color detail, or the title when the product has no Color detail (T-shirts)."""
    return _detail(rec, "Color") or _title(rec)


def gender(rec) -> Optional[str]:
    """Gender of the product from the title (the Ideal For detail is the fallback). 'women' is tested first."""
    title = _title(rec)
    if re.search(r"\bwomen\b|\bwomens\b|\bwomen's\b", title):
        return "women"
    if re.search(r"\bmen\b|\bmens\b|\bmen's\b", title):
        return "men"
    ideal = _detail(rec, "Ideal For")
    if "women" in ideal:
        return "women"
    if "men" in ideal:
        return "men"
    return None


def _first_failure(checks):
    """checks = [(passed, reason_if_failed), ...] -> (0, reason) for the first failed check, else None."""
    for passed, reason in checks:
        if not passed:
            return 0, reason
    return None


# ----------------------------------------------------------------------------------------------- one rule per query

def _judge_1(rec):
    title, color, fabric = _title(rec), _color(rec), _detail(rec, "Fabric")
    failed = _first_failure([
        (gender(rec) == "men", f"not for men ({gender(rec)})"),
        ("track pant" in title, "not a track pant"),
        (bool(color), "colour not stated"),
        ("black" in color, f"colour is not black ({color})"),
        (bool(fabric), "fabric not stated"),
        ("polyester" in fabric or "polycotton" in fabric, f"fabric is not polyester ({fabric})"),
    ])
    if failed:
        return failed
    note = "; polycotton counted as polyester" if "polycotton" in fabric and "polyester" not in fabric else ""
    note += f"; black among several colours ({color})" if "," in color else ""
    return 1, "meets all the criteria" + note


def _judge_2(rec):
    title, color = _title(rec), _color(rec)
    v_neck = (re.search(r"\bv[ -]neck\b", title) is not None
              or any(re.search(r"\bv[ -]neck\b", _detail(rec, k)) for k in ("Type", "Neck")))
    failed = _first_failure([
        (gender(rec) == "women", f"not for women ({gender(rec)})"),
        ("t-shirt" in title, "not a T-shirt"),
        (v_neck, "not a V neck"),
        ("black" in color, "colour is not black"),
    ])
    if failed:
        return failed
    return 1, "meets all the criteria" + ("; black among several colours" if "," in color.split("t-shirt")[0] else "")


def _judge_3(rec):
    title, color = _title(rec), _color(rec)
    hooded = (_detail(rec, "Hooded") == "yes" or "hood" in _detail(rec, "Neck") or "hood" in _detail(rec, "Type")
              or "hood" in title)
    failed = _first_failure([
        (gender(rec) == "men", f"not for men ({gender(rec)})"),
        ("sweatshirt" in title, "not a sweatshirt"),
        (hooded, "no hood"),
        (bool(color), "colour not stated"),
        ("grey" in color or "gray" in color, f"colour is not grey ({color})"),
    ])
    if failed:
        return failed
    return 1, "meets all the criteria" + (f"; grey among several colours ({color})" if "," in color else "")


def _judge_4(rec):
    title, pattern = _title(rec), _detail(rec, "Pattern")
    is_shirt = bool(re.search(r"\bshirt\b", title)) and "t-shirt" not in title and "fabric" not in title \
        and "fabric" not in _detail(rec, "Generic Name") and "fabric" not in _detail(rec, "Type")
    kind = "T-shirt" if "t-shirt" in title else ("shirt fabric" if "fabric" in title else "other garment")
    failed = _first_failure([
        (gender(rec) == "women", f"not for women ({gender(rec)})"),
        (is_shirt, f"not a shirt ({kind})"),
        ("stripe" in pattern or "stripe" in title, f"not striped ({pattern or 'pattern not stated'})"),
    ])
    return failed or (1, "meets all the criteria")


def _judge_5(rec):
    title, fabric, jtype = _title(rec), _detail(rec, "Fabric"), _detail(rec, "Type")
    failed = _first_failure([
        (gender(rec) == "men", f"not for men ({gender(rec)})"),
        (bool(re.search(r"\bjacket\b", title)), "not a jacket"),
        ("denim" in title or "denim" in fabric or "denim" in jtype, f"not denim (fabric: {fabric or 'not stated'})"),
    ])
    return failed or (1, "meets all the criteria")


_JUDGES = {1: _judge_1, 2: _judge_2, 3: _judge_3, 4: _judge_4, 5: _judge_5}


def judge(query_id: int, rec) -> Tuple[int, str]:
    """
    Binary relevance judgment of one product for one test query.
    :param query_id: 1..5
    :param rec: row of the processed corpus (needs title, product_details and brand_full)
    :return: (label, reason). Manual overrides take precedence over the rules.
    """
    override = MANUAL_OVERRIDES.get((query_id, rec["pid"]))
    return override if override else _JUDGES[query_id](rec)
